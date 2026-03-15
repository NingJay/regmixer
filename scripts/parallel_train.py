#!/usr/bin/env python
"""Parallel regmixer train executor driven by an external control plane."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from regmixer.controlplane import (
    DEFAULT_REMOTE_CONDA_ENV,
    DEFAULT_REMOTE_SHELL_INIT,
    WorkerSlot,
    build_remote_bash_command,
    build_ssh_command,
    read_slot_plan_file,
    resolve_path,
    validate_cluster_mode_paths,
)

DEFAULT_PASSTHROUGH_ENV = "WANDB_MODE,REGMIXER_DISABLE_SWANLAB_SYNC"


@dataclass(frozen=True)
class WorkerConfig:
    scheduler_mode: str
    workdir: str
    python_bin: str
    variant_script: str
    config_path: str
    mix_file: str
    group_id: str
    run_name_prefix: str
    log_dir: str
    summary_dir: str
    pythonpath: str
    output_root_dir: Optional[str] = None
    beaker_user: Optional[str] = None
    global_batch_size: Optional[int] = None
    remote_workdir: Optional[str] = None
    remote_conda_env: str = DEFAULT_REMOTE_CONDA_ENV
    remote_shell_init: str = DEFAULT_REMOTE_SHELL_INIT
    ssh_connect_timeout: int = 5
    passthrough_env: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class TaskSpec:
    mix_index: int
    run_name: str
    log_path: str
    summary_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regmixer mixes over an explicit slot plan.")
    parser.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    parser.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    parser.add_argument("--group-id", required=True, help="Group id passed to run_local_variant.py.")
    parser.add_argument(
        "--run-name-prefix",
        default="parquet-e2e-fit-train",
        help="Run name prefix. Full name is '<prefix>-<mix_index:04d>'.",
    )
    parser.add_argument("--mix-start", type=int, default=0, help="First mix index (inclusive).")
    parser.add_argument("--mix-end", type=int, default=11, help="Last mix index (inclusive).")
    parser.add_argument(
        "--scheduler-mode",
        choices=("local", "cluster"),
        required=True,
        help="Execution mode for the assigned slots.",
    )
    parser.add_argument(
        "--slots-file",
        required=True,
        help="JSON slot plan emitted by scripts/control_plane.py.",
    )
    parser.add_argument("--workdir", default=".", help="Working directory for subprocess execution.")
    parser.add_argument(
        "--remote-workdir",
        default=None,
        help="Shared remote working directory. Defaults to --workdir in cluster mode.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable for run_local_variant.py. Defaults to local sys.executable or remote 'python'.",
    )
    parser.add_argument(
        "--variant-script",
        default="scripts/run_local_variant.py",
        help="Path to run_local_variant.py relative to workdir or absolute.",
    )
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH injected to each task subprocess.")
    parser.add_argument("--log-dir", default="outputs/dryfit_fit/logs", help="Directory for per-task logs.")
    parser.add_argument(
        "--summary-dir",
        default="outputs/dryfit_fit/summaries",
        help="Directory for per-task summaries.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental executor state JSON. Defaults to <summary-dir>/../parallel_train_state.json.",
    )
    parser.add_argument(
        "--output-root-dir",
        default=None,
        help="Optional root directory for training artifacts (replaces /tmp defaults).",
    )
    parser.add_argument(
        "--beaker-user",
        default=None,
        help="Optional pass-through to run_local_variant.py --beaker-user.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Optional pass-through to run_local_variant.py --global-batch-size.",
    )
    parser.add_argument(
        "--remote-conda-env",
        default=DEFAULT_REMOTE_CONDA_ENV,
        help="Conda env activated before remote execution in cluster mode.",
    )
    parser.add_argument(
        "--remote-shell-init",
        default=DEFAULT_REMOTE_SHELL_INIT,
        help="Shell init script sourced before conda activation in cluster mode.",
    )
    parser.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=5,
        help="Per-host SSH connect timeout in seconds.",
    )
    parser.add_argument(
        "--passthrough-env",
        default=DEFAULT_PASSTHROUGH_ENV,
        help="Comma-separated env vars copied into remote task commands when present locally.",
    )
    return parser.parse_args()


def build_mix_indices(start: int, end: int) -> List[int]:
    if start < 0:
        raise ValueError(f"mix-start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"mix-end({end}) must be >= mix-start({start})")
    return list(range(start, end + 1))


def resolve_python_bin(scheduler_mode: str, python_bin: Optional[str]) -> str:
    if python_bin:
        return python_bin
    if scheduler_mode == "cluster":
        return "python"
    return sys.executable


def default_state_file(summary_dir: str) -> str:
    return str(Path(summary_dir).parent / "parallel_train_state.json")


def build_command(cfg: WorkerConfig, mix_index: int, run_name: str, summary_out: str) -> List[str]:
    cmd: List[str] = [
        cfg.python_bin,
        cfg.variant_script,
        "--config",
        cfg.config_path,
        "--mix-file",
        cfg.mix_file,
        "--mix-index",
        str(mix_index),
        "--run-name",
        run_name,
        "--group-id",
        cfg.group_id,
        "--summary-out",
        summary_out,
    ]
    if cfg.output_root_dir:
        cmd.extend(["--output-root-dir", cfg.output_root_dir])
    if cfg.beaker_user:
        cmd.extend(["--beaker-user", cfg.beaker_user])
    if cfg.global_batch_size is not None:
        cmd.extend(["--global-batch-size", str(cfg.global_batch_size)])
    return cmd


def build_task_env(cfg: WorkerConfig, slot: WorkerSlot) -> Dict[str, str]:
    env = {
        "PYTHONPATH": cfg.pythonpath,
        "CUDA_VISIBLE_DEVICES": str(slot.gpu_id),
        "MASTER_PORT": str(29541 + slot.gpu_id),
        "TORCH_DISTRIBUTED_DEFAULT_PORT": str(29542 + slot.gpu_id),
    }
    if cfg.passthrough_env:
        env.update(cfg.passthrough_env)
    return env


def build_remote_task_command(cfg: WorkerConfig, slot: WorkerSlot, task: TaskSpec) -> Tuple[str, str]:
    remote_workdir = cfg.remote_workdir or cfg.workdir
    env = build_task_env(cfg, slot)
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    task_cmd = shlex.join(build_command(cfg, task.mix_index, task.run_name, task.summary_path))
    init_script = cfg.remote_shell_init
    if not init_script.startswith(("~", "$")):
        init_script = shlex.quote(init_script)
    body = (
        "set -euo pipefail && "
        f"source {init_script} && "
        f"conda activate {shlex.quote(cfg.remote_conda_env)} && "
        f"cd {shlex.quote(remote_workdir)} && "
        f"{env_prefix} {task_cmd}"
    )
    return body, task_cmd


def build_task_spec(cfg: WorkerConfig, mix_index: int) -> TaskSpec:
    idx = f"{mix_index:04d}"
    run_name = f"{cfg.run_name_prefix}-{idx}"
    log_path = str(Path(cfg.log_dir) / f"{run_name}.log")
    summary_path = str(Path(cfg.summary_dir) / f"{run_name}.json")
    return TaskSpec(mix_index=mix_index, run_name=run_name, log_path=log_path, summary_path=summary_path)


def build_started_event(slot: WorkerSlot, task: TaskSpec) -> Dict[str, Any]:
    return {
        "event": "started",
        "mix_index": task.mix_index,
        "run_name": task.run_name,
        "host": slot.host,
        "gpu_id": slot.gpu_id,
        "gpu_uuid": slot.gpu_uuid,
        "log_path": task.log_path,
        "summary_path": task.summary_path,
        "started_at": time.time(),
    }


def run_single_task(slot: WorkerSlot, task: TaskSpec, cfg: WorkerConfig) -> Dict[str, Any]:
    log_path = Path(task.log_path)
    summary_path = Path(task.summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_task_env(cfg, slot)
    task_cmd = shlex.join(build_command(cfg, task.mix_index, task.run_name, task.summary_path))
    exec_cmd: List[str]
    proc_env: Optional[Dict[str, str]]
    cwd: Optional[str]

    if cfg.scheduler_mode == "cluster":
        remote_body, task_cmd = build_remote_task_command(cfg, slot, task)
        exec_cmd = build_ssh_command(slot.host, build_remote_bash_command(remote_body), cfg.ssh_connect_timeout)
        proc_env = None
        cwd = None
    else:
        exec_cmd = build_command(cfg, task.mix_index, task.run_name, task.summary_path)
        proc_env = os.environ.copy()
        proc_env.update(env)
        cwd = cfg.workdir

    start_ts = time.time()
    return_code = 99
    error_message = ""

    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(
                f"[train_executor] mode={cfg.scheduler_mode} host={slot.host} gpu_id={slot.gpu_id} "
                f"mix_index={task.mix_index}\n"
            )
            if slot.gpu_uuid:
                logf.write(f"[train_executor] gpu_uuid={slot.gpu_uuid}\n")
            logf.write(f"[train_executor] task_command={task_cmd}\n")
            logf.write(
                "[train_executor] env "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                f"MASTER_PORT={env['MASTER_PORT']} "
                f"TORCH_DISTRIBUTED_DEFAULT_PORT={env['TORCH_DISTRIBUTED_DEFAULT_PORT']}\n"
            )
            if cfg.scheduler_mode == "cluster":
                logf.write(f"[train_executor] ssh_command={shlex.join(exec_cmd)}\n")
            logf.flush()
            proc = subprocess.run(
                exec_cmd,
                cwd=cwd,
                env=proc_env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return_code = int(proc.returncode)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        error_message = f"{type(exc).__name__}: {exc}"
        try:
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"[train_executor] exception={error_message}\n")
        except OSError:
            pass

    duration_sec = round(time.time() - start_ts, 2)
    return {
        "event": "completed",
        "mix_index": task.mix_index,
        "gpu_id": slot.gpu_id,
        "gpu_uuid": slot.gpu_uuid,
        "host": slot.host,
        "run_name": task.run_name,
        "log_path": task.log_path,
        "summary_path": task.summary_path,
        "return_code": return_code,
        "duration_sec": duration_sec,
        "error": error_message,
        "completed_at": time.time(),
    }


def worker_loop(
    slot: WorkerSlot,
    task_queue: "mp.Queue[Optional[int]]",
    event_queue: "mp.Queue[Dict[str, Any]]",
    cfg: WorkerConfig,
) -> None:
    while True:
        mix_index = task_queue.get()
        if mix_index is None:
            return
        task = build_task_spec(cfg, mix_index)
        event_queue.put(build_started_event(slot, task))
        event_queue.put(run_single_task(slot, task, cfg))


def build_passthrough_env(raw: str) -> Dict[str, str]:
    passthrough: Dict[str, str] = {}
    for key in [item.strip() for item in raw.split(",") if item.strip()]:
        if key in os.environ:
            passthrough[key] = os.environ[key]
    return passthrough


def initialize_state(
    cfg: WorkerConfig,
    slots: Sequence[WorkerSlot],
    mix_indices: Sequence[int],
    scan_errors: Dict[str, str],
    slots_file: str,
) -> Dict[str, Any]:
    return {
        "executor": "parallel_train.py",
        "scheduler_mode": cfg.scheduler_mode,
        "workdir": cfg.workdir,
        "config_path": cfg.config_path,
        "mix_file": cfg.mix_file,
        "group_id": cfg.group_id,
        "mix_indices": list(mix_indices),
        "slot_plan_file": slots_file,
        "scan_errors": scan_errors,
        "slots": [asdict(slot) for slot in slots],
        "tasks": {},
        "updated_at": time.time(),
    }


def apply_event_to_state(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    task_key = str(event["mix_index"])
    tasks = state.setdefault("tasks", {})
    task_state = tasks.setdefault(task_key, {"mix_index": event["mix_index"]})

    if event["event"] == "started":
        task_state.update(
            {
                "run_name": event["run_name"],
                "host": event["host"],
                "gpu_id": event["gpu_id"],
                "gpu_uuid": event.get("gpu_uuid"),
                "log_path": event["log_path"],
                "summary_path": event["summary_path"],
                "status": "running",
                "started_at": event["started_at"],
            }
        )
    elif event["event"] == "completed":
        task_state.update(
            {
                "run_name": event["run_name"],
                "host": event["host"],
                "gpu_id": event["gpu_id"],
                "gpu_uuid": event.get("gpu_uuid"),
                "log_path": event["log_path"],
                "summary_path": event["summary_path"],
                "return_code": event["return_code"],
                "duration_sec": event["duration_sec"],
                "error": event["error"],
                "completed_at": event["completed_at"],
                "status": "succeeded" if int(event["return_code"]) == 0 else "failed",
            }
        )
    else:
        raise ValueError(f"unsupported event type: {event['event']}")

    state["updated_at"] = time.time()


def write_state_file(path: str, state: Dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def main() -> int:
    args = parse_args()
    mix_indices = build_mix_indices(args.mix_start, args.mix_end)
    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)

    resolved_log_dir = resolve_path(args.log_dir, workdir)
    resolved_summary_dir = resolve_path(args.summary_dir, workdir)
    state_file = resolve_path(args.state_file, workdir) if args.state_file else default_state_file(resolved_summary_dir)
    slots_file = resolve_path(args.slots_file, workdir)
    slot_plan = read_slot_plan_file(slots_file)
    slots = list(slot_plan.slots)

    if not slots:
        raise RuntimeError(f"slot plan contains no schedulable slots: {slots_file}")
    if slot_plan.scheduler_mode != args.scheduler_mode:
        raise ValueError(
            f"slot plan mode mismatch: slots_file={slot_plan.scheduler_mode} executor={args.scheduler_mode}"
        )

    cfg = WorkerConfig(
        scheduler_mode=args.scheduler_mode,
        workdir=workdir,
        python_bin=resolve_python_bin(args.scheduler_mode, args.python_bin),
        variant_script=resolve_path(args.variant_script, workdir),
        config_path=resolve_path(args.config, workdir),
        mix_file=resolve_path(args.mix_file, workdir),
        group_id=args.group_id,
        run_name_prefix=args.run_name_prefix,
        log_dir=resolved_log_dir,
        summary_dir=resolved_summary_dir,
        pythonpath=args.pythonpath,
        output_root_dir=resolve_path(args.output_root_dir, workdir) if args.output_root_dir else None,
        beaker_user=args.beaker_user,
        global_batch_size=args.global_batch_size,
        remote_workdir=args.remote_workdir or workdir,
        remote_conda_env=args.remote_conda_env,
        remote_shell_init=args.remote_shell_init,
        ssh_connect_timeout=args.ssh_connect_timeout,
        passthrough_env=build_passthrough_env(args.passthrough_env),
    )

    print(
        f"[train_executor] mode={cfg.scheduler_mode} workers={len(slots)} "
        f"requested_workers={slot_plan.requested_workers} mixes={len(mix_indices)}"
    )
    if cfg.scheduler_mode == "cluster":
        print("[train_executor] slots=" + ",".join(slot.label for slot in slots))

    state = initialize_state(cfg, slots, mix_indices, slot_plan.scan_errors, slots_file)
    write_state_file(state_file, state)

    task_queue: "mp.Queue[Optional[int]]" = mp.Queue()
    event_queue: "mp.Queue[Dict[str, Any]]" = mp.Queue()

    for mix_index in mix_indices:
        task_queue.put(mix_index)
    for _ in slots:
        task_queue.put(None)

    workers: List[mp.Process] = []
    for slot in slots:
        proc = mp.Process(target=worker_loop, args=(slot, task_queue, event_queue, cfg), daemon=False)
        proc.start()
        workers.append(proc)

    results: List[Dict[str, Any]] = []
    completed = 0

    try:
        while completed < len(mix_indices):
            event = event_queue.get()
            apply_event_to_state(state, event)
            write_state_file(state_file, state)

            if event["event"] != "completed":
                continue

            completed += 1
            results.append(event)
            ok = int(event["return_code"]) == 0
            status = "OK" if ok else "FAIL"
            print(
                f"[{status}] mix={event['mix_index']:04d} host={event['host']} gpu={event['gpu_id']} "
                f"rc={event['return_code']} log={event['log_path']}"
            )
    except KeyboardInterrupt:
        state["interrupted"] = True
        state["updated_at"] = time.time()
        write_state_file(state_file, state)
        raise
    finally:
        for proc in workers:
            proc.join()

    results.sort(key=lambda item: int(item["mix_index"]))
    failed = [result for result in results if int(result["return_code"]) != 0]
    report = {
        "executor": "parallel_train.py",
        "scheduler_mode": cfg.scheduler_mode,
        "slot_plan_file": slots_file,
        "state_file": state_file,
        "total": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "failed_mix_indices": [int(result["mix_index"]) for result in failed],
        "slots": [slot.label for slot in slots],
        "scan_errors": slot_plan.scan_errors,
        "failures": failed,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
