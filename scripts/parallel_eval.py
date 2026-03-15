#!/usr/bin/env python
"""Parallel OLMES eval executor driven by an external control plane."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, replace
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

DEFAULT_PASSTHROUGH_ENV = "HF_HOME,HF_HUB_CACHE,HF_TOKEN,TRANSFORMERS_CACHE,XDG_CACHE_HOME"
DEFAULT_MAX_RETRIES = 1

DEFAULT_OLMES_TASKS: Tuple[str, ...] = (
    "arc:rc::olmes:full",
    "arc:rc:bpb::olmes:full",
    "mmlu:rc::olmes",
    "mmlu:rc:bpb::olmes",
    "boolq:rc::olmes:full",
    "boolq:rc:bpb::olmes:full",
    "csqa:rc::olmes:full",
    "csqa:rc:bpb::olmes:full",
    "hellaswag:rc::olmes:full",
    "hellaswag:rc:bpb::olmes:full",
    "openbookqa:rc::olmes:full",
    "openbookqa:rc:bpb::olmes:full",
    "piqa:rc::olmes:full",
    "piqa:rc:bpb::olmes:full",
    "socialiqa:rc::olmes:full",
    "socialiqa:rc:bpb::olmes:full",
    "winogrande:rc::olmes:full",
    "winogrande:rc:bpb::olmes:full",
)


@dataclass(frozen=True)
class EvalWorkerConfig:
    scheduler_mode: str
    workdir: str
    raw_results_dir: str
    log_dir: str
    batch_size: int
    force_eval: bool
    tasks: Tuple[str, ...]
    olmes_bin: str = "olmes"
    max_retries: int = DEFAULT_MAX_RETRIES
    remote_workdir: Optional[str] = None
    remote_conda_env: str = DEFAULT_REMOTE_CONDA_ENV
    remote_shell_init: str = DEFAULT_REMOTE_SHELL_INIT
    ssh_connect_timeout: int = 5
    passthrough_env: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class EvalTask:
    model_name: str
    hf_model_dir: str
    mix_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OLMES evals over an explicit slot plan.")
    parser.add_argument("--hf-manifest", required=True, help="Path to eval/hf_models_manifest.csv.")
    parser.add_argument("--raw-results-dir", required=True, help="Directory for eval/raw_olmes outputs.")
    parser.add_argument("--log-dir", required=True, help="Directory for per-model eval logs.")
    parser.add_argument("--batch-size", type=int, default=1, help="OLMES batch size.")
    parser.add_argument(
        "--force-eval",
        type=int,
        default=0,
        help="Re-run evaluation even if metrics.json already exists (1=yes, 0=no).",
    )
    parser.add_argument("--mix-start", type=int, default=0, help="First mix index (inclusive).")
    parser.add_argument("--mix-end", type=int, default=14, help="Last mix index (inclusive).")
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
    parser.add_argument("--olmes-bin", default="olmes", help="OLMES executable used for each task.")
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental executor state JSON. Defaults to <log-dir>/../parallel_eval_state.json.",
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for failed or incomplete tasks.",
    )
    return parser.parse_args()


def default_state_file(log_dir: str) -> str:
    return str(Path(log_dir).parent / "parallel_eval_state.json")


def build_task_list(hf_manifest: str, mix_start: int, mix_end: int) -> List[EvalTask]:
    manifest_path = Path(hf_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"HF manifest does not exist: {hf_manifest}")

    tasks: List[EvalTask] = []
    with manifest_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mix_idx = int(row["mix_index"])
            if not (mix_start <= mix_idx <= mix_end):
                continue
            hf_model_dir = row["hf_model_dir"].strip()
            hf_model_path = Path(hf_model_dir)
            if not hf_model_path.is_absolute():
                hf_model_path = (manifest_path.parent / hf_model_path).resolve()
            tasks.append(
                EvalTask(
                    model_name=hf_model_path.name,
                    hf_model_dir=str(hf_model_path),
                    mix_index=mix_idx,
                )
            )

    if not tasks:
        raise ValueError(f"Found no tasks in {hf_manifest} for mix range {mix_start}-{mix_end}")
    return tasks


def verify_metrics_complete(metrics_file: Path, expected_tasks: int = len(DEFAULT_OLMES_TASKS)) -> bool:
    if not metrics_file.exists():
        return False
    try:
        with metrics_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False

    tasks = data.get("tasks")
    return isinstance(tasks, list) and len(tasks) >= expected_tasks


def is_oom_error(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    try:
        log_content = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    oom_patterns = ("out of memory", "cuda out of memory", "cuda error: out of memory")
    return any(pattern in log_content for pattern in oom_patterns)


def build_eval_command(cfg: EvalWorkerConfig, task: EvalTask, output_dir: str) -> List[str]:
    return [
        cfg.olmes_bin,
        "--model",
        task.model_name,
        "--model-type",
        "hf",
        "--model-args",
        f"model_path={task.hf_model_dir},trust_remote_code=True,add_bos_token=True,max_length=8192",
        "--task",
        *cfg.tasks,
        "--batch-size",
        str(cfg.batch_size),
        "--gpus",
        "1",
        "--output-dir",
        output_dir,
    ]


def build_task_env(cfg: EvalWorkerConfig, slot: WorkerSlot) -> Dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": str(slot.gpu_id),
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "DATASETS_TRUST_REMOTE_CODE": "1",
    }
    if cfg.passthrough_env:
        env.update(cfg.passthrough_env)
    return env


def build_remote_task_command(
    cfg: EvalWorkerConfig,
    slot: WorkerSlot,
    task: EvalTask,
    output_dir: str,
) -> Tuple[str, str]:
    remote_workdir = cfg.remote_workdir or cfg.workdir
    env = build_task_env(cfg, slot)
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    task_cmd = shlex.join(build_eval_command(cfg, task, output_dir))
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


def build_started_event(slot: WorkerSlot, task: EvalTask, log_path: str, output_dir: str) -> Dict[str, Any]:
    return {
        "event": "started",
        "model_name": task.model_name,
        "mix_index": task.mix_index,
        "hf_model_dir": task.hf_model_dir,
        "host": slot.host,
        "gpu_id": slot.gpu_id,
        "gpu_uuid": slot.gpu_uuid,
        "log_path": log_path,
        "output_dir": output_dir,
        "started_at": time.time(),
    }


def complete_event(
    status: str,
    slot: WorkerSlot,
    task: EvalTask,
    log_path: str,
    output_dir: str,
    return_code: int,
    duration_sec: float,
    error: str = "",
) -> Dict[str, Any]:
    return {
        "event": "completed",
        "status": status,
        "model_name": task.model_name,
        "mix_index": task.mix_index,
        "hf_model_dir": task.hf_model_dir,
        "host": slot.host,
        "gpu_id": slot.gpu_id,
        "gpu_uuid": slot.gpu_uuid,
        "return_code": return_code,
        "duration_sec": round(duration_sec, 2),
        "log_path": log_path,
        "output_dir": output_dir,
        "error": error,
        "completed_at": time.time(),
    }


def run_single_eval_task(
    slot: WorkerSlot,
    task: EvalTask,
    cfg: EvalWorkerConfig,
    retry_count: int = 0,
) -> Dict[str, Any]:
    output_dir = str(Path(cfg.raw_results_dir) / task.model_name)
    metrics_file = Path(output_dir) / "metrics.json"
    log_file = Path(cfg.log_dir) / f"eval-{task.model_name}.log"

    if metrics_file.exists() and not cfg.force_eval:
        if verify_metrics_complete(metrics_file, expected_tasks=len(cfg.tasks)):
            return complete_event(
                status="cached",
                slot=slot,
                task=task,
                log_path=str(log_file),
                output_dir=output_dir,
                return_code=0,
                duration_sec=0.0,
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = build_task_env(cfg, slot)
    task_cmd = shlex.join(build_eval_command(cfg, task, output_dir))
    exec_cmd: List[str]
    proc_env: Optional[Dict[str, str]]
    cwd: Optional[str]

    if cfg.scheduler_mode == "cluster":
        remote_body, task_cmd = build_remote_task_command(cfg, slot, task, output_dir)
        exec_cmd = build_ssh_command(slot.host, build_remote_bash_command(remote_body), cfg.ssh_connect_timeout)
        proc_env = None
        cwd = None
    else:
        exec_cmd = build_eval_command(cfg, task, output_dir)
        proc_env = os.environ.copy()
        proc_env.update(env)
        cwd = cfg.workdir

    start_ts = time.time()
    return_code = 99
    error_message = ""

    try:
        with log_file.open("w", encoding="utf-8") as logf:
            logf.write(
                f"[eval_executor] mode={cfg.scheduler_mode} host={slot.host} gpu_id={slot.gpu_id} "
                f"model={task.model_name} mix_index={task.mix_index}\n"
            )
            if slot.gpu_uuid:
                logf.write(f"[eval_executor] gpu_uuid={slot.gpu_uuid}\n")
            logf.write(f"[eval_executor] task_command={task_cmd}\n")
            logf.write(
                "[eval_executor] env "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                f"HF_DATASETS_TRUST_REMOTE_CODE={env['HF_DATASETS_TRUST_REMOTE_CODE']}\n"
            )
            if cfg.scheduler_mode == "cluster":
                logf.write(f"[eval_executor] ssh_command={shlex.join(exec_cmd)}\n")
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
            with log_file.open("a", encoding="utf-8") as logf:
                logf.write(f"[eval_executor] exception={error_message}\n")
        except OSError:
            pass
        return complete_event(
            status="failed",
            slot=slot,
            task=task,
            log_path=str(log_file),
            output_dir=output_dir,
            return_code=return_code,
            duration_sec=time.time() - start_ts,
            error=error_message,
        )

    duration_sec = time.time() - start_ts
    if return_code == 0:
        if verify_metrics_complete(metrics_file, expected_tasks=len(cfg.tasks)):
            return complete_event(
                status="success",
                slot=slot,
                task=task,
                log_path=str(log_file),
                output_dir=output_dir,
                return_code=return_code,
                duration_sec=duration_sec,
            )
        if retry_count < cfg.max_retries:
            time.sleep(60)
            return run_single_eval_task(slot, task, cfg, retry_count + 1)
        return complete_event(
            status="incomplete",
            slot=slot,
            task=task,
            log_path=str(log_file),
            output_dir=output_dir,
            return_code=return_code,
            duration_sec=duration_sec,
            error="metrics.json is incomplete",
        )

    if is_oom_error(log_file) and cfg.batch_size > 1 and retry_count == 0:
        cfg_retry = replace(cfg, batch_size=1)
        return run_single_eval_task(slot, task, cfg_retry, retry_count=1)

    if retry_count < cfg.max_retries:
        time.sleep(60 * (2 ** retry_count))
        return run_single_eval_task(slot, task, cfg, retry_count + 1)

    return complete_event(
        status="failed",
        slot=slot,
        task=task,
        log_path=str(log_file),
        output_dir=output_dir,
        return_code=return_code,
        duration_sec=duration_sec,
    )


def worker_loop(
    slot: WorkerSlot,
    task_queue: "mp.Queue[Optional[EvalTask]]",
    event_queue: "mp.Queue[Dict[str, Any]]",
    cfg: EvalWorkerConfig,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        output_dir = str(Path(cfg.raw_results_dir) / task.model_name)
        log_path = str(Path(cfg.log_dir) / f"eval-{task.model_name}.log")
        event_queue.put(build_started_event(slot, task, log_path, output_dir))
        event_queue.put(run_single_eval_task(slot, task, cfg))


def build_passthrough_env(raw: str) -> Dict[str, str]:
    passthrough: Dict[str, str] = {}
    for key in [item.strip() for item in raw.split(",") if item.strip()]:
        if key in os.environ:
            passthrough[key] = os.environ[key]
    return passthrough


def initialize_state(
    cfg: EvalWorkerConfig,
    slots: Sequence[WorkerSlot],
    tasks: Sequence[EvalTask],
    hf_manifest: str,
    scan_errors: Dict[str, str],
    slots_file: str,
) -> Dict[str, Any]:
    return {
        "executor": "parallel_eval.py",
        "scheduler_mode": cfg.scheduler_mode,
        "workdir": cfg.workdir,
        "hf_manifest": hf_manifest,
        "raw_results_dir": cfg.raw_results_dir,
        "log_dir": cfg.log_dir,
        "slot_plan_file": slots_file,
        "scan_errors": scan_errors,
        "slots": [asdict(slot) for slot in slots],
        "task_mix_indices": [task.mix_index for task in tasks],
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
                "model_name": event["model_name"],
                "hf_model_dir": event["hf_model_dir"],
                "host": event["host"],
                "gpu_id": event["gpu_id"],
                "gpu_uuid": event.get("gpu_uuid"),
                "log_path": event["log_path"],
                "output_dir": event["output_dir"],
                "status": "running",
                "started_at": event["started_at"],
            }
        )
    elif event["event"] == "completed":
        task_state.update(
            {
                "model_name": event["model_name"],
                "hf_model_dir": event["hf_model_dir"],
                "host": event["host"],
                "gpu_id": event["gpu_id"],
                "gpu_uuid": event.get("gpu_uuid"),
                "log_path": event["log_path"],
                "output_dir": event["output_dir"],
                "return_code": event["return_code"],
                "duration_sec": event["duration_sec"],
                "error": event["error"],
                "completed_at": event["completed_at"],
                "status": event["status"],
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
    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)

    resolved_manifest = resolve_path(args.hf_manifest, workdir)
    resolved_raw_results_dir = resolve_path(args.raw_results_dir, workdir)
    resolved_log_dir = resolve_path(args.log_dir, workdir)
    state_file = resolve_path(args.state_file, workdir) if args.state_file else default_state_file(resolved_log_dir)
    slots_file = resolve_path(args.slots_file, workdir)
    slot_plan = read_slot_plan_file(slots_file)
    slots = list(slot_plan.slots)
    tasks = build_task_list(resolved_manifest, args.mix_start, args.mix_end)

    if not slots:
        raise RuntimeError(f"slot plan contains no schedulable slots: {slots_file}")
    if slot_plan.scheduler_mode != args.scheduler_mode:
        raise ValueError(
            f"slot plan mode mismatch: slots_file={slot_plan.scheduler_mode} executor={args.scheduler_mode}"
        )

    cfg = EvalWorkerConfig(
        scheduler_mode=args.scheduler_mode,
        workdir=workdir,
        raw_results_dir=resolved_raw_results_dir,
        log_dir=resolved_log_dir,
        batch_size=args.batch_size,
        force_eval=bool(args.force_eval),
        tasks=DEFAULT_OLMES_TASKS,
        olmes_bin=args.olmes_bin,
        max_retries=args.max_retries,
        remote_workdir=args.remote_workdir or workdir,
        remote_conda_env=args.remote_conda_env,
        remote_shell_init=args.remote_shell_init,
        ssh_connect_timeout=args.ssh_connect_timeout,
        passthrough_env=build_passthrough_env(args.passthrough_env),
    )

    print(
        f"[eval_executor] mode={cfg.scheduler_mode} workers={len(slots)} "
        f"requested_workers={slot_plan.requested_workers} models={len(tasks)}"
    )
    if cfg.scheduler_mode == "cluster":
        print("[eval_executor] slots=" + ",".join(slot.label for slot in slots))
    print(f"[eval_executor] tasks_per_model={len(cfg.tasks)} batch_size={cfg.batch_size} force_eval={cfg.force_eval}")

    state = initialize_state(cfg, slots, tasks, resolved_manifest, slot_plan.scan_errors, slots_file)
    write_state_file(state_file, state)

    task_queue: "mp.Queue[Optional[EvalTask]]" = mp.Queue()
    event_queue: "mp.Queue[Dict[str, Any]]" = mp.Queue()

    for task in tasks:
        task_queue.put(task)
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
        while completed < len(tasks):
            event = event_queue.get()
            apply_event_to_state(state, event)
            write_state_file(state_file, state)

            if event["event"] != "completed":
                continue

            completed += 1
            results.append(event)
            status = event["status"]
            if status == "success":
                label = "OK"
            elif status == "cached":
                label = "CACHED"
            else:
                label = status.upper()
            duration_str = f"{event.get('duration_sec', 0):.1f}s"
            print(
                f"[{label}] {event['model_name']} mix={event['mix_index']:02d} "
                f"host={event['host']} gpu={event['gpu_id']} time={duration_str} "
                f"log={event['log_path']}"
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
    failed = [result for result in results if result["status"] == "failed"]
    incomplete = [result for result in results if result["status"] == "incomplete"]
    cached = [result for result in results if result["status"] == "cached"]
    success = [result for result in results if result["status"] == "success"]

    report = {
        "executor": "parallel_eval.py",
        "scheduler_mode": cfg.scheduler_mode,
        "slot_plan_file": slots_file,
        "state_file": state_file,
        "total": len(results),
        "success": len(success),
        "cached": len(cached),
        "incomplete": len(incomplete),
        "failed": len(failed),
        "failed_mix_indices": [int(result["mix_index"]) for result in failed + incomplete],
        "slots": [slot.label for slot in slots],
        "scan_errors": slot_plan.scan_errors,
        "failures": failed + incomplete,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if (failed or incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
