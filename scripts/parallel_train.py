#!/usr/bin/env python
"""Parallel regmixer variant scheduler for local and shared-cluster execution."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_CLUSTER_HOSTS = "hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15"
DEFAULT_REMOTE_CONDA_ENV = "regmixer"
DEFAULT_REMOTE_SHELL_INIT = "~/.bashrc"
DEFAULT_PASSTHROUGH_ENV = "WANDB_MODE,REGMIXER_DISABLE_SWANLAB_SYNC"


@dataclass(frozen=True)
class WorkerSlot:
    host: str
    gpu_id: int
    gpu_uuid: Optional[str] = None
    memory_used_mb: Optional[int] = None
    utilization_gpu: Optional[int] = None

    @property
    def label(self) -> str:
        return f"{self.host}:{self.gpu_id}"


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


@dataclass(frozen=True)
class GpuSnapshot:
    gpu_id: int
    gpu_uuid: str
    memory_used_mb: int
    utilization_gpu: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regmixer mixes with a local or shared-cluster worker queue."
    )
    parser.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    parser.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    parser.add_argument("--group-id", required=True, help="Group id passed to run_local_variant.py.")
    parser.add_argument(
        "--run-name-prefix",
        default="parquet-e2e-fit-train",
        help="Run name prefix. Full name is '<prefix>-<mix_index:04d>'.",
    )
    parser.add_argument(
        "--mix-start",
        type=int,
        default=0,
        help="First mix index (inclusive).",
    )
    parser.add_argument(
        "--mix-end",
        type=int,
        default=11,
        help="Last mix index (inclusive).",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=("local", "cluster"),
        default="local",
        help="Use local GPUs directly or discover free GPUs across a shared SSH host pool.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids allowed per worker host.",
    )
    parser.add_argument(
        "--hosts",
        default=DEFAULT_CLUSTER_HOSTS,
        help="Comma-separated SSH hosts scanned in cluster mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional cap on discovered worker slots after scanning.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory for subprocess execution.",
    )
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
    parser.add_argument(
        "--pythonpath",
        default="src",
        help="PYTHONPATH injected to each task subprocess.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/dryfit_fit/logs",
        help="Directory for per-task logs.",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/dryfit_fit/summaries",
        help="Directory for per-task summaries.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental scheduler state JSON. Defaults to <summary-dir>/../parallel_train_state.json.",
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


def parse_csv_list(raw: str) -> List[str]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("comma-separated value list is empty")
    return items


def parse_gpu_ids(raw: str) -> List[int]:
    gpu_ids = [int(item) for item in parse_csv_list(raw)]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"gpu id list contains duplicates: {gpu_ids}")
    return gpu_ids


def parse_hosts(raw: str) -> List[str]:
    hosts = parse_csv_list(raw)
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"host list contains duplicates: {hosts}")
    return hosts


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


def resolve_path(value: str, base_dir: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


def default_state_file(summary_dir: str) -> str:
    return str(Path(summary_dir).parent / "parallel_train_state.json")


def parse_nvidia_smi_gpu_output(stdout: str) -> List[GpuSnapshot]:
    snapshots: List[GpuSnapshot] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected GPU row: {line}")
        snapshots.append(
            GpuSnapshot(
                gpu_id=int(parts[0]),
                gpu_uuid=parts[1],
                memory_used_mb=int(parts[2]),
                utilization_gpu=int(parts[3]),
            )
        )
    return snapshots


def parse_nvidia_smi_compute_output(stdout: str) -> Set[str]:
    gpu_uuids: Set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("gpu_uuid"):
            continue
        if "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if not parts or not parts[0]:
            continue
        gpu_uuids.add(parts[0])
    return gpu_uuids


def select_idle_slots(
    host: str,
    gpu_snapshots: Sequence[GpuSnapshot],
    busy_gpu_uuids: Set[str],
    allowed_gpu_ids: Iterable[int],
) -> Tuple[List[WorkerSlot], List[int]]:
    allowed = set(allowed_gpu_ids)
    idle_slots: List[WorkerSlot] = []
    busy_gpu_ids: List[int] = []
    for snapshot in gpu_snapshots:
        if snapshot.gpu_id not in allowed:
            continue
        if snapshot.gpu_uuid in busy_gpu_uuids:
            busy_gpu_ids.append(snapshot.gpu_id)
            continue
        idle_slots.append(
            WorkerSlot(
                host=host,
                gpu_id=snapshot.gpu_id,
                gpu_uuid=snapshot.gpu_uuid,
                memory_used_mb=snapshot.memory_used_mb,
                utilization_gpu=snapshot.utilization_gpu,
            )
        )
    return idle_slots, busy_gpu_ids


def build_ssh_command(host: str, remote_command: str, connect_timeout: int) -> List[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        host,
        remote_command,
    ]


def build_remote_bash_command(body: str) -> str:
    return f"bash -lc {shlex.quote(body)}"


def run_remote_capture(host: str, body: str, connect_timeout: int) -> str:
    proc = subprocess.run(
        build_ssh_command(host, build_remote_bash_command(body), connect_timeout),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"ssh probe failed for {host}: {stderr or f'rc={proc.returncode}'}")
    return proc.stdout


def discover_cluster_slots(
    hosts: Sequence[str],
    allowed_gpu_ids: Sequence[int],
    connect_timeout: int,
) -> Tuple[List[WorkerSlot], Dict[str, str]]:
    slots: List[WorkerSlot] = []
    scan_errors: Dict[str, str] = {}
    gpu_query = "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits"
    compute_query = "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null || true"

    for host in hosts:
        try:
            gpu_stdout = run_remote_capture(host, gpu_query, connect_timeout)
            compute_stdout = run_remote_capture(host, compute_query, connect_timeout)
            gpu_snapshots = parse_nvidia_smi_gpu_output(gpu_stdout)
            busy_gpu_uuids = parse_nvidia_smi_compute_output(compute_stdout)
            host_slots, busy_gpu_ids = select_idle_slots(host, gpu_snapshots, busy_gpu_uuids, allowed_gpu_ids)
            busy_desc = ",".join(str(gpu_id) for gpu_id in sorted(busy_gpu_ids)) or "-"
            idle_desc = ",".join(str(slot.gpu_id) for slot in host_slots) or "-"
            print(f"[scan] host={host} idle_gpus={idle_desc} busy_gpus={busy_desc}")
            slots.extend(host_slots)
        except Exception as exc:
            scan_errors[host] = str(exc)
            print(f"[scan][ERROR] host={host} {exc}", file=sys.stderr)

    return slots, scan_errors


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
    return TaskSpec(
        mix_index=mix_index,
        run_name=run_name,
        log_path=log_path,
        summary_path=summary_path,
    )


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
                f"[scheduler] mode={cfg.scheduler_mode} host={slot.host} gpu_id={slot.gpu_id} mix_index={task.mix_index}\n"
            )
            if slot.gpu_uuid:
                logf.write(f"[scheduler] gpu_uuid={slot.gpu_uuid}\n")
            logf.write(f"[scheduler] task_command={task_cmd}\n")
            logf.write(
                "[scheduler] env "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                f"MASTER_PORT={env['MASTER_PORT']} "
                f"TORCH_DISTRIBUTED_DEFAULT_PORT={env['TORCH_DISTRIBUTED_DEFAULT_PORT']}\n"
            )
            if cfg.scheduler_mode == "cluster":
                logf.write(f"[scheduler] ssh_command={shlex.join(exec_cmd)}\n")
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
                logf.write(f"[scheduler] exception={error_message}\n")
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
) -> Dict[str, Any]:
    return {
        "scheduler_mode": cfg.scheduler_mode,
        "workdir": cfg.workdir,
        "config_path": cfg.config_path,
        "mix_file": cfg.mix_file,
        "group_id": cfg.group_id,
        "mix_indices": list(mix_indices),
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


def build_worker_slots(
    scheduler_mode: str,
    hosts: Sequence[str],
    gpu_ids: Sequence[int],
    connect_timeout: int,
    max_workers: Optional[int],
) -> Tuple[List[WorkerSlot], Dict[str, str]]:
    if scheduler_mode == "cluster":
        slots, scan_errors = discover_cluster_slots(hosts, gpu_ids, connect_timeout)
        if max_workers is not None:
            slots = slots[:max_workers]
        if not slots:
            error_text = "; ".join(f"{host}: {msg}" for host, msg in scan_errors.items()) or "no idle GPU slots found"
            raise RuntimeError(f"cluster mode found no schedulable slots ({error_text})")
        return slots, scan_errors

    local_host = socket.gethostname()
    slots = [WorkerSlot(host=local_host, gpu_id=gpu_id) for gpu_id in gpu_ids]
    if max_workers is not None:
        slots = slots[:max_workers]
    return slots, {}


def validate_cluster_mode_paths(workdir: str, remote_workdir: Optional[str]) -> None:
    if not Path(workdir).is_absolute():
        raise ValueError("cluster mode requires an absolute --workdir")
    if remote_workdir and not Path(remote_workdir).is_absolute():
        raise ValueError("cluster mode requires an absolute --remote-workdir when provided")


def main() -> int:
    args = parse_args()
    mix_indices = build_mix_indices(args.mix_start, args.mix_end)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    hosts = parse_hosts(args.hosts)

    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)

    resolved_log_dir = resolve_path(args.log_dir, workdir)
    resolved_summary_dir = resolve_path(args.summary_dir, workdir)
    state_file = resolve_path(args.state_file, workdir) if args.state_file else default_state_file(resolved_summary_dir)

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

    slots, scan_errors = build_worker_slots(
        scheduler_mode=args.scheduler_mode,
        hosts=hosts,
        gpu_ids=gpu_ids,
        connect_timeout=args.ssh_connect_timeout,
        max_workers=args.max_workers,
    )

    print(f"[scheduler] mode={cfg.scheduler_mode} workers={len(slots)} mixes={len(mix_indices)}")
    if cfg.scheduler_mode == "cluster":
        print("[scheduler] slots=" + ",".join(slot.label for slot in slots))

    state = initialize_state(cfg, slots, mix_indices, scan_errors)
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
        "scheduler_mode": cfg.scheduler_mode,
        "state_file": state_file,
        "total": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "failed_mix_indices": [int(result["mix_index"]) for result in failed],
        "slots": [slot.label for slot in slots],
        "scan_errors": scan_errors,
        "failures": failed,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
