#!/usr/bin/env python
"""Parallel OLMES evaluation scheduler for local or shared-cluster execution."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_CLUSTER_HOSTS = "hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15"
DEFAULT_REMOTE_CONDA_ENV = "regmixer"
DEFAULT_REMOTE_SHELL_INIT = "~/.bashrc"
DEFAULT_PASSTHROUGH_ENV = "HF_HOME,HF_HUB_CACHE,HF_TOKEN,TRANSFORMERS_CACHE,XDG_CACHE_HOME"
DEFAULT_SSH_PROBE_TIMEOUT = 15
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
class GpuSnapshot:
    gpu_id: int
    gpu_uuid: str
    memory_used_mb: int
    utilization_gpu: int


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
    parser = argparse.ArgumentParser(
        description="Run OLMES evaluations with a local or shared-cluster worker queue."
    )
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
        default="local",
        help="Use local GPUs directly or discover free GPUs across a shared SSH host pool.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="all",
        help="Comma-separated GPU ids allowed per worker host, or 'all'.",
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
        "--olmes-bin",
        default="olmes",
        help="OLMES executable used for each task.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to incremental scheduler state JSON. Defaults to <log-dir>/../parallel_eval_state.json.",
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
        "--ssh-probe-timeout",
        type=int,
        default=DEFAULT_SSH_PROBE_TIMEOUT,
        help="Execution timeout in seconds for remote probe commands such as nvidia-smi.",
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


def parse_csv_list(raw: str) -> List[str]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("comma-separated value list is empty")
    return items


def parse_gpu_ids(raw: str) -> Optional[List[int]]:
    if raw.strip().lower() == "all":
        return None
    gpu_ids = [int(item) for item in parse_csv_list(raw)]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"gpu id list contains duplicates: {gpu_ids}")
    return gpu_ids


def parse_hosts(raw: str) -> List[str]:
    hosts = parse_csv_list(raw)
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"host list contains duplicates: {hosts}")
    return hosts


def resolve_worker_limit(requested_max_workers: Optional[int], tasks: Sequence[EvalTask]) -> int:
    if requested_max_workers is not None:
        return requested_max_workers
    return len(tasks)


def resolve_path(value: str, base_dir: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


def default_state_file(log_dir: str) -> str:
    return str(Path(log_dir).parent / "parallel_eval_state.json")


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


def discover_local_gpu_ids() -> List[int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"local gpu probe failed: {stderr or f'rc={proc.returncode}'}")
    return [snapshot.gpu_id for snapshot in parse_nvidia_smi_gpu_output(proc.stdout)]


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
    allowed_gpu_ids: Optional[Iterable[int]],
) -> Tuple[List[WorkerSlot], List[int]]:
    allowed = set(allowed_gpu_ids) if allowed_gpu_ids is not None else None
    idle_slots: List[WorkerSlot] = []
    busy_gpu_ids: List[int] = []
    for snapshot in gpu_snapshots:
        if allowed is not None and snapshot.gpu_id not in allowed:
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


def run_remote_capture(host: str, body: str, connect_timeout: int, probe_timeout: int) -> str:
    try:
        proc = subprocess.run(
            build_ssh_command(host, build_remote_bash_command(body), connect_timeout),
            check=False,
            capture_output=True,
            text=True,
            timeout=probe_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ssh probe timed out for {host} after {probe_timeout}s") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"ssh probe failed for {host}: {stderr or f'rc={proc.returncode}'}")
    return proc.stdout


def discover_cluster_slots(
    hosts: Sequence[str],
    allowed_gpu_ids: Optional[Sequence[int]],
    connect_timeout: int,
    probe_timeout: int,
) -> Tuple[List[WorkerSlot], Dict[str, str]]:
    slots: List[WorkerSlot] = []
    scan_errors: Dict[str, str] = {}
    gpu_query = "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits"
    compute_query = "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null || true"

    for host in hosts:
        try:
            gpu_stdout = run_remote_capture(host, gpu_query, connect_timeout, probe_timeout)
            compute_stdout = run_remote_capture(host, compute_query, connect_timeout, probe_timeout)
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


def build_task_list(hf_manifest: str, mix_start: int, mix_end: int) -> List[EvalTask]:
    manifest_path = Path(hf_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"HF manifest does not exist: {hf_manifest}")

    tasks: List[EvalTask] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
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
        with metrics_file.open(encoding="utf-8") as f:
            data = json.load(f)
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

    oom_patterns = (
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
    )
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
                f"[eval_scheduler] mode={cfg.scheduler_mode} host={slot.host} gpu_id={slot.gpu_id} "
                f"model={task.model_name} mix_index={task.mix_index}\n"
            )
            if slot.gpu_uuid:
                logf.write(f"[eval_scheduler] gpu_uuid={slot.gpu_uuid}\n")
            logf.write(f"[eval_scheduler] task_command={task_cmd}\n")
            logf.write(
                "[eval_scheduler] env "
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                f"HF_DATASETS_TRUST_REMOTE_CODE={env['HF_DATASETS_TRUST_REMOTE_CODE']}\n"
            )
            if cfg.scheduler_mode == "cluster":
                logf.write(f"[eval_scheduler] ssh_command={shlex.join(exec_cmd)}\n")
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
                logf.write(f"[eval_scheduler] exception={error_message}\n")
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
) -> Dict[str, Any]:
    return {
        "scheduler_mode": cfg.scheduler_mode,
        "workdir": cfg.workdir,
        "hf_manifest": hf_manifest,
        "raw_results_dir": cfg.raw_results_dir,
        "log_dir": cfg.log_dir,
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


def build_worker_slots(
    scheduler_mode: str,
    hosts: Sequence[str],
    gpu_ids: Optional[Sequence[int]],
    connect_timeout: int,
    probe_timeout: int,
    max_workers: Optional[int],
) -> Tuple[List[WorkerSlot], Dict[str, str]]:
    if scheduler_mode == "cluster":
        slots, scan_errors = discover_cluster_slots(hosts, gpu_ids, connect_timeout, probe_timeout)
        if max_workers is not None:
            slots = slots[:max_workers]
        if not slots:
            error_text = "; ".join(f"{host}: {msg}" for host, msg in scan_errors.items()) or "no idle GPU slots found"
            raise RuntimeError(f"cluster mode found no schedulable slots ({error_text})")
        return slots, scan_errors

    local_host = socket.gethostname()
    if gpu_ids is None:
        gpu_ids = discover_local_gpu_ids()
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
    tasks = build_task_list(args.hf_manifest, args.mix_start, args.mix_end)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    worker_limit = resolve_worker_limit(args.max_workers, tasks)
    hosts = parse_hosts(args.hosts)

    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)

    resolved_manifest = resolve_path(args.hf_manifest, workdir)
    resolved_raw_results_dir = resolve_path(args.raw_results_dir, workdir)
    resolved_log_dir = resolve_path(args.log_dir, workdir)
    state_file = resolve_path(args.state_file, workdir) if args.state_file else default_state_file(resolved_log_dir)

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

    slots, scan_errors = build_worker_slots(
        scheduler_mode=args.scheduler_mode,
        hosts=hosts,
        gpu_ids=gpu_ids,
        connect_timeout=args.ssh_connect_timeout,
        probe_timeout=args.ssh_probe_timeout,
        max_workers=worker_limit,
    )

    print(
        f"[eval_scheduler] mode={cfg.scheduler_mode} workers={len(slots)} "
        f"target_workers={worker_limit} models={len(tasks)}"
    )
    if cfg.scheduler_mode == "cluster":
        print("[eval_scheduler] slots=" + ",".join(slot.label for slot in slots))
    print(f"[eval_scheduler] tasks_per_model={len(cfg.tasks)} batch_size={cfg.batch_size} force_eval={cfg.force_eval}")

    state = initialize_state(cfg, slots, tasks, resolved_manifest, scan_errors)
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
        "scheduler_mode": cfg.scheduler_mode,
        "state_file": state_file,
        "total": len(results),
        "success": len(success),
        "cached": len(cached),
        "incomplete": len(incomplete),
        "failed": len(failed),
        "failed_mix_indices": [int(result["mix_index"]) for result in failed + incomplete],
        "slots": [slot.label for slot in slots],
        "scan_errors": scan_errors,
        "failures": failed + incomplete,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if (failed or incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
