from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_CLUSTER_HOSTS = "hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15"
DEFAULT_REMOTE_CONDA_ENV = "regmixer"
DEFAULT_REMOTE_SHELL_INIT = "~/.bashrc"
DEFAULT_SSH_PROBE_TIMEOUT = 15


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

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "WorkerSlot":
        return cls(
            host=str(payload["host"]),
            gpu_id=int(payload["gpu_id"]),
            gpu_uuid=_optional_str(payload.get("gpu_uuid")),
            memory_used_mb=_optional_int(payload.get("memory_used_mb")),
            utilization_gpu=_optional_int(payload.get("utilization_gpu")),
        )


@dataclass(frozen=True)
class GpuSnapshot:
    gpu_id: int
    gpu_uuid: str
    memory_used_mb: int
    utilization_gpu: int


@dataclass(frozen=True)
class SlotPlan:
    scheduler_mode: str
    requested_workers: int
    slots: Tuple[WorkerSlot, ...]
    scan_errors: Dict[str, str]
    created_at: float


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    return int(value)


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


def resolve_worker_limit(requested_max_workers: Optional[int], task_count: int) -> int:
    if requested_max_workers is not None:
        return requested_max_workers
    return task_count


def resolve_path(value: str, base_dir: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


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
        if parts and parts[0]:
            gpu_uuids.add(parts[0])
    return gpu_uuids


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
    local_gpu_ids = list(gpu_ids) if gpu_ids is not None else discover_local_gpu_ids()
    slots = [WorkerSlot(host=local_host, gpu_id=gpu_id) for gpu_id in local_gpu_ids]
    if max_workers is not None:
        slots = slots[:max_workers]
    return slots, {}


def validate_cluster_mode_paths(workdir: str, remote_workdir: Optional[str]) -> None:
    if not Path(workdir).is_absolute():
        raise ValueError("cluster mode requires an absolute --workdir")
    if remote_workdir and not Path(remote_workdir).is_absolute():
        raise ValueError("cluster mode requires an absolute --remote-workdir when provided")


def write_slot_plan_file(
    path: str,
    scheduler_mode: str,
    requested_workers: int,
    slots: Sequence[WorkerSlot],
    scan_errors: Dict[str, str],
) -> None:
    payload = {
        "scheduler_mode": scheduler_mode,
        "requested_workers": requested_workers,
        "allocated_workers": len(slots),
        "scan_errors": scan_errors,
        "slots": [asdict(slot) for slot in slots],
        "created_at": time.time(),
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path_obj)


def read_slot_plan_file(path: str) -> SlotPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    slots = tuple(WorkerSlot.from_dict(slot) for slot in payload.get("slots", []))
    return SlotPlan(
        scheduler_mode=str(payload["scheduler_mode"]),
        requested_workers=int(payload["requested_workers"]),
        slots=slots,
        scan_errors={str(key): str(value) for key, value in payload.get("scan_errors", {}).items()},
        created_at=float(payload.get("created_at", 0.0)),
    )
