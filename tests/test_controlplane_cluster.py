import json
from pathlib import Path

from regmixer.controlplane import cluster


def test_parse_nvidia_smi_gpu_output_parses_rows():
    stdout = "\n".join(["0, GPU-aaa, 0, 0", "1, GPU-bbb, 2048, 31"])

    snapshots = cluster.parse_nvidia_smi_gpu_output(stdout)

    assert [snapshot.gpu_id for snapshot in snapshots] == [0, 1]
    assert snapshots[1].gpu_uuid == "GPU-bbb"
    assert snapshots[1].memory_used_mb == 2048
    assert snapshots[1].utilization_gpu == 31


def test_parse_nvidia_smi_compute_output_ignores_empty_state():
    assert cluster.parse_nvidia_smi_compute_output("No running processes found") == set()


def test_parse_gpu_ids_accepts_all():
    assert cluster.parse_gpu_ids("all") is None


def test_resolve_worker_limit_defaults_to_task_count():
    assert cluster.resolve_worker_limit(None, 4) == 4
    assert cluster.resolve_worker_limit(2, 4) == 2


def test_select_idle_slots_filters_busy_and_disallowed_gpu_ids():
    snapshots = [
        cluster.GpuSnapshot(gpu_id=0, gpu_uuid="GPU-0", memory_used_mb=0, utilization_gpu=0),
        cluster.GpuSnapshot(gpu_id=1, gpu_uuid="GPU-1", memory_used_mb=0, utilization_gpu=0),
        cluster.GpuSnapshot(gpu_id=2, gpu_uuid="GPU-2", memory_used_mb=0, utilization_gpu=0),
    ]

    slots, busy_gpu_ids = cluster.select_idle_slots(
        host="hpcgpu11",
        gpu_snapshots=snapshots,
        busy_gpu_uuids={"GPU-1"},
        allowed_gpu_ids=[0, 1],
    )

    assert [slot.label for slot in slots] == ["hpcgpu11:0"]
    assert busy_gpu_ids == [1]


def test_build_worker_slots_discovers_local_gpus_when_gpu_ids_are_omitted(monkeypatch):
    monkeypatch.setattr(cluster, "discover_local_gpu_ids", lambda: [0, 1, 2])

    slots, scan_errors = cluster.build_worker_slots(
        scheduler_mode="local",
        hosts=[],
        gpu_ids=None,
        connect_timeout=5,
        probe_timeout=5,
        max_workers=2,
    )

    assert [slot.gpu_id for slot in slots] == [0, 1]
    assert scan_errors == {}


def test_run_remote_capture_times_out_probe(monkeypatch):
    def fake_run(*args, **kwargs):
        raise cluster.subprocess.TimeoutExpired(cmd="ssh hpcgpu12 ...", timeout=9)

    monkeypatch.setattr(cluster.subprocess, "run", fake_run)

    try:
        cluster.run_remote_capture("hpcgpu12", "nvidia-smi", connect_timeout=5, probe_timeout=9)
    except RuntimeError as exc:
        assert "ssh probe timed out for hpcgpu12 after 9s" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_slot_plan_round_trip(tmp_path):
    slot_plan_path = tmp_path / "slot_plan.json"
    slots = [cluster.WorkerSlot(host="hpcgpu09", gpu_id=3, gpu_uuid="GPU-3")]

    cluster.write_slot_plan_file(
        str(slot_plan_path),
        scheduler_mode="cluster",
        requested_workers=4,
        slots=slots,
        scan_errors={"hpcgpu10": "timed out"},
    )
    payload = json.loads(slot_plan_path.read_text(encoding="utf-8"))
    loaded = cluster.read_slot_plan_file(str(slot_plan_path))

    assert payload["allocated_workers"] == 1
    assert loaded.scheduler_mode == "cluster"
    assert loaded.requested_workers == 4
    assert loaded.scan_errors == {"hpcgpu10": "timed out"}
    assert [slot.label for slot in loaded.slots] == ["hpcgpu09:3"]
