import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_parallel_train_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "parallel_train.py"
    spec = importlib.util.spec_from_file_location("parallel_train", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parallel_train = load_parallel_train_module()


def test_parse_nvidia_smi_gpu_output_parses_rows():
    stdout = "\n".join(
        [
            "0, GPU-aaa, 0, 0",
            "1, GPU-bbb, 2048, 31",
        ]
    )

    snapshots = parallel_train.parse_nvidia_smi_gpu_output(stdout)

    assert [snapshot.gpu_id for snapshot in snapshots] == [0, 1]
    assert snapshots[1].gpu_uuid == "GPU-bbb"
    assert snapshots[1].memory_used_mb == 2048
    assert snapshots[1].utilization_gpu == 31


def test_parse_nvidia_smi_compute_output_ignores_empty_state():
    stdout = "No running processes found"

    busy = parallel_train.parse_nvidia_smi_compute_output(stdout)

    assert busy == set()


def test_select_idle_slots_filters_busy_and_disallowed_gpu_ids():
    snapshots = [
        parallel_train.GpuSnapshot(gpu_id=0, gpu_uuid="GPU-0", memory_used_mb=0, utilization_gpu=0),
        parallel_train.GpuSnapshot(gpu_id=1, gpu_uuid="GPU-1", memory_used_mb=0, utilization_gpu=0),
        parallel_train.GpuSnapshot(gpu_id=2, gpu_uuid="GPU-2", memory_used_mb=0, utilization_gpu=0),
    ]

    slots, busy_gpu_ids = parallel_train.select_idle_slots(
        host="hpcgpu11",
        gpu_snapshots=snapshots,
        busy_gpu_uuids={"GPU-1"},
        allowed_gpu_ids=[0, 1],
    )

    assert [slot.label for slot in slots] == ["hpcgpu11:0"]
    assert busy_gpu_ids == [1]


def test_build_remote_task_command_contains_expected_remote_bootstrap(tmp_path):
    workdir = tmp_path / "regmixer"
    workdir.mkdir()
    cfg = parallel_train.WorkerConfig(
        scheduler_mode="cluster",
        workdir=str(workdir),
        python_bin="python",
        variant_script=str(workdir / "scripts" / "run_local_variant.py"),
        config_path=str(workdir / "config.yaml"),
        mix_file=str(workdir / "mixes.json"),
        group_id="round1a",
        run_name_prefix="round1a-train",
        log_dir=str(workdir / "logs"),
        summary_dir=str(workdir / "summaries"),
        pythonpath="src",
        remote_workdir=str(workdir),
        passthrough_env={"WANDB_MODE": "offline"},
    )
    slot = parallel_train.WorkerSlot(host="hpcgpu15", gpu_id=3, gpu_uuid="GPU-3")
    task = parallel_train.TaskSpec(
        mix_index=5,
        run_name="round1a-train-0005",
        log_path=str(workdir / "logs" / "round1a-train-0005.log"),
        summary_path=str(workdir / "summaries" / "round1a-train-0005.json"),
    )

    remote_body, task_cmd = parallel_train.build_remote_task_command(cfg, slot, task)

    assert "source ~/.bashrc" in remote_body
    assert "conda activate regmixer" in remote_body
    assert f"cd {workdir}" in remote_body
    assert "CUDA_VISIBLE_DEVICES=3" in remote_body
    assert "WANDB_MODE=offline" in remote_body
    assert str(workdir / "scripts" / "run_local_variant.py") in task_cmd


def test_run_remote_capture_times_out_probe(monkeypatch):
    def fake_run(*args, **kwargs):
        raise parallel_train.subprocess.TimeoutExpired(cmd="ssh hpcgpu12 ...", timeout=9)

    monkeypatch.setattr(parallel_train.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"ssh probe timed out for hpcgpu12 after 9s"):
        parallel_train.run_remote_capture("hpcgpu12", "nvidia-smi", connect_timeout=5, probe_timeout=9)


def test_state_file_tracks_started_and_completed_tasks(tmp_path):
    state_path = tmp_path / "parallel_train_state.json"
    cfg = parallel_train.WorkerConfig(
        scheduler_mode="cluster",
        workdir=str(tmp_path),
        python_bin="python",
        variant_script=str(tmp_path / "scripts" / "run_local_variant.py"),
        config_path=str(tmp_path / "config.yaml"),
        mix_file=str(tmp_path / "mixes.json"),
        group_id="round1a",
        run_name_prefix="round1a-train",
        log_dir=str(tmp_path / "logs"),
        summary_dir=str(tmp_path / "summaries"),
        pythonpath="src",
        remote_workdir=str(tmp_path),
    )
    slot = parallel_train.WorkerSlot(host="hpcgpu09", gpu_id=0, gpu_uuid="GPU-0")
    state = parallel_train.initialize_state(cfg, [slot], [0], {})

    parallel_train.apply_event_to_state(
        state,
        {
            "event": "started",
            "mix_index": 0,
            "run_name": "round1a-train-0000",
            "host": "hpcgpu09",
            "gpu_id": 0,
            "gpu_uuid": "GPU-0",
            "log_path": str(tmp_path / "logs" / "round1a-train-0000.log"),
            "summary_path": str(tmp_path / "summaries" / "round1a-train-0000.json"),
            "started_at": 1.0,
        },
    )
    parallel_train.apply_event_to_state(
        state,
        {
            "event": "completed",
            "mix_index": 0,
            "run_name": "round1a-train-0000",
            "host": "hpcgpu09",
            "gpu_id": 0,
            "gpu_uuid": "GPU-0",
            "log_path": str(tmp_path / "logs" / "round1a-train-0000.log"),
            "summary_path": str(tmp_path / "summaries" / "round1a-train-0000.json"),
            "return_code": 0,
            "duration_sec": 12.5,
            "error": "",
            "completed_at": 2.0,
        },
    )
    parallel_train.write_state_file(str(state_path), state)

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["tasks"]["0"]["status"] == "succeeded"
    assert payload["tasks"]["0"]["host"] == "hpcgpu09"
    assert payload["tasks"]["0"]["return_code"] == 0
