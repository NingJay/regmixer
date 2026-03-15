import csv
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PLANE = REPO_ROOT / "scripts" / "control_plane.py"
FAKE_VARIANT = REPO_ROOT / "tests" / "fixtures" / "fake_variant_runner.py"
FAKE_OLMES = REPO_ROOT / "tests" / "fixtures" / "fake_olmes.py"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _ensure_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_control_plane_train_local_integration(tmp_path):
    _ensure_executable(FAKE_VARIANT)
    mix_file = tmp_path / "mixes.json"
    config_file = tmp_path / "config.yaml"
    control_state = tmp_path / "control_plane_state.json"
    executor_state = tmp_path / "parallel_train_state.json"
    slots_file = tmp_path / "train_slot_plan.json"
    summary_dir = tmp_path / "summaries"
    log_dir = tmp_path / "logs"
    output_root = tmp_path / "artifacts"
    mix_file.write_text(json.dumps({"mixes": [{}]}), encoding="utf-8")
    config_file.write_text("name: fake\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(CONTROL_PLANE),
            "train",
            "--scheduler-mode",
            "local",
            "--gpu-ids",
            "0",
            "--control-state-file",
            str(control_state),
            "--slots-file",
            str(slots_file),
            "--executor-state-file",
            str(executor_state),
            "--config",
            str(config_file),
            "--mix-file",
            str(mix_file),
            "--group-id",
            "integration",
            "--run-name-prefix",
            "it-train",
            "--mix-start",
            "0",
            "--mix-end",
            "0",
            "--workdir",
            str(REPO_ROOT),
            "--variant-script",
            str(FAKE_VARIANT),
            "--log-dir",
            str(log_dir),
            "--summary-dir",
            str(summary_dir),
            "--output-root-dir",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    summary_path = summary_dir / "it-train-0000.json"
    assert summary_path.exists()
    control_payload = json.loads(control_state.read_text(encoding="utf-8"))
    executor_payload = json.loads(executor_state.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert control_payload["phases"]["train"]["status"] == "completed"
    assert control_payload["phases"]["train"]["slot_plan"]["allocated_workers"] == 1
    assert executor_payload["tasks"]["0"]["status"] == "succeeded"
    assert summary_payload["cuda_visible_devices"] == "0"


def test_control_plane_eval_local_integration(tmp_path):
    _ensure_executable(FAKE_OLMES)
    hf_root = tmp_path / "hf_models" / "model-a"
    hf_root.mkdir(parents=True)
    raw_results_dir = tmp_path / "raw_olmes"
    log_dir = tmp_path / "eval_logs"
    manifest = tmp_path / "hf_models_manifest.csv"
    control_state = tmp_path / "control_plane_state.json"
    executor_state = tmp_path / "parallel_eval_state.json"
    slots_file = tmp_path / "eval_slot_plan.json"

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_name", "mix_index", "global_step", "checkpoint_step_dir", "hf_model_dir"])
        writer.writerow(["run-a", "0", "1", str(tmp_path / "step0"), str(hf_root)])

    proc = subprocess.run(
        [
            sys.executable,
            str(CONTROL_PLANE),
            "eval",
            "--scheduler-mode",
            "local",
            "--gpu-ids",
            "0",
            "--control-state-file",
            str(control_state),
            "--slots-file",
            str(slots_file),
            "--executor-state-file",
            str(executor_state),
            "--hf-manifest",
            str(manifest),
            "--raw-results-dir",
            str(raw_results_dir),
            "--log-dir",
            str(log_dir),
            "--mix-start",
            "0",
            "--mix-end",
            "0",
            "--workdir",
            str(REPO_ROOT),
            "--olmes-bin",
            str(FAKE_OLMES),
        ],
        cwd=REPO_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    metrics_path = raw_results_dir / "model-a" / "metrics.json"
    assert metrics_path.exists()
    control_payload = json.loads(control_state.read_text(encoding="utf-8"))
    executor_payload = json.loads(executor_state.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert control_payload["phases"]["eval"]["status"] == "completed"
    assert control_payload["phases"]["eval"]["slot_plan"]["allocated_workers"] == 1
    assert executor_payload["tasks"]["0"]["status"] == "success"
    assert len(metrics_payload["tasks"]) > 1
