import importlib.util
import json
import sys
from pathlib import Path


def load_parallel_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "parallel_eval.py"
    spec = importlib.util.spec_from_file_location("parallel_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parallel_eval = load_parallel_eval_module()


def test_build_task_list_filters_mix_range(tmp_path):
    manifest = tmp_path / "hf_models_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "run_name,mix_index,global_step,checkpoint_step_dir,hf_model_dir",
                "run-a,0,1,/tmp/step0,/tmp/model-a",
                "run-b,2,1,/tmp/step0,relative/model-b",
            ]
        ),
        encoding="utf-8",
    )

    tasks = parallel_eval.build_task_list(str(manifest), mix_start=1, mix_end=2)

    assert [(task.model_name, task.mix_index) for task in tasks] == [("model-b", 2)]
    assert tasks[0].hf_model_dir == str((manifest.parent / "relative/model-b").resolve())


def test_verify_metrics_complete_requires_expected_task_count(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"tasks": [{}, {}, {}]}), encoding="utf-8")

    assert parallel_eval.verify_metrics_complete(metrics_file, expected_tasks=3)
    assert not parallel_eval.verify_metrics_complete(metrics_file, expected_tasks=4)


def test_build_remote_task_command_contains_expected_remote_bootstrap(tmp_path):
    workdir = tmp_path / "regmixer"
    workdir.mkdir()
    cfg = parallel_eval.EvalWorkerConfig(
        scheduler_mode="cluster",
        workdir=str(workdir),
        raw_results_dir=str(workdir / "raw_olmes"),
        log_dir=str(workdir / "eval_logs"),
        batch_size=2,
        force_eval=False,
        tasks=parallel_eval.DEFAULT_OLMES_TASKS,
        remote_workdir=str(workdir),
        passthrough_env={"HF_HOME": "/shared/hf"},
    )
    slot = parallel_eval.WorkerSlot(host="hpcgpu15", gpu_id=3, gpu_uuid="GPU-3")
    task = parallel_eval.EvalTask(
        model_name="round1a-train-0005-step11445-hf",
        hf_model_dir=str(workdir / "hf_models" / "round1a-train-0005-step11445-hf"),
        mix_index=5,
    )

    remote_body, task_cmd = parallel_eval.build_remote_task_command(
        cfg,
        slot,
        task,
        str(workdir / "raw_olmes" / task.model_name),
    )

    assert "source ~/.bashrc" in remote_body
    assert "conda activate regmixer" in remote_body
    assert f"cd {workdir}" in remote_body
    assert "CUDA_VISIBLE_DEVICES=3" in remote_body
    assert "HF_HOME=/shared/hf" in remote_body
    assert "olmes" in task_cmd
    assert task.hf_model_dir in task_cmd


def test_state_file_tracks_cached_completion(tmp_path):
    state_path = tmp_path / "parallel_eval_state.json"
    cfg = parallel_eval.EvalWorkerConfig(
        scheduler_mode="cluster",
        workdir=str(tmp_path),
        raw_results_dir=str(tmp_path / "raw_olmes"),
        log_dir=str(tmp_path / "eval_logs"),
        batch_size=1,
        force_eval=False,
        tasks=parallel_eval.DEFAULT_OLMES_TASKS,
        remote_workdir=str(tmp_path),
    )
    slot = parallel_eval.WorkerSlot(host="hpcgpu09", gpu_id=0, gpu_uuid="GPU-0")
    task = parallel_eval.EvalTask(
        model_name="round1a-train-0000-step11445-hf",
        hf_model_dir=str(tmp_path / "hf_models" / "round1a-train-0000-step11445-hf"),
        mix_index=0,
    )
    state = parallel_eval.initialize_state(
        cfg,
        [slot],
        [task],
        str(tmp_path / "hf_models_manifest.csv"),
        {},
        str(tmp_path / "eval_slot_plan.json"),
    )

    parallel_eval.apply_event_to_state(
        state,
        parallel_eval.build_started_event(
            slot,
            task,
            str(tmp_path / "eval_logs" / "eval-round1a-train-0000-step11445-hf.log"),
            str(tmp_path / "raw_olmes" / task.model_name),
        ),
    )
    parallel_eval.apply_event_to_state(
        state,
        parallel_eval.complete_event(
            status="cached",
            slot=slot,
            task=task,
            log_path=str(tmp_path / "eval_logs" / "eval-round1a-train-0000-step11445-hf.log"),
            output_dir=str(tmp_path / "raw_olmes" / task.model_name),
            return_code=0,
            duration_sec=0.0,
        ),
    )
    parallel_eval.write_state_file(str(state_path), state)

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["slot_plan_file"].endswith("eval_slot_plan.json")
    assert payload["tasks"]["0"]["status"] == "cached"
    assert payload["tasks"]["0"]["host"] == "hpcgpu09"
    assert payload["tasks"]["0"]["return_code"] == 0
