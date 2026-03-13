import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd


def load_visualize_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_round1a_results.py"
    spec = importlib.util.spec_from_file_location("visualize_round1a_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visualize_round1a_results = load_visualize_module()


def test_barycentric_to_cartesian_maps_vertices():
    x, y = visualize_round1a_results.barycentric_to_cartesian([1, 0, 0], [0, 1, 0], [0, 0, 1])

    assert x.tolist() == [0.0, 1.0, 0.5]
    assert y.tolist() == [0.0, 0.0, math.sqrt(3) / 2.0]


def test_load_eval_metrics_builds_standard_score_from_repo_task_standard(tmp_path):
    csv_path = tmp_path / "eval_metrics.csv"

    row = {
        "run_name": "run-0",
        "mix_index": 0,
        "weight__actual:high": 0.2,
        "weight__actual:medium": 0.3,
        "weight__actual:medium_high": 0.5,
        "task__arc_easy": 0.10,
        "task__arc_challenge": 0.20,
        "task__boolq": 0.30,
        "task__commonsense_qa": 0.40,
        "task__hellaswag": 0.50,
        "task__openbookqa": 0.60,
        "task__piqa": 0.70,
        "task__social_iqa": 0.80,
        "task__winogrande": 0.90,
    }
    for group_name, weights in visualize_round1a_results.MMLU_GROUP_WEIGHTS.items():
        for task_name in weights:
            row[f"task__{task_name}"] = 0.0
    row["task__mmlu_abstract_algebra"] = 1.0
    row["task__mmlu_formal_logic"] = 1.0

    pd.DataFrame([row]).to_csv(csv_path, index=False)

    df = visualize_round1a_results.load_eval_metrics(str(csv_path))
    loaded = df.iloc[0]

    expected_raw_mean = sum(
        float(value) for key, value in row.items() if key.startswith("task__")
    ) / len([key for key in row if key.startswith("task__")])
    expected_standard_score = (
        0.10
        + 0.20
        + 0.30
        + 0.40
        + 0.50
        + 0.60
        + 0.70
        + 0.80
        + 0.90
        + visualize_round1a_results.MMLU_GROUP_WEIGHTS["mmlu_stem"]["mmlu_abstract_algebra"]
        + visualize_round1a_results.MMLU_GROUP_WEIGHTS["mmlu_humanities"]["mmlu_formal_logic"]
    ) / 13.0

    assert math.isclose(float(loaded["task_mean"]), expected_raw_mean)
    assert math.isclose(float(loaded["raw_task_mean"]), expected_raw_mean)
    assert math.isclose(
        float(loaded["standard__mmlu_stem"]),
        visualize_round1a_results.MMLU_GROUP_WEIGHTS["mmlu_stem"]["mmlu_abstract_algebra"],
    )
    assert math.isclose(
        float(loaded["standard__mmlu_humanities"]),
        visualize_round1a_results.MMLU_GROUP_WEIGHTS["mmlu_humanities"]["mmlu_formal_logic"],
    )
    assert math.isclose(float(loaded["standard__csqa"]), 0.40)
    assert math.isclose(float(loaded["standard__socialiqa"]), 0.80)
    assert math.isclose(float(loaded["standard_score"]), expected_standard_score)
    assert float(loaded["standard_score"]) != float(loaded["raw_task_mean"])


def test_select_top_variable_tasks_prefers_highest_std():
    df = pd.DataFrame(
        {
            "task__stable": [0.1, 0.1, 0.1],
            "task__medium": [0.0, 0.5, 1.0],
            "task__high": [0.0, 1.0, 2.0],
        }
    )

    selected = visualize_round1a_results.select_top_variable_tasks(
        df,
        ["task__stable", "task__medium", "task__high"],
        top_k=2,
    )

    assert selected == ["task__high", "task__medium"]
