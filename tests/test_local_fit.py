import json
from pathlib import Path

import pytest

from regmixer.local_fit import run_local_fit
from regmixer.eval.task_standards import MMLU_GROUP_WEIGHTS


def _write_round1a_eval_metrics(csv_path: Path) -> None:
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
    for weights in MMLU_GROUP_WEIGHTS.values():
        for task_name in weights:
            row[f"task__{task_name}"] = 0.0
    row["task__mmlu_abstract_algebra"] = 1.0
    row["task__mmlu_formal_logic"] = 1.0

    header = list(row.keys())
    csv_path.write_text(
        ",".join(header) + "\n" + ",".join(str(row[column]) for column in header) + "\n",
        encoding="utf-8",
    )


def test_run_local_fit_uses_round1a_standard_objective(tmp_path: Path):
    eval_metrics = tmp_path / "eval_metrics.csv"
    output = tmp_path / "p_star.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    _write_round1a_eval_metrics(eval_metrics)

    run_local_fit(
        config_path=config_path,
        eval_metrics=eval_metrics,
        output=output,
        mix_file=None,
        metric_columns=None,
        search_samples=16,
        seed=7,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["objective"] == "maximize_round1a_standard_mean"
    assert result["metric_columns"] == [
        "arc_easy",
        "arc_challenge",
        "boolq",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
        "mmlu_stem",
        "mmlu_other",
        "mmlu_social_sciences",
        "mmlu_humanities",
    ]
    assert pytest.approx(result["best_observed_score"]) == (
        0.10
        + 0.20
        + 0.30
        + 0.40
        + 0.50
        + 0.60
        + 0.70
        + 0.80
        + 0.90
        + MMLU_GROUP_WEIGHTS["mmlu_stem"]["mmlu_abstract_algebra"]
        + MMLU_GROUP_WEIGHTS["mmlu_humanities"]["mmlu_formal_logic"]
    ) / 13.0


def test_run_local_fit_errors_when_objective_cannot_be_inferred(tmp_path: Path):
    eval_metrics = tmp_path / "eval_metrics.csv"
    output = tmp_path / "p_star.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    eval_metrics.write_text(
        "\n".join(
            [
                "run_name,mix_index,weight__actual:high,weight__actual:medium,weight__actual:medium_high,task__foo",
                "run-0,0,0.2,0.3,0.5,0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Could not infer a defined optimization objective"):
        run_local_fit(
            config_path=config_path,
            eval_metrics=eval_metrics,
            output=output,
            mix_file=None,
            metric_columns=None,
            search_samples=16,
            seed=7,
        )
