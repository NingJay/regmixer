import math
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


def _write_explicit_eval_metrics(csv_path: Path) -> None:
    csv_path.write_text(
        "\n".join(
            [
                "run_name,mix_index,weight__actual:high,weight__actual:medium,metric__score",
                "run-0,0,0.9,0.1,0.2",
                "run-1,1,0.1,0.9,0.8",
                "run-2,2,0.5,0.5,0.6",
            ]
        ),
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
        output_dir=None,
        mix_file=None,
        metric_columns=None,
        search_samples=16,
        seed=7,
        regression_type="quadratic",
        compare_regressions=False,
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
            output_dir=None,
            mix_file=None,
            metric_columns=None,
            search_samples=16,
            seed=7,
            regression_type="quadratic",
            compare_regressions=False,
        )


class _StubRegressor:
    def __init__(self, target: tuple[float, float], peak_score: float):
        self.target = target
        self.peak_score = peak_score
        self.model = [peak_score, *target]

    def predict(self, x):
        target = self.target
        return self.peak_score - ((x[:, 0] - target[0]) ** 2 + (x[:, 1] - target[1]) ** 2)


def test_run_local_fit_compares_quadratic_log_linear_and_lightgbm(tmp_path: Path, monkeypatch):
    eval_metrics = tmp_path / "eval_metrics.csv"
    summary_output = tmp_path / "comparison_summary.json"
    output_dir = tmp_path / "fit_outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    _write_explicit_eval_metrics(eval_metrics)

    def _fake_fit_eval_regression(regression_type, x, y):
        del x, y
        if regression_type == "log_linear":
            return _StubRegressor((0.1, 0.9), 0.95)
        if regression_type == "lightgbm":
            return _StubRegressor((0.5, 0.5), 0.85)
        raise AssertionError(f"unexpected regression type: {regression_type}")

    monkeypatch.setattr("regmixer.local_fit._fit_eval_regression", _fake_fit_eval_regression)

    run_local_fit(
        config_path=config_path,
        eval_metrics=eval_metrics,
        output=summary_output,
        output_dir=output_dir,
        mix_file=None,
        metric_columns="metric__score",
        search_samples=32,
        seed=7,
        regression_type="quadratic",
        compare_regressions=True,
    )

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["best_method_by_predicted_score"] == "log_linear"
    assert set(summary["methods"]) == {"quadratic", "log_linear", "lightgbm"}

    quadratic_result = json.loads((output_dir / "methods" / "quadratic.json").read_text(encoding="utf-8"))
    log_linear_result = json.loads((output_dir / "methods" / "log_linear.json").read_text(encoding="utf-8"))
    lightgbm_result = json.loads((output_dir / "methods" / "lightgbm.json").read_text(encoding="utf-8"))

    assert quadratic_result["regression_type"] == "quadratic"
    assert log_linear_result["regression_type"] == "log_linear"
    assert lightgbm_result["regression_type"] == "lightgbm"
    assert log_linear_result["weights"] == {"actual:high": 0.1, "actual:medium": 0.9}
    assert lightgbm_result["weights"] == {"actual:high": 0.5, "actual:medium": 0.5}


def test_run_local_fit_uses_output_dir_for_single_method(tmp_path: Path, monkeypatch):
    eval_metrics = tmp_path / "eval_metrics.csv"
    output = tmp_path / "proposal.json"
    output_dir = tmp_path / "single_fit"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    _write_explicit_eval_metrics(eval_metrics)

    monkeypatch.setattr(
        "regmixer.local_fit._fit_eval_regression",
        lambda regression_type, x, y: _StubRegressor((0.1, 0.9), 0.95),
    )

    result_path = run_local_fit(
        config_path=config_path,
        eval_metrics=eval_metrics,
        output=output,
        output_dir=output_dir,
        mix_file=None,
        metric_columns="metric__score",
        search_samples=16,
        seed=7,
        regression_type="log_linear",
        compare_regressions=False,
    )

    assert result_path == output_dir / "proposal.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["regression_type"] == "log_linear"


@pytest.mark.filterwarnings(
    "ignore:Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior"
)
@pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)
def test_run_local_fit_real_regressors_compare_smoke(tmp_path: Path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")

    eval_metrics = tmp_path / "eval_metrics.csv"
    summary_output = tmp_path / "comparison_summary.json"
    output_dir = tmp_path / "fit_outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    _write_explicit_eval_metrics(eval_metrics)

    result_path = run_local_fit(
        config_path=config_path,
        eval_metrics=eval_metrics,
        output=summary_output,
        output_dir=output_dir,
        mix_file=None,
        metric_columns="metric__score",
        search_samples=32,
        seed=7,
        regression_type="quadratic",
        compare_regressions=True,
    )

    assert result_path == summary_output
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert set(summary["methods"]) == {"quadratic", "log_linear", "lightgbm"}

    for regression_type in ("quadratic", "log_linear", "lightgbm"):
        payload = json.loads((output_dir / "methods" / f"{regression_type}.json").read_text(encoding="utf-8"))
        assert payload["regression_type"] == regression_type
        assert payload["weights_sum"] == pytest.approx(1.0)
        assert math.isfinite(payload["predicted_best_score"])
