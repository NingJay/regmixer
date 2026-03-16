import json
from pathlib import Path

from regmixer.eval.task_standards import MMLU_GROUP_WEIGHTS
from regmixer.round1a_visualization import (
    write_fit_comparison_visualizations,
    write_round1a_visualizations,
)


def _write_round1a_eval_metrics(csv_path: Path) -> None:
    rows = [
        {
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
        },
        {
            "run_name": "run-1",
            "mix_index": 1,
            "weight__actual:high": 0.4,
            "weight__actual:medium": 0.4,
            "weight__actual:medium_high": 0.2,
            "task__arc_easy": 0.30,
            "task__arc_challenge": 0.35,
            "task__boolq": 0.45,
            "task__commonsense_qa": 0.50,
            "task__hellaswag": 0.55,
            "task__openbookqa": 0.65,
            "task__piqa": 0.75,
            "task__social_iqa": 0.85,
            "task__winogrande": 0.95,
        },
    ]

    for row in rows:
        for weights in MMLU_GROUP_WEIGHTS.values():
            for task_name in weights:
                row[f"task__{task_name}"] = 0.0
        row["task__mmlu_abstract_algebra"] = 1.0
        row["task__mmlu_formal_logic"] = 1.0

    header = list(rows[0].keys())
    csv_lines = [",".join(header)]
    for row in rows:
        csv_lines.append(",".join(str(row[column]) for column in header))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def _write_fit_json(path: Path, regression_type: str, weights: dict[str, float]) -> None:
    payload = {
        "best_observed_run": "run-1",
        "best_observed_score": 0.6,
        "best_observed_weights": {
            "actual:high": 0.4,
            "actual:medium": 0.4,
            "actual:medium_high": 0.2,
        },
        "method": f"{regression_type}_regression_dirichlet_search",
        "metric_columns": [
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
        ],
        "num_domains": 3,
        "num_metrics": 13,
        "num_runs": 2,
        "objective": "maximize_round1a_standard_mean",
        "p_star_actual_quality": {
            "high": weights["actual:high"],
            "medium": weights["actual:medium"],
            "medium_high": weights["actual:medium_high"],
        },
        "predicted_best_score": 0.61,
        "regression_parameters": [0.1, 0.2, 0.3],
        "regression_type": regression_type,
        "weights": weights,
        "weights_sum": 1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_write_round1a_visualizations_writes_expected_artifacts(tmp_path: Path):
    eval_metrics = tmp_path / "eval_metrics.csv"
    fit_json = tmp_path / "quadratic.json"
    output_dir = tmp_path / "visualizations"

    _write_round1a_eval_metrics(eval_metrics)
    _write_fit_json(
        fit_json,
        "quadratic",
        {"actual:high": 0.5, "actual:medium": 0.3, "actual:medium_high": 0.2},
    )

    manifest = write_round1a_visualizations(
        eval_metrics_path=eval_metrics,
        fit_json_path=fit_json,
        output_dir=output_dir,
        heatmap_top_k=4,
    )

    assert manifest["fit_json_regression_type"] == "quadratic"
    assert (output_dir / "round1a_actual_quality_simplex.png").exists()
    assert (output_dir / "round1a_actual_quality_weight_comparison.png").exists()
    assert (output_dir / "round1a_actual_quality_task_heatmap.png").exists()
    assert (output_dir / "round1a_actual_quality_visualizations.json").exists()


def test_write_fit_comparison_visualizations_writes_summary_and_per_method_artifacts(tmp_path: Path):
    eval_metrics = tmp_path / "eval_metrics.csv"
    compare_summary = tmp_path / "fit_compare" / "comparison_summary.json"
    methods_dir = tmp_path / "fit_compare" / "methods"
    output_dir = tmp_path / "fit_compare" / "visualizations"

    _write_round1a_eval_metrics(eval_metrics)
    quadratic_json = methods_dir / "quadratic.json"
    log_linear_json = methods_dir / "log_linear.json"
    _write_fit_json(
        quadratic_json,
        "quadratic",
        {"actual:high": 0.5, "actual:medium": 0.3, "actual:medium_high": 0.2},
    )
    _write_fit_json(
        log_linear_json,
        "log_linear",
        {"actual:high": 0.2, "actual:medium": 0.2, "actual:medium_high": 0.6},
    )

    compare_summary.parent.mkdir(parents=True, exist_ok=True)
    compare_summary.write_text(
        json.dumps(
            {
                "best_method_by_predicted_score": "quadratic",
                "metric_columns": ["arc_easy"],
                "objective": "maximize_round1a_standard_mean",
                "methods": {
                    "quadratic": {
                        "best_observed_score": 0.6,
                        "output_path": str(quadratic_json),
                        "predicted_best_score": 0.61,
                        "weights": {"actual:high": 0.5, "actual:medium": 0.3, "actual:medium_high": 0.2},
                    },
                    "log_linear": {
                        "best_observed_score": 0.6,
                        "output_path": str(log_linear_json),
                        "predicted_best_score": 0.59,
                        "weights": {"actual:high": 0.2, "actual:medium": 0.2, "actual:medium_high": 0.6},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = write_fit_comparison_visualizations(
        eval_metrics_path=eval_metrics,
        fit_comparison_json_path=compare_summary,
        output_dir=output_dir,
        heatmap_top_k=4,
    )

    assert manifest["best_method_by_predicted_score"] == "quadratic"
    assert (output_dir / "regression_comparison.png").exists()
    assert (output_dir / "quadratic" / "round1a_actual_quality_simplex.png").exists()
    assert (output_dir / "log_linear" / "round1a_actual_quality_simplex.png").exists()
    assert (output_dir / "fit_comparison_visualizations.json").exists()
