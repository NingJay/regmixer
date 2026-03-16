from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regmixer.eval.task_standards import (
    MMLU_GROUP_WEIGHTS,
    ROUND1A_STANDARD_METRICS,
    build_round1a_standard_metrics,
)

WEIGHT_COLUMNS = (
    "weight__actual:high",
    "weight__actual:medium",
    "weight__actual:medium_high",
)
STANDARD_METRIC_PREFIX = "standard__"
STANDARD_SCORE_COLUMN = "standard_score"
RAW_SCORE_COLUMN = "raw_task_mean"


def load_eval_metrics(eval_metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_metrics_path)
    task_columns = extract_task_columns(df)
    if not task_columns:
        raise RuntimeError(f"No task__* columns found in {eval_metrics_path}")

    missing_weights = [col for col in WEIGHT_COLUMNS if col not in df.columns]
    if missing_weights:
        raise RuntimeError(
            f"round1a visualization requires actual quality weights, missing: {missing_weights}"
        )

    df = df.copy()
    for column in [*WEIGHT_COLUMNS, *task_columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "task_mean" in df.columns:
        df["task_mean"] = pd.to_numeric(df["task_mean"], errors="coerce")
    else:
        df["task_mean"] = df[task_columns].mean(axis=1)
    df[RAW_SCORE_COLUMN] = df["task_mean"]

    task_scores = df.loc[:, task_columns].copy()
    task_scores.columns = [column.replace("task__", "", 1) for column in task_scores.columns]
    try:
        standard_metrics = build_round1a_standard_metrics(task_scores)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to build round1a standard metrics from {eval_metrics_path}"
        ) from exc

    for metric_name in standard_metrics.columns:
        df[f"{STANDARD_METRIC_PREFIX}{metric_name}"] = standard_metrics[metric_name]
    df[STANDARD_SCORE_COLUMN] = standard_metrics.mean(axis=1)

    df = df.dropna(subset=[*WEIGHT_COLUMNS, RAW_SCORE_COLUMN, STANDARD_SCORE_COLUMN])
    if df.empty:
        raise RuntimeError("No valid rows remain after coercing task/weight columns to numeric.")

    return df


def load_fit_result(fit_json_path: Path) -> Dict[str, object]:
    return json.loads(fit_json_path.read_text(encoding="utf-8"))


def barycentric_to_cartesian(
    high: Iterable[float],
    medium: Iterable[float],
    medium_high: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    high_arr = np.asarray(list(high), dtype=np.float64)
    medium_arr = np.asarray(list(medium), dtype=np.float64)
    medium_high_arr = np.asarray(list(medium_high), dtype=np.float64)
    weights_sum = high_arr + medium_arr + medium_high_arr
    if np.any(weights_sum <= 0):
        raise ValueError("all barycentric coordinates must have positive sum")

    x = (medium_arr + 0.5 * medium_high_arr) / weights_sum
    y = ((math.sqrt(3) / 2.0) * medium_high_arr) / weights_sum
    return x, y


def extract_task_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("task__")]


def extract_standard_metric_columns(df: pd.DataFrame) -> List[str]:
    return [
        f"{STANDARD_METRIC_PREFIX}{metric_name}"
        for metric_name in ROUND1A_STANDARD_METRICS
        if f"{STANDARD_METRIC_PREFIX}{metric_name}" in df.columns
    ]


def select_top_variable_tasks(
    df: pd.DataFrame,
    task_columns: Sequence[str],
    top_k: int,
) -> List[str]:
    if top_k <= 0:
        raise ValueError("heatmap top-k must be positive")
    ranked = (
        df.loc[:, task_columns]
        .std(axis=0, numeric_only=True)
        .sort_values(ascending=False)
        .index.tolist()
    )
    return ranked[: min(top_k, len(ranked))]


def get_best_standard_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df[STANDARD_SCORE_COLUMN].idxmax()]


def describe_fit_objective(fit_result: Dict[str, object]) -> str:
    objective = str(fit_result.get("objective", "") or "")
    regression_type = str(fit_result.get("regression_type", "") or "")
    regression_label = regression_type.replace("_", "-") if regression_type else ""

    if objective == "maximize_round1a_standard_mean":
        base_label = "Round1a standard fit p*"
    elif objective == "maximize_mean_metric":
        base_label = "Legacy mean-metric fit p*"
    elif objective:
        base_label = f"{objective} p*"
    else:
        base_label = "Fit p*"

    if regression_label:
        return f"{base_label} ({regression_label})"
    return base_label


def plot_simplex(df: pd.DataFrame, fit_result: Dict[str, object], output_path: Path) -> None:
    x, y = barycentric_to_cartesian(
        df["weight__actual:high"],
        df["weight__actual:medium"],
        df["weight__actual:medium_high"],
    )
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")

    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color="#2f2a24", linewidth=2)

    scatter = ax.scatter(
        x,
        y,
        c=df[STANDARD_SCORE_COLUMN],
        cmap="YlOrRd",
        s=130,
        edgecolors="#2f2a24",
        linewidths=0.8,
        alpha=0.95,
    )

    for _, row in df.iterrows():
        point_x, point_y = barycentric_to_cartesian(
            [row["weight__actual:high"]],
            [row["weight__actual:medium"]],
            [row["weight__actual:medium_high"]],
        )
        ax.text(
            float(point_x[0]) + 0.012,
            float(point_y[0]) + 0.008,
            f"{int(row['mix_index']):02d}",
            fontsize=8,
            color="#2f2a24",
        )

    best = get_best_standard_row(df)
    best_x, best_y = barycentric_to_cartesian(
        [best["weight__actual:high"]],
        [best["weight__actual:medium"]],
        [best["weight__actual:medium_high"]],
    )
    ax.scatter(
        best_x,
        best_y,
        marker="*",
        s=360,
        color="#005f73",
        edgecolors="white",
        linewidths=1.4,
        zorder=5,
        label=f"Best standard ({best['run_name']}, {best[STANDARD_SCORE_COLUMN]:.4f})",
    )

    p_star = fit_result.get("p_star_actual_quality") or {}
    if p_star:
        fit_label = describe_fit_objective(fit_result)
        pred_x, pred_y = barycentric_to_cartesian(
            [float(p_star.get("high", 0.0))],
            [float(p_star.get("medium", 0.0))],
            [float(p_star.get("medium_high", 0.0))],
        )
        ax.scatter(
            pred_x,
            pred_y,
            marker="X",
            s=220,
            color="#0a9396",
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
            label=fit_label,
        )

    ax.text(-0.02, -0.04, "High", fontsize=12, fontweight="bold")
    ax.text(1.02, -0.04, "Medium", fontsize=12, fontweight="bold", ha="right")
    ax.text(
        0.5,
        math.sqrt(3) / 2.0 + 0.04,
        "Medium High",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    ax.set_title("Round1a Standardized Actual Quality Search on the 3-Way Simplex", fontsize=16, pad=18)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2.0 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.03)
    colorbar.set_label("Round1a standard score", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_weight_comparison(df: pd.DataFrame, fit_result: Dict[str, object], output_path: Path) -> None:
    best_row = get_best_standard_row(df)
    fit_label = describe_fit_objective(fit_result)

    predicted = fit_result.get("p_star_actual_quality") or {}
    predicted_values = np.array(
        [
            float(predicted.get("high", 0.0)),
            float(predicted.get("medium", 0.0)),
            float(predicted.get("medium_high", 0.0)),
        ],
        dtype=np.float64,
    )
    observed_values = np.array(
        [
            float(best_row["weight__actual:high"]),
            float(best_row["weight__actual:medium"]),
            float(best_row["weight__actual:medium_high"]),
        ],
        dtype=np.float64,
    )

    labels = ["High", "Medium", "Medium High"]
    y = np.arange(len(labels))
    height = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.barh(y - height / 2, observed_values, height=height, color="#bb3e03", label="Best standard")
    ax.barh(y + height / 2, predicted_values, height=height, color="#0a9396", label=fit_label)

    for idx, value in enumerate(observed_values):
        ax.text(value + 0.01, y[idx] - height / 2, f"{value:.3f}", va="center", fontsize=10)
    for idx, value in enumerate(predicted_values):
        ax.text(value + 0.01, y[idx] + height / 2, f"{value:.3f}", va="center", fontsize=10)

    title = (
        f"Best Standard Mix vs {fit_label}\n"
        f"standard_best={best_row[STANDARD_SCORE_COLUMN]:.4f}"
    )
    if "predicted_best_score" in fit_result:
        title += f"  raw_fit_pred={float(fit_result['predicted_best_score']):.4f}"

    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mixture weight")
    ax.set_title(title, fontsize=15, pad=14)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_task_heatmap(df: pd.DataFrame, top_tasks: Sequence[str], output_path: Path) -> None:
    ranked = df.sort_values(STANDARD_SCORE_COLUMN, ascending=False).reset_index(drop=True)
    matrix = ranked.loc[:, top_tasks].to_numpy(dtype=np.float64)
    row_labels = [task.replace(STANDARD_METRIC_PREFIX, "") for task in top_tasks]
    col_labels = [
        f"{name[-4:]} ({score:.3f})"
        for name, score in zip(ranked["run_name"], ranked[STANDARD_SCORE_COLUMN])
    ]

    fig_width = max(10.0, 0.48 * len(col_labels))
    fig_height = max(8.0, 0.42 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
    image = ax.imshow(matrix.T, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels, fontsize=9)
    ax.set_title(
        f"Round1a Standard Metric Heatmap (Top {len(top_tasks)} Most Variable Metrics)",
        fontsize=15,
        pad=14,
    )
    ax.set_xlabel("Runs ranked by round1a standard score")
    ax.set_ylabel("Standard metrics")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    colorbar.set_label("Standard metric score", fontsize=10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fit_comparison_summary(
    method_results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> Path:
    methods = list(method_results.keys())
    domains = list(next(iter(method_results.values()))["weights"].keys())

    predicted_scores = [float(method_results[method]["predicted_best_score"]) for method in methods]
    observed_scores = [float(method_results[method]["best_observed_score"]) for method in methods]
    weight_matrix = np.array(
        [[float(method_results[method]["weights"][domain]) for domain in domains] for method in methods],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [1.0, 1.4]},
        layout="constrained",
    )

    score_ax = axes[0]
    x = np.arange(len(methods))
    width = 0.36
    score_ax.bar(x - width / 2, observed_scores, width=width, label="best observed", color="#8c7b6b")
    score_ax.bar(
        x + width / 2,
        predicted_scores,
        width=width,
        label="predicted best",
        color="#d95f0e",
    )
    score_ax.set_xticks(x, [method.replace("_", "-") for method in methods])
    score_ax.set_ylabel("Score")
    score_ax.set_title("Regression Fit Comparison")
    score_ax.legend(frameon=False)

    weight_ax = axes[1]
    bottom = np.zeros(len(methods), dtype=np.float64)
    palette = ["#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    for idx, domain in enumerate(domains):
        values = weight_matrix[:, idx]
        weight_ax.bar(
            [method.replace("_", "-") for method in methods],
            values,
            bottom=bottom,
            label=domain,
            color=palette[idx % len(palette)],
        )
        bottom += values
    weight_ax.set_ylim(0.0, 1.0)
    weight_ax.set_ylabel("Proposed weight")
    weight_ax.set_title("Proposed Mixture Weights")
    weight_ax.legend(frameon=False, ncol=min(3, len(domains)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_manifest(
    df: pd.DataFrame,
    fit_result: Dict[str, object],
    top_tasks: Sequence[str],
    output_dir: Path,
) -> Dict[str, object]:
    best_row = get_best_standard_row(df)
    return {
        "aggregation": {
            "name": "round1a_standard_mean",
            "description": (
                "Equal average over ALL_CORE_TASKS plus the four standard MMLU aggregates "
                "defined by regmixer eval utilities."
            ),
            "standard_metrics": list(ROUND1A_STANDARD_METRICS),
            "mmlu_groups": list(MMLU_GROUP_WEIGHTS.keys()),
        },
        "best_standard_run": str(best_row["run_name"]),
        "best_standard_score": float(best_row[STANDARD_SCORE_COLUMN]),
        "fit_json_best_observed_run": fit_result.get("best_observed_run"),
        "fit_json_best_observed_score": fit_result.get("best_observed_score"),
        "fit_json_predicted_best_score": fit_result.get("predicted_best_score"),
        "fit_json_regression_type": fit_result.get("regression_type"),
        "num_runs": int(len(df)),
        "num_raw_tasks": int(len(extract_task_columns(df))),
        "num_standard_metrics": int(len(extract_standard_metric_columns(df))),
        "top_variable_standard_metrics": [task.replace(STANDARD_METRIC_PREFIX, "") for task in top_tasks],
        "artifacts": {
            "simplex": str(output_dir / "round1a_actual_quality_simplex.png"),
            "weight_comparison": str(output_dir / "round1a_actual_quality_weight_comparison.png"),
            "task_heatmap": str(output_dir / "round1a_actual_quality_task_heatmap.png"),
        },
    }


def write_round1a_visualizations(
    *,
    eval_metrics_path: Path,
    fit_json_path: Path,
    output_dir: Path,
    heatmap_top_k: int = 24,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_metrics(eval_metrics_path)
    fit_result = load_fit_result(fit_json_path)
    standard_metric_columns = extract_standard_metric_columns(df)
    top_tasks = select_top_variable_tasks(df, standard_metric_columns, heatmap_top_k)

    simplex_path = output_dir / "round1a_actual_quality_simplex.png"
    weight_comparison_path = output_dir / "round1a_actual_quality_weight_comparison.png"
    heatmap_path = output_dir / "round1a_actual_quality_task_heatmap.png"
    manifest_path = output_dir / "round1a_actual_quality_visualizations.json"

    plot_simplex(df, fit_result, simplex_path)
    plot_weight_comparison(df, fit_result, weight_comparison_path)
    plot_task_heatmap(df, top_tasks, heatmap_path)

    manifest = build_manifest(df, fit_result, top_tasks, output_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def write_fit_comparison_visualizations(
    *,
    eval_metrics_path: Path,
    fit_comparison_json_path: Path,
    output_dir: Path,
    heatmap_top_k: int = 24,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_summary = json.loads(fit_comparison_json_path.read_text(encoding="utf-8"))
    method_payloads: Dict[str, Dict[str, object]] = {}
    method_manifests: Dict[str, Dict[str, object]] = {}

    for method_name, method_summary in comparison_summary["methods"].items():
        fit_json_path = Path(method_summary["output_path"])
        method_payload = load_fit_result(fit_json_path)
        method_payloads[method_name] = method_payload
        method_output_dir = output_dir / method_name
        method_manifests[method_name] = write_round1a_visualizations(
            eval_metrics_path=eval_metrics_path,
            fit_json_path=fit_json_path,
            output_dir=method_output_dir,
            heatmap_top_k=heatmap_top_k,
        )

    comparison_plot_path = plot_fit_comparison_summary(
        method_payloads,
        output_dir / "regression_comparison.png",
    )
    manifest = {
        "fit_comparison_json": str(fit_comparison_json_path),
        "best_method_by_predicted_score": comparison_summary["best_method_by_predicted_score"],
        "artifacts": {
            "comparison_plot": str(comparison_plot_path),
            "per_method_visualizations": {
                method_name: method_manifests[method_name]["artifacts"]
                for method_name in method_manifests
            },
        },
        "methods": {
            method_name: {
                "fit_json": comparison_summary["methods"][method_name]["output_path"],
                "predicted_best_score": comparison_summary["methods"][method_name]["predicted_best_score"],
                "visualization_manifest": str(output_dir / method_name / "round1a_actual_quality_visualizations.json"),
            }
            for method_name in comparison_summary["methods"]
        },
    }
    manifest_path = output_dir / "fit_comparison_visualizations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest
