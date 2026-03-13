from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from regmixer.eval.task_standards import ROUND1A_STANDARD_METRICS, build_round1a_standard_metrics

logger = logging.getLogger(__name__)


def _infer_mix_file(eval_metrics: Path, mix_file: Optional[Path]) -> Path:
    if mix_file:
        return mix_file

    output_dir = eval_metrics.resolve().parent
    candidates = sorted(output_dir.glob("*mixes.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not infer mix file under {output_dir}. Provide --mix-file explicitly."
        )
    raise ValueError(
        "Found multiple mix files; provide --mix-file explicitly: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_mix_weights(mix_file: Path) -> dict[int, dict[str, float]]:
    with mix_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        idx: {name: float(value[0]) for name, value in mix.items()}
        for idx, mix in enumerate(payload["mixes"])
    }


def _inject_weight_columns(df: pd.DataFrame, mix_file: Path) -> pd.DataFrame:
    if "mix_index" not in df.columns:
        raise KeyError("eval metrics CSV must contain a 'mix_index' column")

    mix_weights = _load_mix_weights(mix_file)
    all_domains: list[str] = []
    for idx in sorted(mix_weights.keys()):
        for domain in mix_weights[idx]:
            if domain not in all_domains:
                all_domains.append(domain)

    for domain in all_domains:
        col = f"weight__{domain}"
        df[col] = df["mix_index"].map(lambda i: mix_weights.get(int(i), {}).get(domain, np.nan))

    return df


def _quadratic_features(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape={weights.shape}")

    n_samples, n_domains = weights.shape
    pieces = [weights, weights**2]

    interactions = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            interactions.append((weights[:, i] * weights[:, j])[:, None])
    if interactions:
        pieces.append(np.concatenate(interactions, axis=1))

    design = np.concatenate(pieces, axis=1)
    assert design.shape[0] == n_samples
    return design


def _fit_quadratic_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = _quadratic_features(x)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs


def _predict_quadratic(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    design = _quadratic_features(x)
    return design @ coeffs


def _shrink_actual_prefix(weights: dict[str, float]) -> Optional[dict[str, float]]:
    if not weights:
        return None
    if not all(key.startswith("actual:") for key in weights):
        return None
    return {key.split(":", 1)[1]: value for key, value in weights.items()}


def _resolve_metric_frame(
    df: pd.DataFrame, metric_columns: Optional[str]
) -> tuple[pd.DataFrame, list[str], str]:
    if metric_columns:
        metrics = [metric.strip() for metric in metric_columns.split(",") if metric.strip()]
        if not metrics:
            raise RuntimeError("--metric-columns was provided but no metric names were parsed.")
        missing_metrics = [metric for metric in metrics if metric not in df.columns]
        if missing_metrics:
            raise RuntimeError(f"Explicit metric columns were not found in eval CSV: {missing_metrics}")
        return df.loc[:, metrics].copy(), metrics, "maximize_mean_metric"

    task_columns = [col for col in df.columns if col.startswith("task__")]
    if not task_columns:
        raise RuntimeError(
            "No task__* columns found. Provide --metric-columns or export a standards-compatible eval CSV."
        )

    task_scores = df.loc[:, task_columns].copy()
    task_scores.columns = [column.replace("task__", "", 1) for column in task_scores.columns]
    try:
        standard_metrics = build_round1a_standard_metrics(task_scores)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "Could not infer a defined optimization objective from eval CSV. "
            "Provide --metric-columns explicitly or export the full round1a standard task set."
        ) from exc

    return standard_metrics, list(ROUND1A_STANDARD_METRICS), "maximize_round1a_standard_mean"


def run_local_fit(
    *,
    config_path: Path,
    eval_metrics: Path,
    output: Path,
    mix_file: Optional[Path],
    metric_columns: Optional[str],
    search_samples: int,
    seed: int,
) -> Path:
    config_path = Path(config_path)
    eval_metrics = Path(eval_metrics)
    output = Path(output)
    mix_file = Path(mix_file) if mix_file is not None else None
    del config_path  # kept for interface compatibility with existing run scripts

    df = pd.read_csv(eval_metrics)
    if df.empty:
        raise RuntimeError(f"eval metrics CSV is empty: {eval_metrics}")

    weight_columns = [col for col in df.columns if col.startswith("weight__")]
    if not weight_columns:
        resolved_mix_file = _infer_mix_file(eval_metrics, mix_file)
        logger.info("Injecting weight columns from %s", resolved_mix_file)
        df = _inject_weight_columns(df, resolved_mix_file)
        weight_columns = [col for col in df.columns if col.startswith("weight__")]

    if not weight_columns:
        raise RuntimeError("No weight columns found or derived for fit-mixture.")

    df = df.copy()
    df[weight_columns] = df[weight_columns].apply(pd.to_numeric, errors="coerce")
    metric_frame, metrics, objective_name = _resolve_metric_frame(df, metric_columns)
    metric_frame = metric_frame.apply(pd.to_numeric, errors="coerce")
    for column in metric_frame.columns:
        df[column] = metric_frame[column]
    df = df.dropna(subset=weight_columns + metrics)
    if df.empty:
        raise RuntimeError("No valid rows remain after dropping NaN metrics/weights.")

    df["mean_score"] = df[metrics].mean(axis=1)
    x = df[weight_columns].to_numpy(dtype=np.float64)
    y = df["mean_score"].to_numpy(dtype=np.float64)

    if x.shape[1] == 1:
        proposed = np.array([1.0], dtype=np.float64)
        coeffs = np.array([0.0], dtype=np.float64)
        predicted = float(y.mean())
        method = "single_domain_no_regression"
    else:
        coeffs = _fit_quadratic_regression(x, y)
        rng = np.random.default_rng(seed)
        random_candidates = rng.dirichlet(np.ones(x.shape[1]), size=max(1, int(search_samples)))
        corners = np.eye(x.shape[1])
        candidates = np.concatenate([x, random_candidates, corners], axis=0)
        predictions = _predict_quadratic(candidates, coeffs)
        best_idx = int(np.argmax(predictions))
        proposed = candidates[best_idx]
        predicted = float(predictions[best_idx])
        method = "quadratic_regression_dirichlet_search"

    best_observed_idx = int(np.argmax(y))
    best_observed_row = df.iloc[best_observed_idx]
    best_observed_weights = {
        col.replace("weight__", ""): float(best_observed_row[col]) for col in weight_columns
    }

    proposed_weights = {
        col.replace("weight__", ""): float(value) for col, value in zip(weight_columns, proposed)
    }
    topic_only = _shrink_actual_prefix(proposed_weights)

    result = {
        "method": method,
        "objective": objective_name,
        "num_runs": int(len(df)),
        "num_domains": int(len(weight_columns)),
        "num_metrics": int(len(metrics)),
        "metric_columns": metrics,
        "best_observed_run": str(best_observed_row.get("run_name", "")),
        "best_observed_score": float(best_observed_row["mean_score"]),
        "best_observed_weights": best_observed_weights,
        "predicted_best_score": predicted,
        "weights": proposed_weights,
        "weights_sum": float(sum(proposed_weights.values())),
        "regression_coefficients": coeffs.tolist(),
    }
    if topic_only is not None:
        result["p_star_actual_quality"] = topic_only

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    logger.info("Local fit output written to %s", output)
    return output
