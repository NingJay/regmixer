from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from regmixer.eval.task_standards import ROUND1A_STANDARD_METRICS, build_round1a_standard_metrics

logger = logging.getLogger(__name__)

SUPPORTED_REGRESSION_TYPES = ("quadratic", "log_linear", "lightgbm")
DEFAULT_COMPARE_REGRESSION_TYPES = ("quadratic", "log_linear", "lightgbm")

METHOD_NAMES = {
    "quadratic": "quadratic_regression_dirichlet_search",
    "log_linear": "log_linear_regression_dirichlet_search",
    "lightgbm": "lightgbm_regression_dirichlet_search",
}


@dataclass(frozen=True)
class FitOutputPaths:
    summary_output: Path
    method_outputs: dict[str, Path]


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


def _fit_eval_regression(regression_type: str, x: np.ndarray, y: np.ndarray) -> Any:
    try:
        from regmixer.regression_models import build_local_regression
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Regression type '{regression_type}' requires optional dependency '{exc.name}'. "
            "Activate the regmixer environment before running fit-mixture."
        ) from exc

    return build_local_regression(regression_type=regression_type, x=x, y=y)


def _predict_eval_regression(model: Any, x: np.ndarray) -> np.ndarray:
    predictions = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    if predictions.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"Regression model returned {predictions.shape[0]} predictions for {x.shape[0]} rows."
        )
    return predictions


def _resolve_regression_types(
    regression_type: str,
    compare_regressions: bool,
) -> list[str]:
    if regression_type not in SUPPORTED_REGRESSION_TYPES:
        raise ValueError(
            f"Unsupported regression type '{regression_type}'. "
            f"Expected one of: {', '.join(SUPPORTED_REGRESSION_TYPES)}"
        )

    if compare_regressions:
        return list(DEFAULT_COMPARE_REGRESSION_TYPES)

    return [regression_type]


def _build_candidate_weights(x: np.ndarray, search_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_candidates = rng.dirichlet(np.ones(x.shape[1]), size=max(1, int(search_samples)))
    corners = np.eye(x.shape[1])
    return np.concatenate([x, random_candidates, corners], axis=0)


def _extract_regression_parameters(
    regression_type: str,
    model_or_coeffs: Any,
) -> Optional[list[float]]:
    if regression_type == "quadratic":
        return np.asarray(model_or_coeffs, dtype=np.float64).tolist()

    raw_model = getattr(model_or_coeffs, "model", None)
    if raw_model is None:
        return None
    if isinstance(raw_model, np.ndarray):
        return raw_model.astype(np.float64).tolist()
    if isinstance(raw_model, (list, tuple)):
        return [float(value) for value in raw_model]

    params = getattr(raw_model, "params", None)
    if params is not None:
        return np.asarray(params, dtype=np.float64).tolist()

    return None


def _extract_regression_metadata(regression_type: str, model: Any) -> Optional[dict[str, Any]]:
    if regression_type != "lightgbm":
        return None

    estimator = getattr(model, "model", None)
    if estimator is None:
        return None

    metadata: dict[str, Any] = {}
    feature_importances = getattr(estimator, "feature_importances_", None)
    if feature_importances is not None:
        metadata["feature_importances"] = np.asarray(feature_importances, dtype=np.int64).tolist()

    n_estimators = getattr(estimator, "n_estimators_", None)
    if n_estimators is not None:
        metadata["n_estimators"] = int(n_estimators)

    return metadata or None


def _search_best_weights(
    *,
    regression_type: str,
    x: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, float, str, Optional[list[float]], Optional[dict[str, Any]]]:
    if regression_type == "quadratic":
        coeffs = _fit_quadratic_regression(x, y)
        predictions = _predict_quadratic(candidates, coeffs)
        coeff_payload = _extract_regression_parameters(regression_type, coeffs)
        metadata = None
    else:
        model = _fit_eval_regression(regression_type, x, y)
        predictions = _predict_eval_regression(model, candidates)
        coeff_payload = _extract_regression_parameters(regression_type, model)
        metadata = _extract_regression_metadata(regression_type, model)

    if not np.isfinite(predictions).any():
        raise RuntimeError(f"Regression type '{regression_type}' produced no finite predictions.")

    safe_predictions = np.where(np.isfinite(predictions), predictions, -np.inf)
    best_idx = int(np.argmax(safe_predictions))
    return (
        candidates[best_idx],
        float(safe_predictions[best_idx]),
        METHOD_NAMES[regression_type],
        coeff_payload,
        metadata,
    )


def _resolve_output_paths(
    *,
    output: Path,
    output_dir: Optional[Path],
    regression_types: Sequence[str],
    compare_regressions: bool,
) -> FitOutputPaths:
    output = Path(output)
    artifact_root = Path(output_dir) if output_dir is not None else None

    if compare_regressions:
        summary_path = output if output.suffix else output / "comparison.json"
        if artifact_root is None:
            artifact_root = output if not output.suffix else output.parent / output.stem
        methods_dir = artifact_root / "methods"
        method_outputs = {
            regression_name: methods_dir / f"{regression_name}.json"
            for regression_name in regression_types
        }
        return FitOutputPaths(summary_output=summary_path, method_outputs=method_outputs)

    regression_name = regression_types[0]
    if artifact_root is not None:
        filename = output.name if output.suffix else f"{regression_name}.json"
        result_path = artifact_root / filename
    elif output.suffix:
        result_path = output
    else:
        result_path = output / f"{regression_name}.json"

    return FitOutputPaths(summary_output=result_path, method_outputs={regression_name: result_path})


def _build_result_summary(
    *,
    objective_name: str,
    metrics: list[str],
    results: dict[str, dict[str, Any]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    best_method = max(
        results.items(),
        key=lambda item: item[1]["predicted_best_score"],
    )[0]

    summary = {
        "objective": objective_name,
        "metric_columns": metrics,
        "methods": {
            regression_name: {
                "output_path": str(output_paths[regression_name]),
                "predicted_best_score": result["predicted_best_score"],
                "best_observed_score": result["best_observed_score"],
                "weights": result["weights"],
            }
            for regression_name, result in results.items()
        },
        "best_method_by_predicted_score": best_method,
    }
    return summary


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
    output_dir: Optional[Path],
    mix_file: Optional[Path],
    metric_columns: Optional[str],
    search_samples: int,
    seed: int,
    regression_type: str,
    compare_regressions: bool,
) -> Path:
    config_path = Path(config_path)
    eval_metrics = Path(eval_metrics)
    output = Path(output)
    output_dir = Path(output_dir) if output_dir is not None else None
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

    regression_types = _resolve_regression_types(regression_type, compare_regressions)
    output_paths = _resolve_output_paths(
        output=output,
        output_dir=output_dir,
        regression_types=regression_types,
        compare_regressions=compare_regressions,
    )
    candidates = _build_candidate_weights(x, search_samples, seed) if x.shape[1] > 1 else x

    if x.shape[1] == 1:
        single_method_outputs = {}
        for current_regression_type in regression_types:
            single_method_outputs[current_regression_type] = {
                "proposed": np.array([1.0], dtype=np.float64),
                "predicted": float(y.mean()),
                "method": "single_domain_no_regression",
                "coefficients": [0.0],
                "metadata": None,
            }
    else:
        single_method_outputs = {}
        for current_regression_type in regression_types:
            proposed, predicted, method, coefficients, metadata = _search_best_weights(
                regression_type=current_regression_type,
                x=x,
                y=y,
                candidates=candidates,
            )
            single_method_outputs[current_regression_type] = {
                "proposed": proposed,
                "predicted": predicted,
                "method": method,
                "coefficients": coefficients,
                "metadata": metadata,
            }

    best_observed_idx = int(np.argmax(y))
    best_observed_row = df.iloc[best_observed_idx]
    best_observed_weights = {
        col.replace("weight__", ""): float(best_observed_row[col]) for col in weight_columns
    }

    results: dict[str, dict[str, Any]] = {}
    for current_regression_type, payload in single_method_outputs.items():
        proposed_weights = {
            col.replace("weight__", ""): float(value)
            for col, value in zip(weight_columns, payload["proposed"])
        }
        topic_only = _shrink_actual_prefix(proposed_weights)

        result = {
            "method": payload["method"],
            "regression_type": current_regression_type,
            "objective": objective_name,
            "num_runs": int(len(df)),
            "num_domains": int(len(weight_columns)),
            "num_metrics": int(len(metrics)),
            "metric_columns": metrics,
            "best_observed_run": str(best_observed_row.get("run_name", "")),
            "best_observed_score": float(best_observed_row["mean_score"]),
            "best_observed_weights": best_observed_weights,
            "predicted_best_score": payload["predicted"],
            "weights": proposed_weights,
            "weights_sum": float(sum(proposed_weights.values())),
            "regression_parameters": payload["coefficients"],
        }
        if current_regression_type == "quadratic" and payload["coefficients"] is not None:
            result["regression_coefficients"] = payload["coefficients"]
        if payload["metadata"] is not None:
            result["regression_metadata"] = payload["metadata"]
        if topic_only is not None:
            result["p_star_actual_quality"] = topic_only

        method_path = output_paths.method_outputs[current_regression_type]
        method_path.parent.mkdir(parents=True, exist_ok=True)
        with method_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        results[current_regression_type] = result

    if compare_regressions:
        summary = _build_result_summary(
            objective_name=objective_name,
            metrics=metrics,
            results=results,
            output_paths=output_paths.method_outputs,
        )
        output_paths.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with output_paths.summary_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info("Local fit comparison output written to %s", output_paths.summary_output)
        return output_paths.summary_output

    logger.info("Local fit output written to %s", output_paths.summary_output)
    return output_paths.summary_output
