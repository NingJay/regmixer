from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from regmixer.round1a_visualization import barycentric_to_cartesian

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignArtifacts:
    summary: dict[str, object]
    domains: list[str]
    candidate_weights: np.ndarray
    selected_weights: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    anchor_indices: list[int]
    anchor_labels: list[str]
    logdet: float
    singular_values: list[float]
    condition_number: Optional[float]
    expanded_columns: list[str]
    effective_rank: int


def default_design_summary_path(mix_output_path: Path) -> Path:
    stem = mix_output_path.stem
    if stem.endswith("_mixes"):
        stem = stem[: -len("_mixes")]
    return mix_output_path.with_name(f"{stem}_design_summary.json")


def default_design_output_dir(mix_output_path: Path) -> Path:
    return mix_output_path.parent / "design"


def build_design_matrix(
    weights: np.ndarray,
    *,
    domains: Sequence[str],
    basis: str,
    log_floor: float,
) -> tuple[np.ndarray, list[str]]:
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape={weights.shape}")

    if basis != "quadratic_logratio_2d":
        raise ValueError(f"Unsupported design basis: {basis}")
    if weights.shape[1] != 3:
        raise ValueError(
            f"quadratic_logratio_2d requires exactly 3 domains, got {weights.shape[1]} for {domains}"
        )

    high_idx, medium_idx, medium_high_idx = _resolve_round1a_domain_order(domains)
    high = weights[:, high_idx]
    medium = weights[:, medium_idx]
    medium_high = weights[:, medium_high_idx]

    columns = [
        np.ones(len(weights), dtype=np.float64),
        np.log((high + log_floor) / (medium + log_floor)),
        np.log((medium_high + log_floor) / (medium + log_floor)),
        high,
        medium_high,
        high**2,
        medium_high**2,
        high * medium_high,
    ]
    names = [
        "bias",
        "log((high+floor)/(medium+floor))",
        "log((medium_high+floor)/(medium+floor))",
        "high",
        "medium_high",
        "high^2",
        "medium_high^2",
        "high*medium_high",
    ]
    matrix = np.column_stack(columns).astype(np.float64)
    return matrix, names


def select_random_design(
    candidate_weights: np.ndarray,
    *,
    domains: Sequence[str],
    num_variants: int,
    basis: str,
    seed: int,
    ridge: float,
    log_floor: float,
    min_distance: float = 0.0,
    anchor_indices: Optional[Sequence[int]] = None,
    anchor_labels: Optional[Sequence[str]] = None,
) -> SelectionResult:
    if num_variants > len(candidate_weights):
        raise ValueError(
            f"Requested {num_variants} design points from a pool of {len(candidate_weights)} candidates."
        )

    rng = np.random.default_rng(seed)
    selected = list(anchor_indices or [])
    selected_set = set(selected)
    remaining = rng.permutation(len(candidate_weights)).tolist()
    for idx in remaining:
        if idx in selected_set:
            continue
        if _violates_min_distance(candidate_weights, idx, selected, min_distance):
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        if len(selected) == num_variants:
            break

    if len(selected) != num_variants:
        raise ValueError(
            f"Could not select {num_variants} points with min_distance={min_distance} from "
            f"{len(candidate_weights)} candidates."
        )

    design_matrix, expanded_columns = build_design_matrix(
        candidate_weights[selected],
        domains=domains,
        basis=basis,
        log_floor=log_floor,
    )
    singular_values = np.linalg.svd(design_matrix, compute_uv=False).astype(np.float64)
    logdet = _slogdet(design_matrix.T @ design_matrix + ridge * np.eye(design_matrix.shape[1]))
    condition_number = _safe_condition_number(singular_values)
    return SelectionResult(
        selected_indices=selected,
        anchor_indices=list(anchor_indices or []),
        anchor_labels=list(anchor_labels or []),
        logdet=logdet,
        singular_values=singular_values.tolist(),
        condition_number=condition_number,
        expanded_columns=expanded_columns,
        effective_rank=int(np.linalg.matrix_rank(design_matrix)),
    )


def select_d_opt_design(
    candidate_weights: np.ndarray,
    *,
    domains: Sequence[str],
    num_variants: int,
    basis: str,
    seed: int,
    ridge: float,
    log_floor: float,
    min_distance: float = 0.0,
    anchor_weights: Optional[Sequence[tuple[str, np.ndarray]]] = None,
) -> SelectionResult:
    weights = np.asarray(candidate_weights, dtype=np.float64)
    if num_variants > len(weights):
        raise ValueError(
            f"Requested {num_variants} design points from a pool of {len(weights)} candidates."
        )

    design_matrix, expanded_columns = build_design_matrix(
        weights,
        domains=domains,
        basis=basis,
        log_floor=log_floor,
    )
    basis_dim = design_matrix.shape[1]
    if num_variants < basis_dim:
        raise ValueError(
            f"D-opt selection with basis '{basis}' requires variants >= basis_dim ({basis_dim}), got {num_variants}."
        )

    anchor_indices, anchor_labels = _resolve_anchor_indices(weights, anchor_weights)
    selected = list(anchor_indices)
    selected_set = set(anchor_indices)

    rng = np.random.default_rng(seed)
    initial_order = rng.permutation(len(weights)).tolist()
    current_xtx = ridge * np.eye(basis_dim, dtype=np.float64)
    if selected:
        current_xtx += design_matrix[selected].T @ design_matrix[selected]

    while len(selected) < num_variants:
        best_idx = None
        best_objective = -np.inf
        for idx in initial_order:
            if idx in selected_set:
                continue
            if _violates_min_distance(weights, idx, selected, min_distance):
                continue
            candidate_xtx = current_xtx + np.outer(design_matrix[idx], design_matrix[idx])
            objective = _slogdet(candidate_xtx)
            if objective > best_objective:
                best_objective = objective
                best_idx = int(idx)

        if best_idx is None:
            raise ValueError(
                f"Could not grow a D-opt design to {num_variants} points with min_distance={min_distance}."
            )

        selected.append(best_idx)
        selected_set.add(best_idx)
        current_xtx += np.outer(design_matrix[best_idx], design_matrix[best_idx])

    current_logdet = _slogdet(current_xtx)
    anchor_index_set = set(anchor_indices)
    improved = True
    while improved:
        improved = False
        for pos, selected_idx in enumerate(list(selected)):
            if selected_idx in anchor_index_set:
                continue

            base_selection = [idx for idx in selected if idx != selected_idx]
            base_xtx = ridge * np.eye(basis_dim, dtype=np.float64)
            if base_selection:
                base_xtx += design_matrix[base_selection].T @ design_matrix[base_selection]

            best_replacement = selected_idx
            best_replacement_logdet = current_logdet
            replacement_order = rng.permutation(len(weights)).tolist()
            for candidate_idx in replacement_order:
                if candidate_idx in base_selection:
                    continue
                if _violates_min_distance(weights, candidate_idx, base_selection, min_distance):
                    continue
                candidate_xtx = base_xtx + np.outer(design_matrix[candidate_idx], design_matrix[candidate_idx])
                candidate_logdet = _slogdet(candidate_xtx)
                if candidate_logdet > best_replacement_logdet + 1e-9:
                    best_replacement = int(candidate_idx)
                    best_replacement_logdet = candidate_logdet

            if best_replacement != selected_idx:
                selected[pos] = best_replacement
                selected_set = set(selected)
                current_xtx = base_xtx + np.outer(
                    design_matrix[best_replacement], design_matrix[best_replacement]
                )
                current_logdet = best_replacement_logdet
                improved = True
                break

    selected_matrix = design_matrix[selected]
    singular_values = np.linalg.svd(selected_matrix, compute_uv=False).astype(np.float64)
    condition_number = _safe_condition_number(singular_values)
    return SelectionResult(
        selected_indices=selected,
        anchor_indices=anchor_indices,
        anchor_labels=anchor_labels,
        logdet=current_logdet,
        singular_values=singular_values.tolist(),
        condition_number=condition_number,
        expanded_columns=expanded_columns,
        effective_rank=int(np.linalg.matrix_rank(selected_matrix)),
    )


def build_anchor_weights(
    num_domains: int,
    anchor_points: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    anchors: list[tuple[str, np.ndarray]] = []
    for anchor in anchor_points:
        if anchor == "vertices":
            for idx in range(num_domains):
                weights = np.zeros(num_domains, dtype=np.float64)
                weights[idx] = 1.0
                anchors.append((f"vertex_{idx}", weights))
        elif anchor == "centroid":
            anchors.append(("centroid", np.full(num_domains, 1.0 / num_domains, dtype=np.float64)))
        else:
            raise ValueError(f"Unsupported anchor point: {anchor}")
    return anchors


def write_design_artifacts(
    artifacts: DesignArtifacts,
    *,
    summary_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")

    if len(artifacts.domains) == 3:
        _plot_candidate_simplex(
            artifacts.candidate_weights,
            artifacts.selected_weights,
            output_dir / "candidate_simplex.png",
            title="Candidate pool vs selected design",
        )
        _plot_selected_simplex(
            artifacts.selected_weights,
            output_dir / "selected_simplex.png",
            title="Selected design points",
        )

    singular_values = np.asarray(artifacts.summary.get("singular_values", []), dtype=np.float64)
    _plot_singular_values(singular_values, output_dir / "singular_values.png")
    _plot_condition_number(
        float(artifacts.summary.get("condition_number"))
        if artifacts.summary.get("condition_number") is not None
        else None,
        output_dir / "condition_number.png",
    )


def _resolve_round1a_domain_order(domains: Sequence[str]) -> tuple[int, int, int]:
    normalized = [domain.split(":")[-1] for domain in domains]
    if {"high", "medium", "medium_high"}.issubset(set(normalized)):
        return normalized.index("high"), normalized.index("medium"), normalized.index("medium_high")
    return 0, 1, 2


def _resolve_anchor_indices(
    candidate_weights: np.ndarray,
    anchor_weights: Optional[Sequence[tuple[str, np.ndarray]]],
) -> tuple[list[int], list[str]]:
    if not anchor_weights:
        return [], []

    indices: list[int] = []
    labels: list[str] = []
    for label, anchor in anchor_weights:
        matches = np.where(np.linalg.norm(candidate_weights - anchor, axis=1) < 1e-9)[0]
        if len(matches) == 0:
            raise ValueError(f"Anchor '{label}' was not present in the candidate pool.")
        idx = int(matches[0])
        if idx not in indices:
            indices.append(idx)
            labels.append(label)
    return indices, labels


def _violates_min_distance(
    weights: np.ndarray,
    candidate_idx: int,
    selected_indices: Sequence[int],
    min_distance: float,
) -> bool:
    if min_distance <= 0.0 or not selected_indices:
        return False

    candidate = weights[candidate_idx]
    selected = weights[list(selected_indices)]
    distances = np.linalg.norm(selected - candidate, axis=1)
    return bool(np.any(distances < min_distance))


def _slogdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        return float("-inf")
    return float(value)


def _safe_condition_number(singular_values: np.ndarray) -> Optional[float]:
    if singular_values.size == 0:
        return None
    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))
    if min_sv <= 0.0:
        return None
    return max_sv / min_sv


def _plot_candidate_simplex(
    candidate_weights: np.ndarray,
    selected_weights: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    cx, cy = barycentric_to_cartesian(
        candidate_weights[:, 0], candidate_weights[:, 1], candidate_weights[:, 2]
    )
    sx, sy = barycentric_to_cartesian(
        selected_weights[:, 0], selected_weights[:, 1], selected_weights[:, 2]
    )

    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color="#2f2a24", linewidth=2)
    ax.scatter(cx, cy, s=26, color="#b8c1c8", alpha=0.65, edgecolors="none", label="Candidates")
    ax.scatter(sx, sy, s=90, color="#d97706", edgecolors="white", linewidths=0.8, label="Selected")
    ax.set_title(title)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2.0 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.05))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_selected_simplex(selected_weights: np.ndarray, output_path: Path, *, title: str) -> None:
    sx, sy = barycentric_to_cartesian(
        selected_weights[:, 0], selected_weights[:, 1], selected_weights[:, 2]
    )
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color="#2f2a24", linewidth=2)
    ax.scatter(sx, sy, s=110, color="#0a9396", edgecolors="white", linewidths=0.8)
    for idx, (x, y) in enumerate(zip(sx, sy)):
        ax.text(float(x) + 0.012, float(y) + 0.01, f"{idx:02d}", fontsize=8, color="#2f2a24")
    ax.set_title(title)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2.0 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_singular_values(singular_values: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    if singular_values.size == 0:
        ax.text(0.5, 0.5, "No singular values", ha="center", va="center")
    else:
        ax.bar(np.arange(len(singular_values)), singular_values, color="#0a9396")
        ax.set_xticks(np.arange(len(singular_values)))
        ax.set_xlabel("Basis component")
        ax.set_ylabel("Singular value")
    ax.set_title("Selected design matrix singular values")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_condition_number(condition_number: Optional[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    value = float(condition_number) if condition_number is not None else math.nan
    if np.isnan(value):
        ax.text(0.5, 0.5, "Condition number undefined", ha="center", va="center")
    else:
        ax.bar([0], [value], color="#d97706", width=0.5)
        ax.set_xticks([0], ["selected"])
        ax.set_ylabel("Condition number")
        ax.set_yscale("log")
    ax.set_title("Selected design matrix condition number")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
