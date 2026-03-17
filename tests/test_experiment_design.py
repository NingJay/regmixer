import json
import warnings
from pathlib import Path

import numpy as np

from regmixer.aliases import ExperimentConfig
from regmixer.experiment_design import (
    DesignArtifacts,
    build_anchor_weights,
    default_design_output_dir,
    default_design_summary_path,
    select_d_opt_design,
    write_design_artifacts,
)
from regmixer.utils import mk_mixes


def _base_config_dict() -> dict:
    return {
        "name": "design-test",
        "description": "design test",
        "budget": "local",
        "workspace": "local",
        "variants": 8,
        "nodes": 1,
        "gpus": 1,
        "max_tokens": 1024,
        "sequence_length": 128,
        "seed": 7,
        "cluster": "local",
        "tokenizer": "openseek",
        "priority": "low",
        "proxy_model_id": "olmo_30m",
        "sources": [
            {
                "name": "actual",
                "topics": [
                    {"name": "high", "paths": ["/tmp/high"]},
                    {"name": "medium", "paths": ["/tmp/medium"]},
                    {"name": "medium_high", "paths": ["/tmp/medium_high"]},
                ],
            }
        ],
    }


def test_experiment_config_maps_legacy_temperature_to_mix_temperature():
    config_dict = _base_config_dict()
    config_dict["temperature"] = 0.75

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = ExperimentConfig(**config_dict)

    assert config.mix_temperature == 0.75
    assert any("temperature is deprecated" in str(w.message) for w in caught)


def test_select_d_opt_design_keeps_anchors_and_full_rank():
    domains = ["actual:high", "actual:medium", "actual:medium_high"]
    candidate_weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1 / 3, 1 / 3, 1 / 3],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.55, 0.15, 0.30],
            [0.4, 0.4, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.2, 0.2, 0.6],
            [0.45, 0.1, 0.45],
        ],
        dtype=np.float64,
    )

    selection = select_d_opt_design(
        candidate_weights,
        domains=domains,
        num_variants=8,
        basis="quadratic_logratio_2d",
        seed=7,
        ridge=1e-6,
        log_floor=1e-4,
        min_distance=0.0,
        anchor_weights=build_anchor_weights(3, ["vertices", "centroid"]),
    )

    assert len(selection.selected_indices) == 8
    assert set(selection.anchor_labels) == {"vertex_0", "vertex_1", "vertex_2", "centroid"}
    assert set(selection.anchor_indices).issubset(set(selection.selected_indices))
    assert selection.effective_rank == 8
    assert selection.condition_number is not None
    assert np.isfinite(selection.logdet)


def test_write_design_artifacts_and_mk_mixes_wiring(tmp_path: Path, monkeypatch):
    candidate_weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1 / 3, 1 / 3, 1 / 3],
            [0.5, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    selected_weights = candidate_weights[:4]
    summary = {
        "candidate_sampling_strategy": "mixed",
        "design_selection_method": "d_opt",
        "candidate_pool_size": 5,
        "selected_indices": [0, 1, 2, 3],
        "anchor_indices": [0, 1, 2, 3],
        "anchor_labels": ["vertex_0", "vertex_1", "vertex_2", "centroid"],
        "design_basis": "quadratic_logratio_2d",
        "expanded_columns": ["bias"],
        "effective_rank": 1,
        "ridge": 1e-6,
        "log_floor": 1e-4,
        "logdet": 1.0,
        "singular_values": [1.0, 0.5],
        "condition_number": 2.0,
        "min_selected_distance": 0.03,
        "seed": 7,
        "domains": ["actual:high", "actual:medium", "actual:medium_high"],
    }
    artifacts = DesignArtifacts(
        summary=summary,
        domains=["actual:high", "actual:medium", "actual:medium_high"],
        candidate_weights=candidate_weights,
        selected_weights=selected_weights,
    )

    summary_path = tmp_path / "round1a_actual_quality_design_summary.json"
    output_dir = tmp_path / "design"
    write_design_artifacts(artifacts, summary_path=summary_path, output_dir=output_dir)

    assert summary_path.exists()
    assert (output_dir / "candidate_simplex.png").exists()
    assert (output_dir / "selected_simplex.png").exists()
    assert (output_dir / "singular_values.png").exists()
    assert (output_dir / "condition_number.png").exists()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: "design-test"',
                'description: "design test"',
                'budget: "local"',
                'workspace: "local"',
                "variants: 4",
                "nodes: 1",
                "gpus: 1",
                "max_tokens: 1024",
                "sequence_length: 128",
                "seed: 7",
                'cluster: "local"',
                'tokenizer: "openseek"',
                'priority: "low"',
                'proxy_model_id: "olmo_30m"',
                "sources:",
                '  - name: "actual"',
                "    topics:",
                '      - name: "high"',
                '        paths: ["/tmp/high"]',
                '      - name: "medium"',
                '        paths: ["/tmp/medium"]',
                '      - name: "medium_high"',
                '        paths: ["/tmp/medium_high"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mixes = [
        {
            "actual:high": (1.0, 1.0),
            "actual:medium": (0.0, 1.0),
            "actual:medium_high": (0.0, 1.0),
        }
    ]

    monkeypatch.setattr(
        "regmixer.utils.mk_mixtures_with_metadata",
        lambda config, group_uuid, use_cache=True: (mixes, artifacts),
    )

    mix_output = tmp_path / "round1a_actual_quality_mixes.json"
    mk_mixes(config_path, "deadbeef", output=mix_output)

    payload = json.loads(mix_output.read_text(encoding="utf-8"))
    assert len(payload["mixes"]) == 1
    assert default_design_summary_path(mix_output).exists()
    assert default_design_output_dir(mix_output).joinpath("candidate_simplex.png").exists()
