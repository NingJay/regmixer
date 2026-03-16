#!/usr/bin/env python
"""Generate comparison visualizations for multiple round1a fit results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regmixer.round1a_visualization import write_fit_comparison_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison visualizations for multiple round1a fit results."
    )
    parser.add_argument("--eval-metrics", required=True, help="Path to eval_metrics.csv.")
    parser.add_argument(
        "--fit-comparison-json",
        required=True,
        help="Path to the fit comparison summary JSON produced by fit-mixture.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <fit-comparison-json parent>/visualizations.",
    )
    parser.add_argument(
        "--heatmap-top-k",
        type=int,
        default=24,
        help="Number of highest-variance standard metrics to include in each heatmap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_metrics = Path(args.eval_metrics).resolve()
    fit_comparison_json = Path(args.fit_comparison_json).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (fit_comparison_json.parent / "visualizations").resolve()
    )

    manifest = write_fit_comparison_visualizations(
        eval_metrics_path=eval_metrics,
        fit_comparison_json_path=fit_comparison_json,
        output_dir=output_dir,
        heatmap_top_k=args.heatmap_top_k,
    )

    print(f"[round1a_fit_compare_visualize] wrote {manifest['artifacts']['comparison_plot']}")
    for method_name, method_info in manifest["methods"].items():
        print(
            "[round1a_fit_compare_visualize] wrote "
            f"{method_info['visualization_manifest']} ({method_name})"
        )
    print(f"[round1a_fit_compare_visualize] wrote {output_dir / 'fit_comparison_visualizations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
