#!/usr/bin/env python
"""Generate standalone visualizations for one round1a fit result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regmixer.round1a_visualization import write_round1a_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone visualizations for one round1a fit result."
    )
    parser.add_argument("--eval-metrics", required=True, help="Path to eval_metrics.csv.")
    parser.add_argument("--fit-json", required=True, help="Path to the selected fit JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <eval-metrics parent>/visualizations.",
    )
    parser.add_argument(
        "--heatmap-top-k",
        type=int,
        default=24,
        help="Number of highest-variance standard metrics to include in the heatmap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_metrics = Path(args.eval_metrics).resolve()
    fit_json = Path(args.fit_json).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (eval_metrics.parent / "visualizations").resolve()
    )

    manifest = write_round1a_visualizations(
        eval_metrics_path=eval_metrics,
        fit_json_path=fit_json,
        output_dir=output_dir,
        heatmap_top_k=args.heatmap_top_k,
    )

    print(f"[round1a_visualize] wrote {manifest['artifacts']['simplex']}")
    print(f"[round1a_visualize] wrote {manifest['artifacts']['weight_comparison']}")
    print(f"[round1a_visualize] wrote {manifest['artifacts']['task_heatmap']}")
    print(
        "[round1a_visualize] wrote "
        f"{output_dir / 'round1a_actual_quality_visualizations.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
