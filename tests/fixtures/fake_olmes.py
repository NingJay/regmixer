#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-args", required=True)
    parser.add_argument("--task", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "tasks": [
            {
                "alias": task_name,
                "task_config": {
                    "task_core": task_name.split(":", 1)[0],
                    "metadata": {"description": "Fake metric", "alias": task_name},
                },
                "metrics": {"primary_score": 0.5},
            }
            for task_name in args.task
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "tasks": len(args.task)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
