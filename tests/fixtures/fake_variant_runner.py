#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mix-file", required=True)
    parser.add_argument("--mix-index", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--output-root-dir", default=None)
    parser.add_argument("--beaker-user", default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "mix_file": args.mix_file,
        "mix_index": args.mix_index,
        "run_name": args.run_name,
        "group_id": args.group_id,
        "output_root_dir": args.output_root_dir,
        "beaker_user": args.beaker_user,
        "global_batch_size": args.global_batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "global_step": 1,
        "save_folder": str(summary_path.parent / f"{args.run_name}-checkpoint"),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
