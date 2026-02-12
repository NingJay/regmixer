#!/usr/bin/env python
"""Parallel local variant scheduler (8 GPU fixed-worker queue model).

Architecture
------------
1) Main process builds a task queue of mix indices (default 0..11).
2) One worker process is created per GPU id (default 0..7). Each worker is
   permanently pinned to a single GPU and consumes tasks from the shared queue.
3) Each consumed task launches `scripts/run_local_variant.py` via subprocess.
4) Task failure is isolated: non-zero exit code is recorded, but worker keeps
   consuming next tasks until queue is exhausted.
5) Final summary prints failed task list and exits non-zero only if failures
   exist.

Per-task environment injection
------------------------------
- CUDA_VISIBLE_DEVICES = <gpu_id>
- MASTER_PORT = 29541 + <gpu_id>
- TORCH_DISTRIBUTED_DEFAULT_PORT = 29542 + <gpu_id>

I/O isolation
-------------
- log:     <log_dir>/<run_name_prefix>-<mix_index:04d>.log
- summary: <summary_dir>/<run_name_prefix>-<mix_index:04d>.json

Example (run remotely through SSH)
----------------------------------
ssh hpcgpu15 "source ~/.bashrc && conda activate regmixer && \
  cd /home/staff/jiayining/LLM101-dicksuck-r2/regmixer && \
  PYTHONPATH=src python scripts/parallel_train.py \
    --config src/regmixer/config/nemotron-cc-round1-parquet-e2e-fit-smoke.yaml \
    --mix-file outputs/nemo_round1_parquet_e2e_fit_smoke_mixes.json \
    --group-id dryfit0001 \
    --run-name-prefix parquet-e2e-fit-train \
    --log-dir outputs/dryfit_fit/logs \
    --summary-dir outputs/dryfit_fit/summaries"
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WorkerConfig:
    workdir: str
    python_bin: str
    variant_script: str
    config_path: str
    mix_file: str
    group_id: str
    run_name_prefix: str
    log_dir: str
    summary_dir: str
    pythonpath: str
    beaker_user: Optional[str] = None
    global_batch_size: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regmixer mixes with fixed-GPU worker queue scheduling."
    )
    parser.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    parser.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    parser.add_argument("--group-id", required=True, help="Group id passed to run_local_variant.py.")
    parser.add_argument(
        "--run-name-prefix",
        default="parquet-e2e-fit-train",
        help="Run name prefix. Full name is '<prefix>-<mix_index:04d>'.",
    )
    parser.add_argument(
        "--mix-start",
        type=int,
        default=0,
        help="First mix index (inclusive).",
    )
    parser.add_argument(
        "--mix-end",
        type=int,
        default=11,
        help="Last mix index (inclusive).",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids. One worker process per GPU id.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory for subprocess execution.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run scripts/run_local_variant.py.",
    )
    parser.add_argument(
        "--variant-script",
        default="scripts/run_local_variant.py",
        help="Path to run_local_variant.py relative to workdir or absolute.",
    )
    parser.add_argument(
        "--pythonpath",
        default="src",
        help="PYTHONPATH injected to each task subprocess.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/dryfit_fit/logs",
        help="Directory for per-task logs.",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/dryfit_fit/summaries",
        help="Directory for per-task summaries.",
    )
    parser.add_argument(
        "--beaker-user",
        default=None,
        help="Optional pass-through to run_local_variant.py --beaker-user.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Optional pass-through to run_local_variant.py --global-batch-size.",
    )
    return parser.parse_args()


def parse_gpu_ids(raw: str) -> List[int]:
    gpu_ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    if not gpu_ids:
        raise ValueError("gpu id list is empty")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"gpu id list contains duplicates: {gpu_ids}")
    return gpu_ids


def build_mix_indices(start: int, end: int) -> List[int]:
    if start < 0:
        raise ValueError(f"mix-start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"mix-end({end}) must be >= mix-start({start})")
    return list(range(start, end + 1))


def build_command(cfg: WorkerConfig, mix_index: int, run_name: str, summary_out: str) -> List[str]:
    cmd: List[str] = [
        cfg.python_bin,
        cfg.variant_script,
        "--config",
        cfg.config_path,
        "--mix-file",
        cfg.mix_file,
        "--mix-index",
        str(mix_index),
        "--run-name",
        run_name,
        "--group-id",
        cfg.group_id,
        "--summary-out",
        summary_out,
    ]
    if cfg.beaker_user:
        cmd.extend(["--beaker-user", cfg.beaker_user])
    if cfg.global_batch_size is not None:
        cmd.extend(["--global-batch-size", str(cfg.global_batch_size)])
    return cmd


def run_single_task(mix_index: int, gpu_id: int, cfg: WorkerConfig) -> Dict[str, Any]:
    idx = f"{mix_index:04d}"
    run_name = f"{cfg.run_name_prefix}-{idx}"
    log_path = Path(cfg.log_dir) / f"{run_name}.log"
    summary_path = Path(cfg.summary_dir) / f"{run_name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = cfg.pythonpath
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MASTER_PORT"] = str(29541 + gpu_id)
    env["TORCH_DISTRIBUTED_DEFAULT_PORT"] = str(29542 + gpu_id)

    cmd = build_command(cfg, mix_index, run_name, str(summary_path))
    start_ts = time.time()
    return_code = 99
    error_message = ""

    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[scheduler] mix_index={mix_index} gpu_id={gpu_id}\n")
            logf.write(f"[scheduler] command={shlex.join(cmd)}\n")
            logf.write(
                "[scheduler] env CUDA_VISIBLE_DEVICES="
                f"{env['CUDA_VISIBLE_DEVICES']} MASTER_PORT={env['MASTER_PORT']} "
                f"TORCH_DISTRIBUTED_DEFAULT_PORT={env['TORCH_DISTRIBUTED_DEFAULT_PORT']}\n"
            )
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=cfg.workdir,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return_code = int(proc.returncode)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        error_message = f"{type(exc).__name__}: {exc}"
        try:
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"[scheduler] exception={error_message}\n")
        except OSError:
            pass

    duration_sec = round(time.time() - start_ts, 2)
    return {
        "mix_index": mix_index,
        "gpu_id": gpu_id,
        "run_name": run_name,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "return_code": return_code,
        "duration_sec": duration_sec,
        "error": error_message,
    }


def worker_loop(gpu_id: int, task_queue: "mp.Queue[Optional[int]]", result_queue: "mp.Queue[Dict[str, Any]]", cfg: WorkerConfig) -> None:
    while True:
        mix_index = task_queue.get()
        if mix_index is None:
            return
        result_queue.put(run_single_task(mix_index, gpu_id, cfg))


def main() -> int:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    mix_indices = build_mix_indices(args.mix_start, args.mix_end)

    cfg = WorkerConfig(
        workdir=str(Path(args.workdir)),
        python_bin=args.python_bin,
        variant_script=args.variant_script,
        config_path=args.config,
        mix_file=args.mix_file,
        group_id=args.group_id,
        run_name_prefix=args.run_name_prefix,
        log_dir=args.log_dir,
        summary_dir=args.summary_dir,
        pythonpath=args.pythonpath,
        beaker_user=args.beaker_user,
        global_batch_size=args.global_batch_size,
    )

    task_queue: "mp.Queue[Optional[int]]" = mp.Queue()
    result_queue: "mp.Queue[Dict[str, Any]]" = mp.Queue()

    for mix_index in mix_indices:
        task_queue.put(mix_index)
    for _ in gpu_ids:
        task_queue.put(None)

    workers: List[mp.Process] = []
    for gpu_id in gpu_ids:
        proc = mp.Process(target=worker_loop, args=(gpu_id, task_queue, result_queue, cfg), daemon=False)
        proc.start()
        workers.append(proc)

    results: List[Dict[str, Any]] = []
    for _ in mix_indices:
        result = result_queue.get()
        results.append(result)
        ok = result["return_code"] == 0
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] mix={result['mix_index']:04d} gpu={result['gpu_id']} "
            f"rc={result['return_code']} log={result['log_path']}"
        )

    for proc in workers:
        proc.join()

    results.sort(key=lambda x: int(x["mix_index"]))
    failed = [r for r in results if int(r["return_code"]) != 0]
    report = {
        "total": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "failed_mix_indices": [int(r["mix_index"]) for r in failed],
        "failures": failed,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
