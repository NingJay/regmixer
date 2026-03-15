#!/usr/bin/env python
"""Control-plane launcher for regmixer train and eval executors."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from regmixer.controlplane import (
    DEFAULT_CLUSTER_HOSTS,
    DEFAULT_REMOTE_CONDA_ENV,
    DEFAULT_REMOTE_SHELL_INIT,
    DEFAULT_SSH_PROBE_TIMEOUT,
    build_worker_slots,
    ensure_phase_state,
    parse_gpu_ids,
    parse_hosts,
    read_control_state,
    resolve_path,
    resolve_worker_limit,
    validate_cluster_mode_paths,
    write_control_state,
    write_slot_plan_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe slots, write unified control state, then launch executors.")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    train = subparsers.add_parser("train", help="Launch scripts/parallel_train.py through the control plane.")
    add_common_control_args(train)
    train.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    train.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    train.add_argument("--group-id", required=True, help="Group id passed to run_local_variant.py.")
    train.add_argument("--run-name-prefix", default="parquet-e2e-fit-train")
    train.add_argument("--mix-start", type=int, default=0)
    train.add_argument("--mix-end", type=int, default=11)
    train.add_argument("--python-bin", default=None)
    train.add_argument("--variant-script", default="scripts/run_local_variant.py")
    train.add_argument("--pythonpath", default="src")
    train.add_argument("--log-dir", default="outputs/dryfit_fit/logs")
    train.add_argument("--summary-dir", default="outputs/dryfit_fit/summaries")
    train.add_argument("--executor-state-file", default=None)
    train.add_argument("--output-root-dir", default=None)
    train.add_argument("--beaker-user", default=None)
    train.add_argument("--global-batch-size", type=int, default=None)
    train.add_argument(
        "--passthrough-env",
        default="WANDB_MODE,REGMIXER_DISABLE_SWANLAB_SYNC",
        help="Comma-separated env vars copied into remote task commands when present locally.",
    )

    eval_parser = subparsers.add_parser("eval", help="Launch scripts/parallel_eval.py through the control plane.")
    add_common_control_args(eval_parser)
    eval_parser.add_argument("--hf-manifest", required=True, help="Path to eval/hf_models_manifest.csv.")
    eval_parser.add_argument("--raw-results-dir", required=True, help="Directory for eval/raw_olmes outputs.")
    eval_parser.add_argument("--log-dir", required=True, help="Directory for per-model eval logs.")
    eval_parser.add_argument("--executor-state-file", default=None)
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.add_argument("--force-eval", type=int, default=0)
    eval_parser.add_argument("--mix-start", type=int, default=0)
    eval_parser.add_argument("--mix-end", type=int, default=14)
    eval_parser.add_argument("--olmes-bin", default="olmes")
    eval_parser.add_argument(
        "--passthrough-env",
        default="HF_HOME,HF_HUB_CACHE,HF_TOKEN,TRANSFORMERS_CACHE,XDG_CACHE_HOME",
        help="Comma-separated env vars copied into remote task commands when present locally.",
    )
    eval_parser.add_argument("--max-retries", type=int, default=1)

    probe = subparsers.add_parser("probe", help="Only scan the slot pool and write the slot plan/control state.")
    add_common_control_args(probe)
    probe.add_argument(
        "--requested-workers",
        type=int,
        required=True,
        help="Desired worker count for the scan-only plan.",
    )
    return parser


def add_common_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scheduler-mode", choices=("local", "cluster"), required=True)
    parser.add_argument("--gpu-ids", default="all")
    parser.add_argument("--hosts", default=DEFAULT_CLUSTER_HOSTS)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--remote-workdir", default=None)
    parser.add_argument("--remote-conda-env", default=DEFAULT_REMOTE_CONDA_ENV)
    parser.add_argument("--remote-shell-init", default=DEFAULT_REMOTE_SHELL_INIT)
    parser.add_argument("--ssh-connect-timeout", type=int, default=5)
    parser.add_argument("--ssh-probe-timeout", type=int, default=DEFAULT_SSH_PROBE_TIMEOUT)
    parser.add_argument("--control-state-file", default=None)
    parser.add_argument("--slots-file", default=None)


def build_mix_indices(start: int, end: int) -> List[int]:
    if start < 0:
        raise ValueError(f"mix-start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"mix-end({end}) must be >= mix-start({start})")
    return list(range(start, end + 1))


def default_control_state_file_for_phase(phase: str, workdir: str, args: argparse.Namespace) -> str:
    if args.control_state_file:
        return resolve_path(args.control_state_file, workdir)
    if phase == "train":
        return str(Path(resolve_path(args.summary_dir, workdir)).parent / "control_plane_state.json")
    if phase == "eval":
        return str(Path(resolve_path(args.log_dir, workdir)).parent / "control_plane_state.json")
    return str(Path(workdir) / "control_plane_state.json")


def default_slots_file_for_phase(phase: str, workdir: str, args: argparse.Namespace) -> str:
    if args.slots_file:
        return resolve_path(args.slots_file, workdir)
    if phase == "train":
        return str(Path(resolve_path(args.summary_dir, workdir)).parent / "train_slot_plan.json")
    if phase == "eval":
        return str(Path(resolve_path(args.log_dir, workdir)).parent / "eval_slot_plan.json")
    return str(Path(workdir) / "slot_plan.json")


def default_executor_state_file_for_phase(phase: str, workdir: str, args: argparse.Namespace) -> str:
    if args.executor_state_file:
        return resolve_path(args.executor_state_file, workdir)
    if phase == "train":
        return str(Path(resolve_path(args.summary_dir, workdir)).parent / "parallel_train_state.json")
    if phase == "eval":
        return str(Path(resolve_path(args.log_dir, workdir)).parent / "parallel_eval_state.json")
    raise ValueError(f"unsupported phase: {phase}")


def update_phase_state(
    state_path: str,
    workdir: str,
    phase: str,
    *,
    status: str,
    request: Optional[Dict[str, object]] = None,
    slot_plan: Optional[Dict[str, object]] = None,
    executor: Optional[Dict[str, object]] = None,
) -> None:
    state = read_control_state(state_path, workdir)
    phase_state = ensure_phase_state(state, phase)
    phase_state["phase"] = phase
    phase_state["status"] = status
    if request is not None:
        phase_state["request"] = {**phase_state.get("request", {}), **request}
    if slot_plan is not None:
        phase_state["slot_plan"] = {**phase_state.get("slot_plan", {}), **slot_plan}
    if executor is not None:
        phase_state["executor"] = {**phase_state.get("executor", {}), **executor}
    phase_state["updated_at"] = time.time()
    write_control_state(state_path, state)


def prepare_slot_plan(args: argparse.Namespace, requested_workers: int) -> Dict[str, object]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    hosts = parse_hosts(args.hosts)
    slots, scan_errors = build_worker_slots(
        scheduler_mode=args.scheduler_mode,
        hosts=hosts,
        gpu_ids=gpu_ids,
        connect_timeout=args.ssh_connect_timeout,
        probe_timeout=args.ssh_probe_timeout,
        max_workers=requested_workers,
    )
    return {
        "requested_workers": requested_workers,
        "allocated_workers": len(slots),
        "hosts": hosts,
        "gpu_ids": args.gpu_ids,
        "scan_errors": scan_errors,
        "slots": slots,
    }


def build_pythonpath_env(workdir: str) -> Dict[str, str]:
    env = os.environ.copy()
    repo_src = str((Path(workdir) / "src").resolve())
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_src if not current else f"{repo_src}{os.pathsep}{current}"
    return env


def build_train_executor_command(workdir: str, args: argparse.Namespace, slots_file: str, executor_state_file: str) -> List[str]:
    cmd = [
        sys.executable,
        str((Path(workdir) / "scripts" / "parallel_train.py").resolve()),
        "--scheduler-mode",
        args.scheduler_mode,
        "--slots-file",
        slots_file,
        "--config",
        resolve_path(args.config, workdir),
        "--mix-file",
        resolve_path(args.mix_file, workdir),
        "--group-id",
        args.group_id,
        "--run-name-prefix",
        args.run_name_prefix,
        "--mix-start",
        str(args.mix_start),
        "--mix-end",
        str(args.mix_end),
        "--workdir",
        workdir,
        "--variant-script",
        resolve_path(args.variant_script, workdir),
        "--pythonpath",
        args.pythonpath,
        "--log-dir",
        resolve_path(args.log_dir, workdir),
        "--summary-dir",
        resolve_path(args.summary_dir, workdir),
        "--state-file",
        executor_state_file,
        "--remote-conda-env",
        args.remote_conda_env,
        "--remote-shell-init",
        args.remote_shell_init,
        "--ssh-connect-timeout",
        str(args.ssh_connect_timeout),
        "--passthrough-env",
        args.passthrough_env,
    ]
    if args.python_bin:
        cmd.extend(["--python-bin", args.python_bin])
    if args.remote_workdir:
        cmd.extend(["--remote-workdir", resolve_path(args.remote_workdir, workdir)])
    if args.output_root_dir:
        cmd.extend(["--output-root-dir", resolve_path(args.output_root_dir, workdir)])
    if args.beaker_user:
        cmd.extend(["--beaker-user", args.beaker_user])
    if args.global_batch_size is not None:
        cmd.extend(["--global-batch-size", str(args.global_batch_size)])
    return cmd


def build_eval_executor_command(workdir: str, args: argparse.Namespace, slots_file: str, executor_state_file: str) -> List[str]:
    cmd = [
        sys.executable,
        str((Path(workdir) / "scripts" / "parallel_eval.py").resolve()),
        "--scheduler-mode",
        args.scheduler_mode,
        "--slots-file",
        slots_file,
        "--hf-manifest",
        resolve_path(args.hf_manifest, workdir),
        "--raw-results-dir",
        resolve_path(args.raw_results_dir, workdir),
        "--log-dir",
        resolve_path(args.log_dir, workdir),
        "--state-file",
        executor_state_file,
        "--batch-size",
        str(args.batch_size),
        "--force-eval",
        str(args.force_eval),
        "--mix-start",
        str(args.mix_start),
        "--mix-end",
        str(args.mix_end),
        "--workdir",
        workdir,
        "--olmes-bin",
        args.olmes_bin,
        "--remote-conda-env",
        args.remote_conda_env,
        "--remote-shell-init",
        args.remote_shell_init,
        "--ssh-connect-timeout",
        str(args.ssh_connect_timeout),
        "--passthrough-env",
        args.passthrough_env,
        "--max-retries",
        str(args.max_retries),
    ]
    if args.remote_workdir:
        cmd.extend(["--remote-workdir", resolve_path(args.remote_workdir, workdir)])
    return cmd


def launch_executor(phase: str, cmd: Sequence[str], workdir: str, control_state_file: str, executor_state_file: str, slots_file: str) -> int:
    update_phase_state(
        control_state_file,
        workdir,
        phase,
        status="launching",
        executor={
            "command": list(cmd),
            "state_file": executor_state_file,
            "slot_plan_file": slots_file,
            "started_at": time.time(),
        },
    )
    env = build_pythonpath_env(workdir)
    proc = subprocess.run(list(cmd), cwd=workdir, env=env, check=False)
    update_phase_state(
        control_state_file,
        workdir,
        phase,
        status="completed" if proc.returncode == 0 else "failed",
        executor={
            "command": list(cmd),
            "state_file": executor_state_file,
            "slot_plan_file": slots_file,
            "return_code": int(proc.returncode),
            "completed_at": time.time(),
        },
    )
    return int(proc.returncode)


def handle_probe(args: argparse.Namespace) -> int:
    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)
    control_state_file = default_control_state_file_for_phase("probe", workdir, args)
    slots_file = default_slots_file_for_phase("probe", workdir, args)
    slot_plan = prepare_slot_plan(args, args.requested_workers)
    write_slot_plan_file(
        slots_file,
        args.scheduler_mode,
        args.requested_workers,
        slot_plan["slots"],
        slot_plan["scan_errors"],
    )
    update_phase_state(
        control_state_file,
        workdir,
        "probe",
        status="completed",
        request={
            "scheduler_mode": args.scheduler_mode,
            "workdir": workdir,
            "requested_workers": args.requested_workers,
            "remote_workdir": resolve_path(args.remote_workdir, workdir) if args.remote_workdir else workdir,
        },
        slot_plan={
            "requested_workers": args.requested_workers,
            "allocated_workers": slot_plan["allocated_workers"],
            "hosts": slot_plan["hosts"],
            "gpu_ids": slot_plan["gpu_ids"],
            "scan_errors": slot_plan["scan_errors"],
            "slots_file": slots_file,
            "slots": [slot.label for slot in slot_plan["slots"]],
        },
    )
    print(
        f"[control_plane] phase=probe mode={args.scheduler_mode} "
        f"requested_workers={args.requested_workers} allocated_workers={slot_plan['allocated_workers']}"
    )
    return 0


def handle_train(args: argparse.Namespace) -> int:
    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)
    mix_indices = build_mix_indices(args.mix_start, args.mix_end)
    requested_workers = resolve_worker_limit(args.max_workers, len(mix_indices))
    control_state_file = default_control_state_file_for_phase("train", workdir, args)
    slots_file = default_slots_file_for_phase("train", workdir, args)
    executor_state_file = default_executor_state_file_for_phase("train", workdir, args)
    request = {
        "scheduler_mode": args.scheduler_mode,
        "workdir": workdir,
        "remote_workdir": resolve_path(args.remote_workdir, workdir) if args.remote_workdir else workdir,
        "mix_indices": mix_indices,
        "requested_workers": requested_workers,
        "config": resolve_path(args.config, workdir),
        "mix_file": resolve_path(args.mix_file, workdir),
        "group_id": args.group_id,
    }
    update_phase_state(control_state_file, workdir, "train", status="probing", request=request)
    slot_plan = prepare_slot_plan(args, requested_workers)
    write_slot_plan_file(
        slots_file,
        args.scheduler_mode,
        requested_workers,
        slot_plan["slots"],
        slot_plan["scan_errors"],
    )
    update_phase_state(
        control_state_file,
        workdir,
        "train",
        status="running",
        slot_plan={
            "requested_workers": requested_workers,
            "allocated_workers": slot_plan["allocated_workers"],
            "hosts": slot_plan["hosts"],
            "gpu_ids": slot_plan["gpu_ids"],
            "scan_errors": slot_plan["scan_errors"],
            "slots_file": slots_file,
            "slots": [slot.label for slot in slot_plan["slots"]],
        },
    )
    print(
        f"[control_plane] phase=train mode={args.scheduler_mode} "
        f"requested_workers={requested_workers} allocated_workers={slot_plan['allocated_workers']} "
        f"mixes={len(mix_indices)}"
    )
    cmd = build_train_executor_command(workdir, args, slots_file, executor_state_file)
    return launch_executor("train", cmd, workdir, control_state_file, executor_state_file, slots_file)


def handle_eval(args: argparse.Namespace) -> int:
    workdir = str(Path(args.workdir).resolve())
    if args.scheduler_mode == "cluster":
        validate_cluster_mode_paths(workdir, args.remote_workdir)
    task_count = len(build_mix_indices(args.mix_start, args.mix_end))
    requested_workers = resolve_worker_limit(args.max_workers, task_count)
    control_state_file = default_control_state_file_for_phase("eval", workdir, args)
    slots_file = default_slots_file_for_phase("eval", workdir, args)
    executor_state_file = default_executor_state_file_for_phase("eval", workdir, args)
    request = {
        "scheduler_mode": args.scheduler_mode,
        "workdir": workdir,
        "remote_workdir": resolve_path(args.remote_workdir, workdir) if args.remote_workdir else workdir,
        "requested_workers": requested_workers,
        "hf_manifest": resolve_path(args.hf_manifest, workdir),
        "raw_results_dir": resolve_path(args.raw_results_dir, workdir),
        "mix_start": args.mix_start,
        "mix_end": args.mix_end,
    }
    update_phase_state(control_state_file, workdir, "eval", status="probing", request=request)
    slot_plan = prepare_slot_plan(args, requested_workers)
    write_slot_plan_file(
        slots_file,
        args.scheduler_mode,
        requested_workers,
        slot_plan["slots"],
        slot_plan["scan_errors"],
    )
    update_phase_state(
        control_state_file,
        workdir,
        "eval",
        status="running",
        slot_plan={
            "requested_workers": requested_workers,
            "allocated_workers": slot_plan["allocated_workers"],
            "hosts": slot_plan["hosts"],
            "gpu_ids": slot_plan["gpu_ids"],
            "scan_errors": slot_plan["scan_errors"],
            "slots_file": slots_file,
            "slots": [slot.label for slot in slot_plan["slots"]],
        },
    )
    print(
        f"[control_plane] phase=eval mode={args.scheduler_mode} "
        f"requested_workers={requested_workers} allocated_workers={slot_plan['allocated_workers']} "
        f"mixes={task_count}"
    )
    cmd = build_eval_executor_command(workdir, args, slots_file, executor_state_file)
    return launch_executor("eval", cmd, workdir, control_state_file, executor_state_file, slots_file)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase == "probe":
        return handle_probe(args)
    if args.phase == "train":
        return handle_train(args)
    if args.phase == "eval":
        return handle_eval(args)
    raise ValueError(f"unsupported phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
