# Regmixer Agent Operations Guide

## Quick reference

- Default local shell bootstrap: `source ~/.bashrc && conda activate regmixer`
- Training queue core: `scripts/parallel_train.py`
- Single-variant executor: `scripts/run_local_variant.py`
- Round1a entrypoint: `scripts/run_round1a.sh`
- Shared host pool default: `hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15`
- Stateful scheduler output: `parallel_train_state.json`

## Hard rules

- Activate the `regmixer` conda environment before running repo commands locally unless the current shell is already inside an equivalent environment.
- Reuse `scripts/parallel_train.py` for round1a scheduling. Do not introduce a second dispatcher unless the user explicitly asks for a replacement.
- Treat a GPU as schedulable only when `nvidia-smi --query-compute-apps` shows no running compute process on that GPU.
- Assume cluster paths are shared across `hpcgpu09-15`; verify before changing that assumption.
- Do not kill remote training jobs, rewrite the host pool, or mutate shared cluster state without explicit consent in the current thread.
- Read the scheduler state file and the per-mix logs before changing scheduling or reporting a failure.

## Default workflow

1. Start local work from `source ~/.bashrc && conda activate regmixer`.
2. For round1a launches, prefer `ROUND1A_SCHEDULER_MODE=cluster` unless the user asks to stay local.
3. Let `parallel_train.py` discover idle `(host, gpu)` worker slots.
4. Monitor `parallel_train_state.json`, `outputs/.../logs/*.log`, and `outputs/.../summaries/*.json`.
5. Only contact the user when a task hits an authorization boundary or a non-trivial failure.

## Ask first

- Killing or restarting remote jobs.
- Changing `run_round1a.sh` stage semantics beyond scheduler wiring.
- Modifying host discovery rules or expanding beyond `hpcgpu09-15`.
- Adding dependencies or background daemons for monitoring.

## Repo control plane

- Monitoring playbook: `.agents/docs/hpcgpu-monitoring-loop.md`
- Active project memory: `.agents/projects/round1a-harness.md`
- Reusable operator skill: `.agents/skills/parallel-train-operator/SKILL.md`
