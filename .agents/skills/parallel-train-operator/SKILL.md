---
name: parallel-train-operator
description: Use this skill when running or monitoring regmixer training through scripts/parallel_train.py, especially for cluster scheduling across hpcgpu09-15 and round1a Step 2 execution.
---

# Parallel Train Operator

Use this skill for long-running regmixer training launches and monitoring.

## Core rule

Treat `scripts/parallel_train.py` as the single source of truth for training task scheduling. Extend or operate it; do not create a sidecar dispatcher for the same queue.

## Launch workflow

1. Start from `source ~/.bashrc && conda activate regmixer` unless the shell is already in that environment.
2. Read `scripts/parallel_train.py`, `scripts/run_round1a.sh`, and the active output directory.
3. For round1a, prefer:
   - `ROUND1A_SCHEDULER_MODE=cluster`
   - `ROUND1A_HOSTS=hpcgpu09,...,hpcgpu15`
4. Let the scheduler discover idle GPU slots by SSH probe.
5. Confirm the new `parallel_train_state.json` is being updated.

## Monitoring workflow

1. Read `.agents/docs/hpcgpu-monitoring-loop.md`.
2. Poll the state file first, then inspect only the logs for mixes that are running or failed.
3. Report concise progress summaries using mix indices, host, GPU, and status.

## Escalation boundaries

Contact the user before:

- killing or restarting a remote task
- changing the host pool
- changing remote environment bootstrapping
- altering round1a stage semantics outside scheduler mode selection

## Outputs to trust

- `parallel_train_state.json`
- `outputs/.../logs/*.log`
- `outputs/.../summaries/*.json`
