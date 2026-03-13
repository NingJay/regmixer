---
name: parallel-train-operator
description: Use this skill when running or monitoring regmixer training through scripts/parallel_train.py, especially for cluster scheduling across hpcgpu09-15 and round1a Step 2 execution.
---

# Parallel Train Operator

Use this skill for long-running regmixer training launches and monitoring.

## Core rule

Treat `scripts/parallel_train.py` as the single source of truth for training task scheduling. Extend or operate it; do not create a sidecar dispatcher for the same queue.

## Repo split

- Use `/home/staff/jiayining/vibe_research/regmixer` as the harness/control plane.
- Use `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` as the runtime tree for real experiments when files or configs differ between trees.

## Launch workflow

1. Start from `source ~/.bashrc && conda activate regmixer` unless the shell is already in that environment.
2. Read `scripts/parallel_train.py`, `scripts/run_round1a.sh`, and the active output directory.
3. For round1a, prefer:
   - `ROUND1A_SCHEDULER_MODE=cluster`
   - `ROUND1A_HOSTS=hpcgpu09,...,hpcgpu15`
4. Let the scheduler discover idle GPU slots by SSH probe.
5. Confirm the scheduler has a probe execution timeout in addition to SSH connect timeout.
6. Confirm the new `parallel_train_state.json` is being updated.

## Validation ladder

1. Run narrow checks first.
   - `py_compile` on touched files
   - targeted scheduler tests
2. Run one real mix on the cluster before scaling out.
3. If the one-mix run reaches real CUDA work and exits cleanly, scale to the 8-GPU or 8-mix run.
4. Do not treat a harness change as validated until the real cluster smoke passes.

## Monitoring workflow

1. Read `.agents/docs/hpcgpu-monitoring-loop.md`.
2. Read `.agents/docs/cluster-runtime-triage.md` when a run fails or appears stuck.
3. Poll the state file first, then inspect only the logs for mixes that are running or failed.
4. Report concise progress summaries using mix indices, host, GPU, and status.

## Failure classification

- `scheduler/probe`: scan stalls, `scan_errors`, no task progress
- `runtime API drift`: import errors, constructor mismatch, config build failure
- `training logic`: trainer or CUDA work has already started and then fails

Do not redesign the scheduler until the failure is clearly in the scheduler bucket.

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
