---
name: parallel-train-operator
description: Use this skill when running or monitoring regmixer training and OLMES evaluation through scripts/parallel_train.py and scripts/parallel_eval.py, especially for cluster scheduling across hpcgpu09-15 and round1a execution.
---

# Parallel Train Operator

Use this skill for long-running regmixer training launches and monitoring.

## Core rule

Treat `scripts/parallel_train.py` as the single source of truth for training task scheduling. Extend or operate it; do not create a sidecar dispatcher for the same queue.
Treat `scripts/parallel_eval.py` as the reusable queue worker for round1a OLMES evaluation, with `cluster` mode as the default execution path.

## Repo split

- Use `/home/staff/jiayining/vibe_research/regmixer` as the harness/control plane.
- Use `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` as the runtime tree for real experiments when files or configs differ between trees.
- Treat this split as temporary compatibility, not a required long-term architecture.

## Natural-language trigger

- If the user says `帮我跑这个实验：<config路径>` or similar, interpret that as permission to execute the experiment end-to-end.
- Infer the repo from the config path.
- Default to cluster execution for training workloads.
- Do the 1-mix smoke first, then scale out if healthy.
- Only ask follow-up questions when the target is genuinely ambiguous or an approval boundary is reached.

## Launch workflow

1. Start from `source ~/.bashrc && conda activate regmixer` unless the shell is already in that environment.
2. Read `scripts/parallel_train.py`, `scripts/run_round1a.sh`, and the active output directory.
3. For round1a, prefer:
   - `ROUND1A_SCHEDULER_MODE=cluster`
   - `ROUND1A_HOSTS=hpcgpu09,...,hpcgpu15`
4. Let the scheduler discover idle GPU slots by SSH probe.
5. Confirm the scheduler has a probe execution timeout in addition to SSH connect timeout.
6. Confirm the new `parallel_train_state.json` is being updated.
7. For round1a Step 4, prefer `scripts/parallel_eval.py` with `EVAL_SCHEDULER_MODE=cluster`, `EVAL_HOSTS=hpcgpu09-15`, and `EVAL_GPU_IDS=all`.
8. Let Step 4 request up to one worker per variant. If the pool exposes fewer idle GPUs than variants, let the remainder queue.

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
4. For Step 4 eval, inspect `parallel_eval_state.json`, `eval/eval_logs/*.log`, and `eval/raw_olmes/*/metrics.json`.
5. Report concise progress summaries using mix indices, host, GPU, and status.

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
