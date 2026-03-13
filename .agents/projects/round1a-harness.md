# Round1a Harness

## Goal

Run `scripts/run_round1a.sh` with agent-operated scheduling so the heavy Step 2 training phase can consume idle GPUs across `hpcgpu09-15` without inventing a second scheduler.

## Current shape

- `scripts/parallel_train.py` is the queue scheduler.
- `scripts/run_local_variant.py` is the single GPU execution unit.
- `/home/staff/jiayining/vibe_research/regmixer` is the harness/control-plane tree.
- `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` is the preferred runtime tree for real launches.
- `scripts/run_round1a.sh` now exposes:
  - `ROUND1A_SCHEDULER_MODE=local|cluster`
  - `ROUND1A_GPU_IDS`
  - `ROUND1A_HOSTS`
- Cluster mode writes `parallel_train_state.json` next to the run outputs.

## Operator notes

- Local operator sessions should begin with `source ~/.bashrc && conda activate regmixer`.
- Shared paths are assumed to exist on every host in the pool.
- Remote execution activates conda env `regmixer` after sourcing `~/.bashrc`.
- Scheduler probe failures should be recorded as `scan_errors`; they should not hang the whole run.
- Round1a still runs stages 1 through 5 in order; only Step 2 scheduling has been generalized.

## Learnings from the March 13 update

- The correct validation ladder is: focused local checks, 1-mix real cluster smoke, then 8-GPU real smoke.
- Distinguish harness health from runtime health. A tiny custom CUDA smoke can prove the scheduler is healthy even when regmixer runtime is broken.
- The user-facing API should stay minimal. `帮我跑这个实验：<config路径>` should be enough for the agent to infer repo, scheduler mode, smoke-first execution, and monitoring behavior.
- Current `olmo_core` compatibility required shims for:
  - `NumpyDatasetType` vs `NumpyFSLDatasetConfig`
  - old vs new `SourceMixtureDatasetConfig` constructor shape
  - short-run scheduler decay on 1-step smoke runs
- Real validation succeeded with:
  - 1-mix smoke run `integration-regmixer-smoke-1gpu-20260312-215437`
  - 8-mix / 8-GPU smoke run `integration-regmixer-smoke-8gpu-20260312-215748`

## Follow-ups

- Add a resumable monitor command if agent-side polling from markdown becomes insufficient.
- Decide whether Step 3 to Step 5 should remain synchronous or move into their own long-running control loops.
