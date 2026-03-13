# Round1a Harness

## Goal

Run `scripts/run_round1a.sh` with agent-operated scheduling so Step 2 training and Step 4 evaluation can both consume idle GPUs across `hpcgpu09-15` without inventing ad-hoc dispatchers.

## Current shape

- `scripts/parallel_train.py` is the queue scheduler.
- `scripts/parallel_eval.py` is the OLMES eval queue and now supports local or cluster scheduling.
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
- Round1a still runs stages 1 through 5 in order; Step 2 training and Step 4 evaluation now both have queue-based scheduling.
- Step 4 defaults to `cluster` mode, targets one worker per variant, and only falls back to queueing when the pool exposes fewer than 15 idle GPUs.

## Learnings from the March 13 update

- The correct validation ladder is: focused local checks, 1-mix real cluster smoke, then 8-GPU real smoke.
- Eval throughput matters too, but local-only 8-GPU queues are not enough for round1a. Step 4 has to scan the shared host pool and start all 15 variants immediately when at least 15 idle GPUs exist.
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
