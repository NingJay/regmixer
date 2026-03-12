# Round1a Harness

## Goal

Run `scripts/run_round1a.sh` with agent-operated scheduling so the heavy Step 2 training phase can consume idle GPUs across `hpcgpu09-15` without inventing a second scheduler.

## Current shape

- `scripts/parallel_train.py` is the queue scheduler.
- `scripts/run_local_variant.py` is the single GPU execution unit.
- `scripts/run_round1a.sh` now exposes:
  - `ROUND1A_SCHEDULER_MODE=local|cluster`
  - `ROUND1A_GPU_IDS`
  - `ROUND1A_HOSTS`
- Cluster mode writes `parallel_train_state.json` next to the run outputs.

## Operator notes

- Local operator sessions should begin with `source ~/.bashrc && conda activate regmixer`.
- Shared paths are assumed to exist on every host in the pool.
- Remote execution activates conda env `regmixer` after sourcing `~/.bashrc`.
- Round1a still runs stages 1 through 5 in order; only Step 2 scheduling has been generalized.

## Follow-ups

- Add a resumable monitor command if agent-side polling from markdown becomes insufficient.
- Decide whether Step 3 to Step 5 should remain synchronous or move into their own long-running control loops.
