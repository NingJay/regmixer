# Round1a Harness

## Goal

Run `scripts/run_round1a.sh` with an agent-operated launch loop so Step 2 training and Step 4 evaluation can both consume idle GPUs across `hpcgpu09-15`, while keeping SSH probing and slot policy outside the executors.

## Current shape

- `scripts/control_plane.py` owns launch-loop behavior:
  - probe host pool
  - detect idle GPU slots
  - write unified `control_plane_state.json`
  - emit per-phase slot plans
  - launch the relevant executor
- `scripts/parallel_train.py` is the train executor queue.
- `scripts/parallel_eval.py` is the OLMES eval executor queue.
- `scripts/run_local_variant.py` is the single-GPU execution unit.
- `/home/staff/jiayining/vibe_research/regmixer` is the harness/control-plane tree.
- `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` is the preferred runtime tree for real launches.
- `scripts/run_round1a.sh` now routes Step 2 and Step 4 through the control plane.

## State model

- Unified control state: `control_plane_state.json`
- Train executor state: `parallel_train_state.json`
- Eval executor state: `parallel_eval_state.json`
- Train slot plan: `train_slot_plan.json`
- Eval slot plan: `eval_slot_plan.json`

The control-plane file is the repo-level launch memory. Executor files remain the detailed task ledger.

## Operator notes

- Local operator sessions should begin with `source ~/.bashrc && conda activate regmixer`.
- Shared paths are assumed to exist on every host in the pool.
- Remote execution activates conda env `regmixer` after sourcing `~/.bashrc`.
- Scheduler probe failures should be recorded as `scan_errors`; they should not hang the whole phase.
- Round1a still runs stages 1 through 6 in order; Step 2 training and Step 4 evaluation now both have a control-plane launch step followed by executor queueing.
- Step 4 defaults to `cluster` mode, targets one worker per variant, and only falls back to queueing when the pool exposes fewer than 15 idle GPUs.

## Learnings from the March 15 control-plane split

- The earlier abstraction was wrong because SSH probing, slot discovery, and executor queueing were mixed together in `parallel_train.py` and `parallel_eval.py`.
- The cleaner boundary is:
  - control plane owns cluster-facing decisions and unified state
  - executors own task fan-out across already assigned slots
- The correct validation ladder is now:
  1. focused local checks
  2. local control-plane integration tests
  3. 1-mix real cluster smoke
  4. wider cluster occupancy only after the 1-mix run is healthy
- Agent markdown now acts as control-plane surface area, not just notes. It documents the launch loop, monitoring cadence, escalation boundaries, and repo memory.
- User-facing API should stay minimal. `帮我跑这个实验：<config路径>` should be enough for the agent to infer repo, scheduler mode, smoke-first execution, and monitoring behavior.

## Follow-ups

- Add a persistent monitor command if markdown-driven polling becomes insufficient.
- Decide whether Step 3 to Step 6 should remain synchronous or move into their own long-running control loops.
- Consider moving repo selection and config-path normalization into a reusable Python contract module if agent surface area keeps growing.
