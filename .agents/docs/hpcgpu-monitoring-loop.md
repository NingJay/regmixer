# Hpcgpu Monitoring Loop

Use this playbook when `scripts/control_plane.py` has launched training or eval in `cluster` mode and the user wants the agent to keep watch instead of checking in manually.

## Inputs

- Unified control state, usually `outputs/<run>/control_plane_state.json`
- Executor state file:
  - `outputs/<run>/parallel_train_state.json` for Step 2 training
  - `outputs/<run>/parallel_eval_state.json` for Step 4 eval
- Round1a pipeline log, if launched via `scripts/run_round1a.sh`
- Per-task logs in `outputs/<run>/logs/`
- Per-task summaries in `outputs/<run>/summaries/`
- Eval logs in `outputs/<run>/eval/eval_logs/`
- Runtime workdir used for the launch, usually `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` for real experiments

## Launch checklist

1. Confirm the local shell is in `conda activate regmixer` before running repo commands or diagnostics.
2. Confirm `scripts/control_plane.py` is the process that performed probing and launched the executor.
3. Read `control_plane_state.json` before doing anything else.
4. If the run is new, record:
   - phase
   - scheduler mode
   - host pool
   - requested workers
   - allocated workers
   - runtime workdir
   - slot-plan path
   - executor state path
5. If the run is resumed, report the current running and failed tasks from the executor state file.

## Monitoring cadence

- First confirmation after launch: within 120 seconds
- Steady-state cadence: every 570 seconds
- Faster cadence is only justified when a task has just failed or a run is entering a new stage

## What to inspect each cycle

1. `control_plane_state.json`
   - `phases.<phase>.status`
   - `phases.<phase>.slot_plan.scan_errors`
   - `phases.<phase>.slot_plan.allocated_workers`
   - `phases.<phase>.executor.return_code`
2. Executor state
   - `tasks[*].status`
   - `tasks[*].return_code`
   - `updated_at`
3. Logs
   - Python tracebacks
   - SSH/auth failures
   - missing conda env or missing workdir
4. Summaries or metrics
   - newly completed variants
   - `global_step`, `save_folder`, or `metrics.json`

## Interpretation rules

- Treat host probe timeouts in `scan_errors` as degraded capacity unless they leave the phase with no usable slots.
- If the executor never launches, the failure is still in the control-plane/probe bucket.
- If a task log reaches model build, dataset materialization, checkpoint save, or forward/backward dry-run, the harness is no longer the primary suspect.
- Use `.agents/docs/cluster-runtime-triage.md` when the failure class is unclear.

## Contact the user only when

- No schedulable GPU slots are available and progress cannot start.
- SSH to one or more hosts is failing and the remaining pool is insufficient.
- A task failed with a non-obvious error.
- A destructive action would be required, such as killing a remote process or changing cluster configuration.

## Do not do without approval

- Kill or restart remote jobs
- Modify SSH config or host lists
- Change cluster drivers, CUDA stacks, or conda environments
- Delete checkpoints or logs
