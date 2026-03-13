# Hpcgpu Monitoring Loop

Use this playbook when `parallel_train.py` is running in `cluster` mode and the user wants the agent to keep watch instead of checking in manually.

## Inputs

- Scheduler state file, usually `outputs/<run>/parallel_train_state.json`
- Round1a pipeline log, if launched via `scripts/run_round1a.sh`
- Per-mix logs in `outputs/<run>/logs/`
- Per-mix summaries in `outputs/<run>/summaries/`
- Runtime workdir used for the launch, usually `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` for real experiments

## Launch checklist

1. Confirm the local shell is in `conda activate regmixer` before running repo commands or diagnostics.
2. Confirm `scripts/parallel_train.py` is the process driving the run.
3. Read the current state file before doing anything else.
4. If the run is new, record:
   - scheduler mode
   - host pool
   - allowed GPU ids
   - runtime workdir
   - output root
5. If the run is resumed, report the current running and failed mix indices.

## Monitoring cadence

- First confirmation after launch: within 120 seconds
- Steady-state cadence: every 570 seconds
- Faster cadence is only justified when a task has just failed or a run is entering a new stage

## What to inspect each cycle

1. `parallel_train_state.json`
   - `scan_errors`
   - `tasks[*].status`
   - `tasks[*].return_code`
   - `updated_at`
2. Per-mix logs
   - Python tracebacks
   - SSH/auth failures
   - missing conda env or missing workdir
3. Per-mix summaries
   - newly completed variants
   - global step and save folder

## Interpretation rules

- Treat host probe timeouts in `scan_errors` as degraded capacity unless they leave the run with no usable slots.
- If a mix log reaches model build, dataset materialization, checkpoint save, or forward/backward dry-run, the harness is no longer the primary suspect.
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
