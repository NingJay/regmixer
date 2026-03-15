# Control-Plane Launch Loop

Use this playbook when the agent is asked to launch training or eval and should handle host probing plus executor startup with minimal user interaction.

## Scope

- Owns launch-time behavior only.
- Does not yet own a persistent background monitoring daemon.
- Does not restart cluster resources or mutate host-pool policy without user approval.

## Control-plane responsibilities

1. Normalize the user request into a phase:
   - `train`
   - `eval`
   - `probe`
2. Resolve repo and workdir.
3. Activate the local `regmixer` environment.
4. Probe local GPUs or SSH hosts for idle slots.
5. Write:
   - `control_plane_state.json`
   - slot-plan file for the phase
6. Launch the executor with the explicit slot plan.
7. Report only when a boundary is hit or the phase finishes.

## Executor responsibilities

- `scripts/parallel_train.py`
  - queue mixes onto the provided slots
  - write per-mix logs and summaries
  - write `parallel_train_state.json`
- `scripts/parallel_eval.py`
  - queue eval tasks onto the provided slots
  - write per-model logs and metrics
  - write `parallel_eval_state.json`

Executors should not do their own SSH host discovery.

## State contract

- Unified file: `control_plane_state.json`
  - phase request
  - slot plan summary
  - executor command and return code
- Detailed file:
  - `parallel_train_state.json`
  - `parallel_eval_state.json`

## Failure buckets

- `control-plane/probe`
  - no slots discovered
  - SSH timeouts
  - invalid slot plan
- `executor/runtime API drift`
  - imports fail
  - config or launcher contract changed
- `training logic`
  - model code starts and then fails

## Escalate only for

- destructive recovery actions
- host-pool changes
- missing shared paths or env bootstrapping that require repo-external intervention
