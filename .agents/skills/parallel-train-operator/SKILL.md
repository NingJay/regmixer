---
name: parallel-train-operator
description: Use this skill when running or monitoring regmixer training and OLMES evaluation through scripts/control_plane.py, especially for cluster launches across hpcgpu09-15 and round1a execution.
---

# Parallel Train Operator

Use this skill for long-running regmixer launch loop operations.

## Core rule

Treat `scripts/control_plane.py` as the launch loop owner.
Treat `scripts/parallel_train.py` and `scripts/parallel_eval.py` as executor-only workers that consume an explicit slot plan.
Do not move SSH probing, host-pool selection, or unified control state back into the executors.

## Repo split

- Use `/home/staff/jiayining/vibe_research/regmixer` as the harness/control plane.
- Use `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` as the runtime tree for real experiments when files or configs differ between trees.
- Treat this split as temporary compatibility, not a required long-term architecture.

## Natural-language trigger

- If the user says `帮我跑这个实验：<config路径>` or similar, interpret that as permission to execute the experiment end-to-end.
- Infer the repo from the config path.
- Default to cluster execution for training workloads.
- Let the launch loop probe first, then start the executor with the discovered slot plan.
- Only ask follow-up questions when the target is genuinely ambiguous or an approval boundary is reached.

## Launch workflow

1. Start from `source ~/.bashrc && conda activate regmixer` unless the shell is already in that environment.
2. Read `scripts/control_plane.py`, the relevant executor, and the active output directory.
3. For round1a, prefer:
   - `ROUND1A_SCHEDULER_MODE=cluster`
   - `ROUND1A_HOSTS=hpcgpu09,...,hpcgpu15`
4. Let the control plane discover idle GPU slots by SSH probe.
5. Confirm the control plane has both SSH connect timeout and probe execution timeout.
6. Confirm the unified `control_plane_state.json` is being updated.
7. Confirm the executor state file is being updated:
   - `parallel_train_state.json` for Step 2
   - `parallel_eval_state.json` for Step 4
8. For Step 4 eval, prefer `scripts/control_plane.py eval` with `cluster` mode, `EVAL_HOSTS=hpcgpu09-15`, and `EVAL_GPU_IDS=all`.
9. Let each phase request up to one worker per variant. If the pool exposes fewer idle GPUs than variants, let the remainder queue in the executor.

## Validation ladder

1. Run narrow checks first.
   - `py_compile` on touched files
   - focused scheduler/control-plane tests
2. Run local control-plane integration tests that launch the real executor path with fake task binaries.
3. Run one real mix on the cluster before scaling out.
4. If the one-mix run reaches real CUDA work and exits cleanly, scale to the wider occupancy run.
5. Do not treat a harness change as validated until the real cluster smoke passes.

## Monitoring workflow

1. Read `.agents/docs/control-plane-launch-loop.md`.
2. Read `.agents/docs/hpcgpu-monitoring-loop.md`.
3. Read `.agents/docs/cluster-runtime-triage.md` when a run fails or appears stuck.
4. Poll `control_plane_state.json` first, then inspect only the executor state and logs for tasks that are running or failed.
5. For Step 4 eval, inspect `parallel_eval_state.json`, `eval/eval_logs/*.log`, and `eval/raw_olmes/*/metrics.json`.
6. Report concise progress summaries using mix indices, host, GPU, phase, and status.

## Failure classification

- `control-plane/probe`: scan stalls, `scan_errors`, no slot plan, no executor launch
- `executor/runtime API drift`: import errors, constructor mismatch, config build failure
- `training logic`: trainer or CUDA work has already started and then fails

Do not redesign the harness until the failure is clearly in the control-plane bucket.

## Escalation boundaries

Contact the user before:

- killing or restarting a remote task
- changing the host pool
- changing remote environment bootstrapping
- altering round1a stage semantics outside control-plane mode selection

## Outputs to trust

- `control_plane_state.json`
- `parallel_train_state.json`
- `parallel_eval_state.json`
- `outputs/.../logs/*.log`
- `outputs/.../summaries/*.json`
