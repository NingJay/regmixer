# Regmixer Agent Operations Guide

## Quick reference

- Default local shell bootstrap: `source ~/.bashrc && conda activate regmixer`
- Harness/control-plane repo: `/home/staff/jiayining/vibe_research/regmixer`
- Preferred runtime repo for real experiments: `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`
- Control-plane entrypoint: `scripts/control_plane.py`
- Train executor: `scripts/parallel_train.py`
- Eval executor: `scripts/parallel_eval.py`
- Single-variant executor: `scripts/run_local_variant.py`
- Round1a entrypoint: `scripts/run_round1a.sh`
- Round1a fit visualizer: `scripts/visualize_round1a_results.py`
- Round1a fit compare visualizer: `scripts/visualize_round1a_fit_comparison.py`
- Shared host pool default: `hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15`
- Unified control state: `control_plane_state.json`
- Executor state files: `parallel_train_state.json`, `parallel_eval_state.json`

## Hard rules

- Activate the `regmixer` conda environment before running repo commands locally unless the current shell is already inside an equivalent environment.
- Treat `scripts/control_plane.py` as the launch loop owner for training and eval. Do not put SSH probing, slot discovery, or host-pool policy back into `parallel_train.py` or `parallel_eval.py`.
- Treat `generate-mixes` as the owner of round1a experiment design. Candidate sampling and final design selection belong there, not in `control_plane.py`, `parallel_*`, or `fit-mixture`.
- Treat `scripts/parallel_train.py` and `scripts/parallel_eval.py` as executor-only queue workers that consume an explicit slot plan.
- Treat `/home/staff/jiayining/vibe_research/regmixer` as the harness/control plane and `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` as the runtime tree when real experiment files or configs only exist there.
- The split between control plane and runtime tree is a temporary compatibility measure, not the desired steady state. Prefer converging back to one repo when the harness and runtime paths can safely be unified.
- Treat a GPU as schedulable only when `nvidia-smi --query-compute-apps` shows no running compute process on that GPU.
- Cluster probing must use both SSH connect timeout and probe execution timeout. A host-level probe timeout becomes `scan_errors`, not a global launch hang.
- Assume cluster paths are shared across `hpcgpu09-15`; verify before changing that assumption.
- Do not kill remote training jobs, rewrite the host pool, or mutate shared cluster state without explicit consent in the current thread.
- Read `control_plane_state.json`, the relevant executor state file, and the per-task logs before changing scheduling or reporting a failure.
- Prefer targeted verification over full-suite testing. Default ladder: edited-file syntax check or focused pytest, then local control-plane integration tests, then 1-mix real cluster smoke, then wider cluster occupancy only after the 1-mix run is healthy.
- Classify failures before redesigning the harness: `control-plane/probe`, `executor/runtime API drift`, or `training logic`.
- For round1a fitting, treat `p_star_actual_quality.json` as the canonical downstream artifact. The default canonical regressor is `log_linear`. Treat `fit_compare/` as diagnostics only.
- For round1a design artifacts, keep `*_mixes.json` and `*_design_summary.json` separate. Never write control metadata into the mix schema.

## Default workflow

1. Start local work from `source ~/.bashrc && conda activate regmixer`.
2. Decide whether the task is harness-only or needs the runtime tree; default real launches to `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`.
3. For round1a launches, prefer `ROUND1A_SCHEDULER_MODE=cluster` unless the user asks to stay local.
4. Let `scripts/control_plane.py` probe the slot pool, write `control_plane_state.json`, emit a slot plan, and then launch the relevant executor.
5. Let Step 2 training flow through `scripts/control_plane.py train`, which hands the explicit slot plan to `scripts/parallel_train.py`.
6. Let Step 4 OLMES flow through `scripts/control_plane.py eval`, which hands the explicit slot plan to `scripts/parallel_eval.py`.
7. Let each phase target one worker per variant when enough idle GPUs exist; if the pool exposes fewer slots, allow the remainder to queue inside the executor.
8. Use `.agents/docs/cluster-runtime-triage.md` when a real run fails or appears to hang.
9. Monitor `control_plane_state.json`, `parallel_train_state.json`, `parallel_eval_state.json`, `outputs/.../logs/*.log`, `outputs/.../summaries/*.json`, and `eval/eval_logs/*.log`.
10. Only contact the user when a task hits an authorization boundary or a non-trivial failure.
11. When round1a fit diagnostics are requested, keep `ROUND1A_FIT_REGRESSION_TYPE` explicit and enable `ROUND1A_COMPARE_REGRESSIONS=1`.
12. When round1a uses `mixed + d_opt`, inspect `*_design_summary.json` and `design/` before blaming the fitter.

## Natural-Language Launch Contract

- If the user says `帮我跑这个实验：<config路径>` or equivalent, treat it as an execution request, not a planning request.
- Infer the execution repo from the config path first.
  - If the config path points into `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`, run there.
  - Otherwise start from the current repo and only switch repos if runtime-only files are missing.
- For training configs, default to `cluster` mode on `hpcgpu09-15` unless the user explicitly asks for local execution.
- For round1a Step 4 eval, default to `cluster` mode, `EVAL_HOSTS=hpcgpu09-15`, and `EVAL_GPU_IDS=all`.
- Default validation ladder:
  1. narrow local checks
  2. local control-plane integration path
  3. one real mix smoke on cluster
  4. full run only after the smoke is healthy
- Do not require the user to provide scheduler mode, host pool, GPU ids, or monitoring instructions unless the task is genuinely ambiguous.

## Ask first

- Killing or restarting remote jobs.
- Changing `run_round1a.sh` stage semantics beyond control-plane wiring.
- Modifying host discovery rules or expanding beyond `hpcgpu09-15`.
- Adding dependencies or background daemons for monitoring.

## Repo control plane

- Launch loop playbook: `.agents/docs/control-plane-launch-loop.md`
- Monitoring playbook: `.agents/docs/hpcgpu-monitoring-loop.md`
- Failure triage playbook: `.agents/docs/cluster-runtime-triage.md`
- Active project memory: `.agents/projects/round1a-harness.md`
- Reusable operator skill: `.agents/skills/parallel-train-operator/SKILL.md`
