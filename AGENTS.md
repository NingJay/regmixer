# Regmixer Agent Operations Guide

## Quick reference

- Default local shell bootstrap: `source ~/.bashrc && conda activate regmixer`
- Harness/control-plane repo: `/home/staff/jiayining/vibe_research/regmixer`
- Preferred runtime repo for real experiments: `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`
- Training queue core: `scripts/parallel_train.py`
- Single-variant executor: `scripts/run_local_variant.py`
- Round1a entrypoint: `scripts/run_round1a.sh`
- Shared host pool default: `hpcgpu09,hpcgpu10,hpcgpu11,hpcgpu12,hpcgpu13,hpcgpu14,hpcgpu15`
- Stateful scheduler output: `parallel_train_state.json`

## Hard rules

- Activate the `regmixer` conda environment before running repo commands locally unless the current shell is already inside an equivalent environment.
- Reuse `scripts/parallel_train.py` for round1a scheduling. Do not introduce a second dispatcher unless the user explicitly asks for a replacement.
- Treat `/home/staff/jiayining/vibe_research/regmixer` as the harness/control plane and `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer` as the runtime tree when real experiment files or configs only exist there.
- The split between control plane and runtime tree is a temporary compatibility measure, not the desired steady state. Prefer converging back to one repo when the harness and runtime paths can safely be unified.
- Treat a GPU as schedulable only when `nvidia-smi --query-compute-apps` shows no running compute process on that GPU.
- Cluster probing must use both SSH connect timeout and probe execution timeout. A host-level probe timeout should become `scan_errors`, not a global scheduler hang.
- Assume cluster paths are shared across `hpcgpu09-15`; verify before changing that assumption.
- Do not kill remote training jobs, rewrite the host pool, or mutate shared cluster state without explicit consent in the current thread.
- Read the scheduler state file and the per-mix logs before changing scheduling or reporting a failure.
- Prefer targeted verification over full-suite testing. Default ladder: edited-file syntax check or focused pytest, then 1-mix real cluster smoke, then 8-GPU occupancy or 8-mix smoke only after the 1-mix run is healthy.
- Classify failures before redesigning the harness: `scheduler/probe`, `runtime API drift`, or `training logic`.

## Default workflow

1. Start local work from `source ~/.bashrc && conda activate regmixer`.
2. Decide whether the task is harness-only or needs the runtime tree; default real launches to `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`.
3. For round1a launches, prefer `ROUND1A_SCHEDULER_MODE=cluster` unless the user asks to stay local.
4. Let `parallel_train.py` discover idle `(host, gpu)` worker slots and inspect `scan_errors` before calling the run blocked.
5. Use `.agents/docs/cluster-runtime-triage.md` when a real run fails or appears to hang.
6. Monitor `parallel_train_state.json`, `outputs/.../logs/*.log`, and `outputs/.../summaries/*.json`.
7. Only contact the user when a task hits an authorization boundary or a non-trivial failure.

## Natural-Language Launch Contract

- If the user says `帮我跑这个实验：<config路径>` or equivalent, treat it as an execution request, not a planning request.
- Infer the execution repo from the config path first.
  - If the config path points into `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`, run there.
  - Otherwise start from the current repo and only switch repos if runtime-only files are missing.
- For training configs, default to `cluster` mode on `hpcgpu09-15` unless the user explicitly asks for local execution.
- Default validation ladder:
  1. narrow local checks
  2. one real mix smoke on cluster
  3. full run only after the smoke is healthy
- Do not require the user to provide scheduler mode, host pool, GPU ids, or monitoring instructions unless the task is genuinely ambiguous.

## Ask first

- Killing or restarting remote jobs.
- Changing `run_round1a.sh` stage semantics beyond scheduler wiring.
- Modifying host discovery rules or expanding beyond `hpcgpu09-15`.
- Adding dependencies or background daemons for monitoring.

## Repo control plane

- Monitoring playbook: `.agents/docs/hpcgpu-monitoring-loop.md`
- Failure triage playbook: `.agents/docs/cluster-runtime-triage.md`
- Active project memory: `.agents/projects/round1a-harness.md`
- Reusable operator skill: `.agents/skills/parallel-train-operator/SKILL.md`
