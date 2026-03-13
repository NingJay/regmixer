# Cluster Runtime Triage

Use this playbook when a real cluster launch fails, hangs, or produces ambiguous evidence.

## Core split

- Harness/control plane: `/home/staff/jiayining/vibe_research/regmixer`
- Runtime tree for real experiments: `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`

If the two trees differ, patch the runtime tree first for anything needed to complete the live run. Mirror the compatible fix back into the harness tree when the behavior should be shared.

## Validation ladder

1. Start from `source ~/.bashrc && conda activate regmixer`.
2. Run only narrow checks for touched files.
   - `python -m py_compile ...`
   - targeted `pytest` for the scheduler
3. Run one real mix on the cluster before scaling out.
   - keep the real runtime workdir on `/home/staff/jiayining/LLM101-dicksuck-r2/regmixer`
   - prefer `--mix-start N --mix-end N --max-workers 1`
4. If one real mix passes and occupies a GPU, scale to 8 GPUs or 8 mixes.
5. Only after both pass should the loop treat the harness change as operationally validated.

## Failure classification

### 1. Scheduler or probe failure

Symptoms:
- `parallel_train.py` stalls before tasks start
- `parallel_train_state.json` is missing or not updating
- logs stop at scan or SSH bootstrap

Actions:
- inspect `scan_errors` first
- treat per-host SSH probe timeouts as degraded capacity, not total failure, if enough slots remain
- confirm remote bootstrap still sources `~/.bashrc` and activates `conda activate regmixer`
- verify the task command uses the intended runtime workdir

### 2. Runtime API drift

Symptoms:
- import errors
- constructor keyword mismatches
- failures during config or dataset build before training starts

Actions:
- read the exact traceback and patch compatibility at the failing boundary
- prefer code-level compatibility shims over redesigning the harness
- current known drift surfaces:
  - `NumpyDatasetType` vs `NumpyFSLDatasetConfig`
  - `SourceMixtureDatasetConfig(source_configs=...)` vs `source_list=...`

### 3. Training logic failure

Symptoms:
- trainer starts, checkpoints or dry-run begin, then training fails
- real CUDA processes exist on the target GPUs

Actions:
- inspect the final traceback in the per-mix log
- treat short-run smokes as a special case; scheduler and LR schedules may need clamps for 1-step runs
- do not blame the harness once the trainer has entered forward/backward or epoch execution

## Isolation rule

If it is unclear whether the issue is the scheduler or regmixer runtime, run a tiny custom CUDA occupancy smoke through `parallel_train.py`. If the custom smoke occupies the expected GPUs, the harness is healthy and the failure is inside regmixer runtime or training logic.

## Evidence to collect

- `parallel_train_state.json`
- one failing per-mix log
- one succeeding summary, if any
- `nvidia-smi --query-compute-apps=...` output from the assigned host
- exact runtime workdir and config path used for the launch

## Escalate to the user only when

- no schedulable slots remain after probe errors
- fixing the issue would require killing jobs, changing SSH config, or changing shared cluster state
- the failure is still ambiguous after the validation ladder above
