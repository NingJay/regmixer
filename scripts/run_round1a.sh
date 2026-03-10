#!/usr/bin/env bash
# ┌──────────────────────────────────────────────────────────────────────────┐
# │                         Round-1a 实验执行总览                           │
# └──────────────────────────────────────────────────────────────────────────┘
#
#   这份脚本是 round1a 的唯一入口，负责把 “生成配比 -> 训练 -> 转 HF ->
#   OLMES 评测 -> 聚合 CSV -> 拟合最优配比” 串成一条完整流水线。
#
#                               ┌────────────────────┐
#                               │ 1. 读取实验配置    │
#                               │ CONFIG_FILE        │
#                               │ 输出根目录等参数   │
#                               └─────────┬──────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. 生成 mixtures                                                        │
# │   调用: python -m regmixer.cli generate-mixes                           │
# │   输入: nemotron-cc-round1a-actual-quality.yaml                         │
# │   输出: round1a_actual_quality_mixes.json                               │
# │   作用: 生成 15 个候选 actual 质量桶配比                                │
# └──────────────────────────────────────────────────────────────────────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 并行训练 15 个 variants                                               │
# │   调用: scripts/parallel_train.py                                        │
# │   范围: mix 0..14                                                        │
# │   资源: 8 张 GPU 固定 worker 队列                                        │
# │   输出:                                                                  │
# │     logs/        每个 run 的训练日志                                     │
# │     summaries/   每个 run 的 summary.json                                │
# │     train_artifacts/ 本地 checkpoint                                     │
# └──────────────────────────────────────────────────────────────────────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. checkpoint 转 HuggingFace 格式                                        │
# │   调用: python -m regmixer.cli convert-hf                                │
# │   依据: summaries/*.json 中的 save_folder / global_step                  │
# │   输出:                                                                  │
# │     eval/hf_models/                每个 run 的 HF checkpoint             │
# │     eval/hf_models_manifest.csv    后续评测使用的索引清单                │
# └──────────────────────────────────────────────────────────────────────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. 本地运行 OLMES 评测                                                   │
# │   评测方式: 只保留 OLMES 一条路径                                        │
# │   任务集合: core + mmlu，同时跑 RC + BPB                                │
# │   输入: eval/hf_models_manifest.csv                                      │
# │   输出: eval/raw_olmes/<model_name>/                                     │
# │     包含 metrics.json、task-*-metrics.json、launcher.log 等              │
# │   特性: 若 metrics.json 已存在，则直接复用，不重复评测                   │
# └──────────────────────────────────────────────────────────────────────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 6. 聚合 OLMES 原始结果到 eval_metrics.csv                                │
# │   数据来源:                                                              │
# │     round1a_actual_quality_mixes.json  -> weight__* 列                  │
# │     raw_olmes/*/metrics.json         -> task__* 列                      │
# │   输出: eval_metrics.csv                                                  │
# │   作用: 形成 fit-mixture 的直接输入表                                    │
# └──────────────────────────────────────────────────────────────────────────┘
#                                         │
#                                         ▼
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 7. 拟合最优 actual 质量桶配比                                             │
# │   调用: python -m regmixer.cli fit-mixture                               │
# │   输入: eval_metrics.csv                                                 │
# │   输出: p_star_actual_quality.json                                       │
# │   结果: 给出 p*_actual_quality，供下一轮实验或正式训练使用               │
# └──────────────────────────────────────────────────────────────────────────┘
#
#   恢复机制:
#   - ROUND1A_START_STEP=1  从头开始
#   - ROUND1A_START_STEP=3  从 convert-hf 开始恢复
#   - ROUND1A_START_STEP=4  只重跑 OLMES + CSV 聚合
#   - ROUND1A_START_STEP=5  只重跑 fit-mixture
#
# ============================================================================
# Round-1a: actual 内部质量桶比例搜索
# ============================================================================
# 目标: 找到 actual 内部最优质量分布（high/medium_high/medium）
# Variants: 15 (3 质量桶 × 5)
# Training: 15 variants × 3B tokens = 45B tokens total
# Usage:
#   bash scripts/run_round1a.sh [OUTPUT_DIR]
#   ROUND1A_OUTPUT_DIR=/path/to/output bash scripts/run_round1a.sh
# Optional:
#   ROUND1A_TRAIN_ROOT_DIR=/path/to/train-artifacts
#   ROUND1A_START_STEP=3  # 从 convert-hf 开始恢复
# ============================================================================

set -e
set -o pipefail

# 解析脚本所在目录。这里的 PROJECT_ROOT 实际上就是 regmixer 根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ensure_python_env() {
  # 已经在正确环境里时直接返回。
  if PYTHONPATH=src python -c "import olmo_core" >/dev/null 2>&1; then
    return
  fi

  echo "[INFO] 'olmo_core' not found in current Python env. Trying conda env 'regmixer'..."
  if [ -f "${HOME}/.bashrc" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
    set -u
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not available and current env is missing required deps."
    echo "        Please install dependencies or run: source ~/.bashrc && conda activate regmixer"
    exit 1
  fi

  set +u
  conda activate regmixer || {
    set -u
    echo "[ERROR] Failed to activate conda env 'regmixer'."
    exit 1
  }
  set -u

  if ! PYTHONPATH=src python -c "import olmo_core" >/dev/null 2>&1; then
    echo "[ERROR] Active Python env is still missing 'olmo_core'."
    exit 1
  fi
}

ensure_python_env

require_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] Required file not found: ${path}"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "[ERROR] Required directory not found: ${path}"
    exit 1
  fi
}

run_single_olmes_eval() {
  local hf_model_dir="$1"
  local output_dir="$2"
  local model_basename
  local gpu_id
  local batch_size
  local force_eval
  local log_file
  local tasks

  if [[ "${hf_model_dir}" != /* ]]; then
    hf_model_dir="${PROJECT_ROOT}/${hf_model_dir}"
  fi
  if [[ "${output_dir}" != /* ]]; then
    output_dir="${PROJECT_ROOT}/${output_dir}"
  fi

  if [ ! -d "${hf_model_dir}" ]; then
    echo "[ERROR] HF model dir not found: ${hf_model_dir}"
    exit 1
  fi

  if ! command -v olmes >/dev/null 2>&1; then
    echo "[ERROR] 'olmes' command not found in the active environment."
    exit 1
  fi

  model_basename="$(basename "${hf_model_dir}")"
  gpu_id="${OLMES_GPU_ID:-${ROUND1A_OLMES_GPU_ID:-1}}"
  batch_size="${OLMES_BATCH_SIZE:-1}"
  force_eval="${OLMES_FORCE_EVAL:-${ROUND1A_FORCE_EVAL:-0}}"
  log_file="${output_dir}/olmes_eval.log"
  mkdir -p "${output_dir}"

  # 已有结果时默认复用，避免重复长时间评测。
  if [ -f "${output_dir}/metrics.json" ] && [ "${force_eval}" != "1" ]; then
    echo "[INFO] Reusing existing OLMES results: ${output_dir}/metrics.json"
    return 0
  fi

  # 统一只保留一条评测路径：OLMES，同时跑 core+mmlu 以及 RC+BPB。
  tasks=(
    "arc:rc::olmes:full"
    "arc:rc:bpb::olmes:full"
    "mmlu:rc::olmes"
    "mmlu:rc:bpb::olmes"
    "boolq:rc::olmes:full"
    "boolq:rc:bpb::olmes:full"
    "csqa:rc::olmes:full"
    "csqa:rc:bpb::olmes:full"
    "hellaswag:rc::olmes:full"
    "hellaswag:rc:bpb::olmes:full"
    "openbookqa:rc::olmes:full"
    "openbookqa:rc:bpb::olmes:full"
    "piqa:rc::olmes:full"
    "piqa:rc:bpb::olmes:full"
    "socialiqa:rc::olmes:full"
    "socialiqa:rc:bpb::olmes:full"
    "winogrande:rc::olmes:full"
    "winogrande:rc:bpb::olmes:full"
  )

  echo "============================================================================"
  echo "Round-1a 本地 OLMES 评测"
  echo "============================================================================"
  echo "Model dir:  ${hf_model_dir}"
  echo "Output dir: ${output_dir}"
  echo "GPU:        ${gpu_id}"
  echo "Batch size: ${batch_size}"
  echo "Log file:   ${log_file}"
  echo "Tasks:      ${#tasks[@]}"
  echo "============================================================================"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  HF_DATASETS_TRUST_REMOTE_CODE=1 \
  DATASETS_TRUST_REMOTE_CODE=1 \
  olmes \
    --model "${model_basename}" \
    --model-type hf \
    --model-args "model_path=${hf_model_dir},trust_remote_code=True,add_bos_token=True,max_length=8192" \
    --task "${tasks[@]}" \
    --batch-size "${batch_size}" \
    --gpus 1 \
    --output-dir "${output_dir}" \
    2>&1 | tee -a "${log_file}"
}

aggregate_olmes_to_csv() {
  local output_dir="$1"
  local mix_start="$2"
  local mix_end="$3"
  local mix_file="${output_dir}/round1a_actual_quality_mixes.json"
  local eval_dir="${output_dir}/eval"
  local hf_manifest="${eval_dir}/hf_models_manifest.csv"
  local raw_results_dir="${eval_dir}/raw_olmes"
  local eval_csv="${output_dir}/eval_metrics.csv"
  local first_metrics_file=""
  local model_basename=""
  local metrics_file=""
  local run_name=""
  local mix_index=""
  local global_step=""
  local checkpoint_step_dir=""
  local hf_model_dir=""
  local weight_key=""
  local mapping=""
  local canonical_name=""
  local task_core=""
  local score=""
  local task_mean=""
  local weight_value=""
  local -a weight_keys=()
  local -a task_mappings=()
  local -a header=()
  local -a row=()
  local -a task_values=()

  if ! command -v jq >/dev/null 2>&1; then
    echo "[ERROR] jq is required to aggregate raw OLMES metrics into CSV."
    exit 1
  fi

  require_file "${mix_file}"
  require_file "${hf_manifest}"
  require_dir "${raw_results_dir}"

  # 先从 mix 文件中抽出所有 weight 列，后面会拼成 CSV header。
  mapfile -t weight_keys < <(
    jq -r '.mixes | map(keys) | add | unique[]' "${mix_file}" | LC_ALL=C sort
  )

  # 用第一份可用的 metrics.json 推断任务列名。
  while IFS=, read -r run_name mix_index global_step checkpoint_step_dir hf_model_dir; do
    [ -z "${run_name}" ] && continue
    hf_model_dir="${hf_model_dir%$'\r'}"
    if (( mix_index < mix_start || mix_index > mix_end )); then
      continue
    fi
    model_basename="$(basename "${hf_model_dir}")"
    metrics_file="${raw_results_dir}/${model_basename}/metrics.json"
    if [ -f "${metrics_file}" ]; then
      first_metrics_file="${metrics_file}"
      break
    fi
  done < <(tail -n +2 "${hf_manifest}")

  if [ -z "${first_metrics_file}" ]; then
    echo "[ERROR] Could not find any OLMES metrics.json under ${raw_results_dir} for mix range ${mix_start}-${mix_end}."
    exit 1
  fi

  # 只保留 RC 主指标，不把 BPB 再展开成另一套 task__ 列。
  mapfile -t task_mappings < <(
    jq -r '
      .tasks[]
      | select((.task_config.metadata.description // "") != "Aggregate metric")
      | select(((.alias // .task_config.metadata.alias // "") | contains(":bpb::")) | not)
      | .task_config.task_core
    ' "${first_metrics_file}" \
    | awk '!seen[$0]++' \
    | while read -r task_core; do
        canonical_name="${task_core}"
        case "${task_core}" in
          csqa)
            canonical_name="commonsense_qa"
            ;;
          socialiqa)
            canonical_name="social_iqa"
            ;;
        esac
        printf '%s\t%s\n' "${canonical_name}" "${task_core}"
      done \
    | LC_ALL=C sort
  )

  if [ "${#task_mappings[@]}" -eq 0 ]; then
    echo "[ERROR] No per-task RC OLMES metrics were found in ${first_metrics_file}."
    exit 1
  fi

  mkdir -p "$(dirname "${eval_csv}")"

  header=(
    "run_name"
    "mix_index"
    "global_step"
    "checkpoint_step_dir"
    "hf_model_dir"
    "raw_metrics_file"
  )

  for weight_key in "${weight_keys[@]}"; do
    header+=("weight__${weight_key}")
  done

  for mapping in "${task_mappings[@]}"; do
    IFS=$'\t' read -r canonical_name task_core <<< "${mapping}"
    header+=("task__${canonical_name}")
  done
  header+=("task_mean")

  {
    (IFS=,; printf '%s\n' "${header[*]}")

    # 再次遍历 manifest，把每个模型的 metrics.json 汇总成一行。
    while IFS=, read -r run_name mix_index global_step checkpoint_step_dir hf_model_dir; do
      [ -z "${run_name}" ] && continue
      hf_model_dir="${hf_model_dir%$'\r'}"
      if (( mix_index < mix_start || mix_index > mix_end )); then
        continue
      fi

      model_basename="$(basename "${hf_model_dir}")"
      metrics_file="${raw_results_dir}/${model_basename}/metrics.json"
      if [ ! -f "${metrics_file}" ]; then
        echo "[ERROR] Missing OLMES metrics file: ${metrics_file}" >&2
        exit 1
      fi

      declare -A SCORE_MAP=()
      while IFS=$'\t' read -r task_core score; do
        SCORE_MAP["${task_core}"]="${score}"
      done < <(
        jq -r '
          .tasks[]
          | select((.task_config.metadata.description // "") != "Aggregate metric")
          | select(((.alias // .task_config.metadata.alias // "") | contains(":bpb::")) | not)
          | [.task_config.task_core, (.metrics.primary_score | tostring)]
          | @tsv
        ' "${metrics_file}"
      )

      row=(
        "${run_name}"
        "${mix_index}"
        "${global_step}"
        "${checkpoint_step_dir}"
        "${hf_model_dir}"
        "${metrics_file}"
      )

      for weight_key in "${weight_keys[@]}"; do
        weight_value="$(
          jq -r --argjson idx "${mix_index}" --arg key "${weight_key}" '.mixes[$idx][$key][0]' "${mix_file}"
        )"
        row+=("${weight_value}")
      done

      task_values=()
      for mapping in "${task_mappings[@]}"; do
        IFS=$'\t' read -r canonical_name task_core <<< "${mapping}"
        score="${SCORE_MAP[${task_core}]:-}"
        if [ -z "${score}" ]; then
          echo "[ERROR] Missing task score for ${task_core} in ${metrics_file}" >&2
          exit 1
        fi
        row+=("${score}")
        task_values+=("${score}")
      done

      task_mean="$(
        printf '%s\n' "${task_values[@]}" \
        | awk '{sum += $1; count += 1} END {if (count == 0) {print ""} else {printf "%.15g", sum / count}}'
      )"
      row+=("${task_mean}")

      (IFS=,; printf '%s\n' "${row[*]}")
    done < <(tail -n +2 "${hf_manifest}")
  } > "${eval_csv}"

  echo "✓ OLMES metrics CSV written: ${eval_csv}"
}

# 输出目录。除非传入绝对路径，否则都放在当前 regmixer 根目录下面。
OUTPUT_DIR="${1:-${ROUND1A_OUTPUT_DIR:-outputs/round1a_all}}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

TRAIN_ROOT_DIR="${ROUND1A_TRAIN_ROOT_DIR:-${OUTPUT_DIR}/train_artifacts}"
if [[ "${TRAIN_ROOT_DIR}" != /* ]]; then
  TRAIN_ROOT_DIR="${PROJECT_ROOT}/${TRAIN_ROOT_DIR}"
fi

# 实验各阶段共享的路径约定。
CONFIG_FILE="src/regmixer/config/nemotron-cc-round1a-actual-quality.yaml"
MIX_FILE="${OUTPUT_DIR}/round1a_actual_quality_mixes.json"
GROUP_ID="round1a-actual-quality"
RUN_NAME_PREFIX="round1a-train"
LOG_DIR="${OUTPUT_DIR}/logs"
SUMMARY_DIR="${OUTPUT_DIR}/summaries"
EVAL_CSV="${OUTPUT_DIR}/eval_metrics.csv"
P_STAR_OUTPUT="${OUTPUT_DIR}/p_star_actual_quality.json"
EVAL_DIR="${OUTPUT_DIR}/eval"
HF_CACHE_DIR="${EVAL_DIR}/hf_models"
HF_MANIFEST="${EVAL_DIR}/hf_models_manifest.csv"
RAW_RESULTS_DIR="${EVAL_DIR}/raw_olmes"
PIPELINE_LOG="${OUTPUT_DIR}/round1a_pipeline.log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}" "${TRAIN_ROOT_DIR}" "${EVAL_DIR}"

# 把后续所有 stdout/stderr 同时写到总日志里，便于排查长任务。
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

# 允许从中间步骤恢复，避免重复跑已经完成的长阶段。
START_STEP="${ROUND1A_START_STEP:-1}"
if ! [[ "${START_STEP}" =~ ^[1-5]$ ]]; then
  echo "[ERROR] ROUND1A_START_STEP must be one of 1,2,3,4,5. Got: ${START_STEP}"
  exit 1
fi

# 固定这次 round1a 的搜索范围与训练资源配置。
MIX_START=0
MIX_END=14  # 15 variants (0-14)
TOTAL_VARIANTS=$((MIX_END - MIX_START + 1))
GPU_IDS="0,1,2,3,4,5,6,7"
GLOBAL_BATCH_SIZE=128

echo "============================================================================"
echo "Round-1a: Actual 质量桶比例搜索"
echo "============================================================================"
echo "Workdir: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Variants: ${MIX_START}-${MIX_END} (${TOTAL_VARIANTS} total)"
echo "GPUs: ${GPU_IDS}"
echo "Start step: ${START_STEP}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train artifacts root: ${TRAIN_ROOT_DIR}"
echo "Pipeline log: ${PIPELINE_LOG}"
echo "============================================================================"
echo ""

# Step 1: 生成 15 个待搜索的 mix 配置。
if [ "${START_STEP}" -le 1 ]; then
  echo "[Step 1/5] Generating mixes..."
  PYTHONPATH=src python -m regmixer.cli generate-mixes \
    --config "${CONFIG_FILE}" \
    -o "${MIX_FILE}"

  if [ ! -f "${MIX_FILE}" ]; then
    echo "[ERROR] Mix file not found: ${MIX_FILE}"
    echo "        generate-mixes completed but did not write expected output."
    exit 1
  fi

  echo "✓ Mixes generated: ${MIX_FILE}"
  echo ""
else
  require_file "${MIX_FILE}"
  echo "[Step 1/5] Skipped (reusing mixes): ${MIX_FILE}"
  echo ""
fi

# Step 2: 本地并行训练所有 variant，训练日志写到 logs/，摘要写到 summaries/。
if [ "${START_STEP}" -le 2 ]; then
  echo "[Step 2/5] Training ${TOTAL_VARIANTS} variants (8 GPU parallel)..."
  echo "⚠️  This will take approximately 26-52 days (depending on throughput)"
  echo "    Use tmux/screen to keep the process running in background"
  echo ""
  read -p "Press Enter to start training (or Ctrl+C to cancel)..."

  export WANDB_MODE=offline
  PYTHONPATH=src python scripts/parallel_train.py \
    --config "${CONFIG_FILE}" \
    --mix-file "${MIX_FILE}" \
    --group-id "${GROUP_ID}" \
    --run-name-prefix "${RUN_NAME_PREFIX}" \
    --log-dir "${LOG_DIR}" \
    --summary-dir "${SUMMARY_DIR}" \
    --mix-start "${MIX_START}" \
    --mix-end "${MIX_END}" \
    --gpu-ids "${GPU_IDS}" \
    --workdir "${PROJECT_ROOT}" \
    --output-root-dir "${TRAIN_ROOT_DIR}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}"

  echo "✓ Training completed"
  echo ""
else
  require_dir "${SUMMARY_DIR}"
  echo "[Step 2/5] Skipped (reusing training summaries): ${SUMMARY_DIR}"
  echo ""
fi

# Step 3: 根据 training summary 定位 checkpoint，并统一转换成 HF 目录。
if [ "${START_STEP}" -le 3 ]; then
  echo "[Step 3/5] Converting checkpoints to HF model directories..."
  PYTHONPATH=src python -m regmixer.cli convert-hf \
    --config "${CONFIG_FILE}" \
    --log-dir "${LOG_DIR}" \
    --hf-cache-dir "${HF_CACHE_DIR}" \
    --output-manifest "${HF_MANIFEST}" \
    --mix-start "${MIX_START}" \
    --mix-end "${MIX_END}"

  echo "✓ HF conversion completed: ${HF_MANIFEST}"
  echo ""
else
  require_file "${HF_MANIFEST}"
  echo "[Step 3/5] Skipped (reusing HF manifest): ${HF_MANIFEST}"
  echo ""
fi

# Step 4: 逐个 HF checkpoint 跑 OLMES，然后把 raw_olmes 聚合成 eval_metrics.csv。
if [ "${START_STEP}" -le 4 ]; then
  echo "[Step 4/5] Evaluating all variants with OLMES..."
  while IFS=, read -r run_name mix_index global_step checkpoint_step_dir hf_model_dir; do
    [ -z "${run_name}" ] && continue
    hf_model_dir="${hf_model_dir%$'\r'}"
    if (( mix_index < MIX_START || mix_index > MIX_END )); then
      continue
    fi

    model_basename="$(basename "${hf_model_dir}")"
    raw_output_dir="${RAW_RESULTS_DIR}/${model_basename}"
    echo "[INFO] OLMES eval for ${run_name} -> ${raw_output_dir}"
    run_single_olmes_eval "${hf_model_dir}" "${raw_output_dir}"
  done < <(tail -n +2 "${HF_MANIFEST}")

  echo "[Step 4/5] Aggregating OLMES metrics into CSV..."
  aggregate_olmes_to_csv "${OUTPUT_DIR}" "${MIX_START}" "${MIX_END}"
  echo "✓ Evaluation completed: ${EVAL_CSV}"
  echo ""
else
  require_file "${EVAL_CSV}"
  echo "[Step 4/5] Skipped (reusing eval CSV): ${EVAL_CSV}"
  echo ""
fi

# Step 5: 用聚合后的 eval CSV 拟合最优 mixture，输出 p*。
if [ "${START_STEP}" -le 5 ]; then
  echo "[Step 5/5] Fitting regression model..."
  PYTHONPATH=src python -m regmixer.cli fit-mixture \
    --config "${CONFIG_FILE}" \
    --eval-metrics "${EVAL_CSV}" \
    --output "${P_STAR_OUTPUT}"

  echo "✓ Regression completed: ${P_STAR_OUTPUT}"
  echo ""
fi

# Display results
echo "============================================================================"
echo "Round-1a Complete!"
echo "============================================================================"
echo "Optimal weights (p*_actual_quality):"
cat "${P_STAR_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. Review ${P_STAR_OUTPUT}"
echo "  2. Copy weights to nemotron-cc-round1b-kind2.yaml (actual topics)"
echo "  3. Run Round-1b: scripts/run_round1b.sh"
echo "============================================================================"
