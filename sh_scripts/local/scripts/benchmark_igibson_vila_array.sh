#!/usr/bin/env bash
set -euo pipefail

# Default flag values
SEED=1
EXPERIMENT_NAME=""
PROMPT_PATH="data/prompts/planning/vila_igibson_json.md"
USE_ACT=false
USE_COT=false

# Parse additional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --act_prompt)
      USE_ACT=true
      shift
      ;;
    --use_cot_prompt)
      USE_COT=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

if [ "$USE_ACT" = true ] && [ "$USE_COT" = true ]; then
  PROMPT_PATH="data/prompts/planning/react_igibson_json.md"
elif [ "$USE_ACT" = true ]; then
  PROMPT_PATH="data/prompts/planning/act_igibson_json.md"
elif [ "$USE_COT" = true ]; then
  PROMPT_PATH="data/prompts/planning/vila_igibson_json_cot.md"
else
  PROMPT_PATH="data/prompts/planning/vila_igibson_json.md"
fi

# No distinction between big and small models as we assume local resources are the same
models=(
  "OpenGVLab/InternVL3-8B"
  "OpenGVLab/InternVL3_5-8B"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
  "google/gemma-3-12b-it"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "allenai/Molmo-7B-D-0924"
  "microsoft/Phi-4-multimodal-instruct"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "deepseek-ai/deepseek-vl2"
  "CohereLabs/aya-vision-8b"
  "nvidia/Cosmos-Reason2-8B"
  "OpenGVLab/InternVL3-78B"
  "OpenGVLab/InternVL3_5-30B-A3B"
  "OpenGVLab/InternVL3_5-38B"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "Qwen/Qwen3-VL-30B-A3B-Instruct"
  "Qwen/Qwen3-VL-32B-Instruct"
  "google/gemma-3-27b-it"
  "llava-hf/llava-onevision-qwen2-72b-ov-hf"
  "CohereLabs/aya-vision-32b"
)

splits=("simple" "medium" "hard")
max_steps=(10 20 30)

# Server/job constants
JOB_TYPE_IDX=3
job_id=

mamba activate ./viplan_env

for MODEL in "${models[@]}"; do
  model_short="${MODEL##*/}"

  for idx in "${!splits[@]}"; do
    PROBLEM_SPLIT="${splits[$idx]}"
    MAX_STEPS="${max_steps[$idx]}"
    echo "=== Running MODEL: ${MODEL}, SPLIT: ${PROBLEM_SPLIT}, MAX_STEPS: ${MAX_STEPS} ==="

    PORT=$((8000 + job_id + 100 * JOB_TYPE_IDX))
    NODE=$(hostname -s)
    BASE_URL="http://${NODE}:${PORT}"

    echo "------------------------------------------------------------"
    echo "Job #$job_id: model=${model_short}, split=${PROBLEM_SPLIT}, max_steps=${MAX_STEPS}"
    echo "Starting server on port ${PORT}…"
    make run PORT=${PORT} &
    sleep 120  # give server time to spin up

    # Paths
    DOMAIN_FILE="data/planning/igibson/domain.pddl"
    PROBLEMS_DIR="data/planning/igibson/${PROBLEM_SPLIT}"

    if [[ -n "${EXPERIMENT_NAME}" ]]; then
      OUTPUT_DIR="results/planning/igibson/${EXPERIMENT_NAME}/vila/${PROBLEM_SPLIT}/${model_short}"
    else
      OUTPUT_DIR="results/planning/igibson/${PROBLEM_SPLIT}/vila/${model_short}"
    fi
    mkdir -p "${OUTPUT_DIR}"

    # Run the benchmark
    python3 -m viplan.experiments.benchmark_vlm_as_planner \
      --base_url "${BASE_URL}" \
      --model_name "${MODEL}" \
      --domain_name "viplan-hh" \
      --domain_file "${DOMAIN_FILE}" \
      --problems_dir "${PROBLEMS_DIR}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --max_steps "${MAX_STEPS}" \
      --max_new_tokens 1024 \
      --seed "${SEED}"

    # clean up the server before next job
    echo "Stopping server on port ${PORT}…"
    pkill -f "make run.*PORT=${PORT}" || true

    job_id=$(( job_id + 1 ))
  done
done

echo "All ${job_id} jobs complete."