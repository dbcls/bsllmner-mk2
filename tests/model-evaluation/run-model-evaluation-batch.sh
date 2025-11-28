#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
  "deepseek-r1:8b"
  "deepseek-r1:32b"
  "gemma3:4b"
  "gemma3:12b"
  "gemma3:27b"
  "gpt-oss:20b"
  "llama3.1:8b"
  "phi4:14b"
  "qwen3:4b"
  "qwen3:8b"
  "qwen3:32b"
)

BS_ENTRIES="/app/tests/zenodo-data/biosample_cellosaurus_mapping_testset.json"
SELECT_CONFIG="/app/scripts/select-config.json"
BASE_RUN_NAME="models-with-large-dataset"

LOG_BASE="${SCRIPT_DIR}/model-evaluation-batch-logs"
mkdir -p "$LOG_BASE"

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | tr ':/' '__')
    run_name="${BASE_RUN_NAME}-${model_safe}"
    log_file="${LOG_BASE}/${model_safe}.log"

    echo "=== Running model: $model (run-name: $run_name) ==="

    {
        start_ts=$(date +%s)
        echo "===== Start: $(date '+%Y-%m-%d %H:%M:%S') (epoch: $start_ts) ====="

        bsllmner2_select \
            --bs-entries "$BS_ENTRIES" \
            --select-config "$SELECT_CONFIG" \
            --run-name "$run_name" \
            --resume \
            --thinking false \
            --no-reasoning \
            --model "$model"

        end_ts=$(date +%s)
        echo "===== Finish: $(date '+%Y-%m-%d %H:%M:%S') (epoch: $end_ts) ====="

        duration=$(( end_ts - start_ts ))
        echo "===== Duration: ${duration} sec ====="
    } &> "$log_file"

    echo "=== Done: $model (log: $log_file) ==="
done
