#!/usr/bin/env bash
set -euo pipefail

# Executed inside the container at /app

usage() {
    echo "Usage: $0 --batch-name <name"
    exit 1
}

BATCH_NAME="default-batch"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-name)
      BATCH_NAME="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      break
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
  "deepseek-r1:8b"
  "deepseek-r1:32b"
  "gemma3:12b"
  "gemma3:27b"
  "phi4:14b"
  "qwen3:8b"
  "qwen3:32b"
  "gpt-oss:20b"
)

BS_ENTRIES="/app/tests/data/eval_biosample.json"
SELECT_CONFIG="/app/scripts/select-config-hg38.json"

RUN_NAME_BASE="model-eval-${BATCH_NAME}"
LOG_BASE="${SCRIPT_DIR}/model-evaluation-batch-logs/${BATCH_NAME}"

mkdir -p "$LOG_BASE"

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | tr ':/' '__')
    run_name="${RUN_NAME_BASE}-${model_safe}"
    log_file="${LOG_BASE}/${model_safe}.log"

    docker restart bsllmner-mk2-ollama > /dev/null 2>&1 || true

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
