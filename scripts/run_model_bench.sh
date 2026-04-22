#!/usr/bin/env bash
# Run bsllmner2_select 600-entry eval against a fixed list of candidate
# Ollama models under uniform conditions (--no-reasoning, --batch-size 128).
#
# Intended to run on dbcls-ai01 where docker compose defines the ollama +
# app services. Safe to launch detached:
#
#   cd /data/bsllmner-mk2
#   mkdir -p bench-logs
#   nohup bash scripts/run_model_bench.sh > bench-logs/nohup.out 2>&1 &
#   disown
#
# Per-model pull + run log lands under bench-logs/<run_name>.{log,err}. A
# single bench-logs/bench.log carries the overall progress, and
# bench-logs/summary.tsv is the final pass/fail table.
#
# NOTE: `set -e` is intentionally omitted so that a failure for one model
# does not abort the whole batch.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/bench-logs"
mkdir -p "${LOG_DIR}"
GLOBAL_LOG="${LOG_DIR}/bench.log"
SUMMARY="${LOG_DIR}/summary.tsv"

BS_ENTRIES="tests/data/eval_biosample.json"
MAPPING="tests/data/eval_gold_standard.tsv"
SELECT_CONFIG="scripts/select-config-hg38.json"
BATCH_SIZE=128
RUN_SUFFIX="nr-b128"

# VRAM 小さい順。途中で重いモデルに行く前に異常を検知できる。
MODELS=(
  "phi4:14b"
  "mistral-small3.1:24b"
  "gpt-oss:20b"
  "qwen3:30b"
  "qwen3:32b"
  "gemma4:31b"
  "qwen3.5:35b"
  "llama3.3:70b"
)

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${GLOBAL_LOG}"
}

sanitize() {
  # qwen3.5:35b -> qwen3_5_35b
  echo "$1" | tr ':/.' '___'
}

log "=== bench run start (pid=$$) ==="
log "REPO_ROOT=${REPO_ROOT}"
log "batch_size=${BATCH_SIZE} reasoning=off"
log "models=${MODELS[*]}"

log "--- docker compose ps ---"
docker compose ps 2>&1 | tee -a "${GLOBAL_LOG}"
log "--- nvidia-smi (inside ollama container) ---"
docker compose exec -T ollama nvidia-smi 2>&1 | tee -a "${GLOBAL_LOG}"
log "--- ollama list ---"
docker compose exec -T ollama ollama list 2>&1 | tee -a "${GLOBAL_LOG}"

if [ ! -s "${SUMMARY}" ]; then
  printf "model\trun_name\twall_sec\tresult_json\tstatus\n" > "${SUMMARY}"
fi

for MODEL in "${MODELS[@]}"; do
  SAFE="$(sanitize "${MODEL}")"
  RUN_NAME="eval-${SAFE}-hg38-${RUN_SUFFIX}"
  LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
  ERR_FILE="${LOG_DIR}/${RUN_NAME}.err"
  RESULT_JSON="${REPO_ROOT}/bsllmner2-results/select/select_${RUN_NAME}.json"

  log "========== MODEL ${MODEL} (run_name=${RUN_NAME}) =========="

  log "[${MODEL}] ollama pull start"
  if docker compose exec -T ollama ollama pull "${MODEL}" \
        >>"${LOG_FILE}" 2>>"${ERR_FILE}"; then
    log "[${MODEL}] ollama pull done"
  else
    log "[${MODEL}] ollama pull FAILED (see ${ERR_FILE}) -- skipping"
    printf "%s\t%s\t-\t-\tpull_failed\n" "${MODEL}" "${RUN_NAME}" >> "${SUMMARY}"
    continue
  fi

  log "[${MODEL}] bsllmner2_select start"
  START=$(date +%s)
  if docker compose exec -T app bsllmner2_select \
        --bs-entries "${BS_ENTRIES}" \
        --mapping "${MAPPING}" \
        --model "${MODEL}" \
        --select-config "${SELECT_CONFIG}" \
        --run-name "${RUN_NAME}" \
        --batch-size "${BATCH_SIZE}" \
        --no-reasoning \
        --debug \
        >>"${LOG_FILE}" 2>>"${ERR_FILE}"; then
    END=$(date +%s)
    WALL=$((END - START))
    log "[${MODEL}] bsllmner2_select done (${WALL}s)"
    STATUS="ok"
  else
    END=$(date +%s)
    WALL=$((END - START))
    log "[${MODEL}] bsllmner2_select FAILED after ${WALL}s (see ${ERR_FILE})"
    STATUS="run_failed"
  fi

  if [ "${WALL}" -gt 18000 ]; then
    log "[${MODEL}] WARNING: wall_sec=${WALL} exceeded 5h -- likely CPU fallback"
    log "[${MODEL}] restarting ollama to refresh CUDA"
    docker compose restart ollama 2>&1 | tee -a "${GLOBAL_LOG}"
    sleep 15
    STATUS="${STATUS}_slow"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${MODEL}" "${RUN_NAME}" "${WALL}" "${RESULT_JSON}" "${STATUS}" \
    >> "${SUMMARY}"
done

log "=== bench run all done ==="
log "--- summary ---"
cat "${SUMMARY}" | tee -a "${GLOBAL_LOG}"
