# Model Evaluation

Benchmarking environment for evaluating LLM model speed and accuracy on Ollama.

## Overview

Evaluation is performed on two axes:

| Axis | Metrics | Scope |
|------|---------|-------|
| **Speed** | tokens/sec, total throughput, wall-clock time | Extract + Select |
| **Accuracy** | Precision, Recall, F1, Accuracy | Select only (vs human-curated gold standard) |

## Dataset

- **Input**: `tests/data/eval_biosample.json` (600 BioSample entries)
- **Gold standard**: `tests/data/eval_gold_standard.tsv` (human-curated ontology mappings for `cell_line` field)

## Quick start

```bash
# 1. Determine appropriate num_ctx for a model
python tests/model-evaluation/analyze_token_usage.py \
    --model qwen3:8b \
    --bs-entries tests/data/eval_biosample.json \
    --select-config tests/data/eval_select_config.json \
    --num-ctx 4096

# 2. Find optimal OLLAMA_NUM_PARALLEL via ternary search
python tests/model-evaluation/speed_exploration.py \
    --model qwen3:8b \
    --num-ctx 4096

# 3. Compare multiple models (after running step 2 for each model)
python tests/model-evaluation/model_comparison.py \
    --results-dir tests/model-evaluation/results
```

## Scripts

### `analyze_token_usage.py`

Measures actual token usage for a given model and dataset, and recommends an appropriate `num_ctx` value.

```bash
python tests/model-evaluation/analyze_token_usage.py \
    --model qwen3:8b \
    --bs-entries tests/data/eval_biosample.json \
    --select-config tests/data/eval_select_config.json \
    --num-ctx 4096
```

Example output:

```
Stage       calls      max      p99      p95     mean
-----------------------------------------------------
extract       600     1326     1290     1100      562
select       1800     2704     2600     1608     1372

Recommended num_ctx: 4096  (next power of 2 above max)
```

### `speed_exploration.py`

Finds the optimal `OLLAMA_NUM_PARALLEL` that maximizes total throughput for a given model, using ternary search.

#### Background

Ollama's `OLLAMA_NUM_PARALLEL` controls the maximum number of concurrent requests per model. This value significantly affects throughput, but the optimal value depends on model size and available VRAM.

- `NUM_PARALLEL` too low: GPU is underutilized, low throughput
- `NUM_PARALLEL` too high: increased KV cache VRAM consumption, context length limits, contention reduces throughput

Throughput (`NUM_PARALLEL * per-request tokens/sec`) follows a unimodal curve against NUM_PARALLEL, making ternary search an efficient way to find the peak.

`num_ctx` (context length) also shares VRAM with `NUM_PARALLEL`, so different `num_ctx` values may yield different optimal `NUM_PARALLEL` values.

#### How to determine `num_ctx`

Use `analyze_token_usage.py` to measure actual token usage for a given model and dataset, and obtain the recommended value.

Since `num_ctx` and `NUM_PARALLEL` share VRAM, increasing `num_ctx` may lower the upper bound for `NUM_PARALLEL`. Keeping `num_ctx` no larger than necessary is key to maximizing throughput.

#### Search algorithm

Ternary search finds the optimal NUM_PARALLEL in O(log N) evaluations:

```
lo=1, hi=64
while hi - lo >= 3:
    m1 = lo + (hi-lo)//3
    m2 = hi - (hi-lo)//3
    if evaluate(m1) < evaluate(m2):
        lo = m1+1
    else:
        hi = m2-1
best = max(range(lo, hi+1), key=evaluate)
```

Each evaluation takes the median of 3 runs to reduce hardware noise.

#### Usage

```bash
# Run on the host machine (where docker compose is available)

# Single model, single num_ctx
python tests/model-evaluation/speed_exploration.py \
    --model qwen3:32b \
    --num-ctx 8192

# Compare multiple num_ctx values
python tests/model-evaluation/speed_exploration.py \
    --model qwen3:32b \
    --num-ctx 4096 8192
```

#### Execution flow

1. Ternary search: `lo=1, hi=64`
2. Compute `m1`, `m2`
3. For each candidate `NUM_PARALLEL`:
   a. Restart Ollama container via `docker compose` (`OLLAMA_NUM_PARALLEL=$N`)
   b. Wait for health check
   c. Send warm-up request
   d. Run `docker exec bsllmner-mk2-app bsllmner2_select` on 50-entry subset, 3 times
   e. Extract `mean_tokens_per_sec` from result JSON, compute median
4. Narrow search range and repeat
5. Verify optimal value with neighbors +/-1
6. Full validation: 600 entries, 3 runs

#### Output

Exploration result (per model x num_ctx):

| Field | Description |
|-------|-------------|
| model | Model name |
| num_ctx | Context length |
| best_num_parallel | Optimal OLLAMA_NUM_PARALLEL |
| best_total_throughput | NUM_PARALLEL * mean_tokens_per_sec |
| search_history | All (num_parallel, throughput) pairs evaluated |

Validation result (per model, full 600 entries, 3 runs):

| Field | Description |
|-------|-------------|
| model | Model name |
| num_parallel | Optimal NUM_PARALLEL |
| num_ctx | Optimal context length |
| median_tokens_per_sec | Median per-request tokens/sec |
| iqr_tokens_per_sec | IQR of tokens/sec |
| median_wall_sec | Median wall-clock time |
| f1 | F1-score |
| precision | Precision |
| recall | Recall |

### `model_comparison.py`

Aggregates validation results from multiple models and generates a unified summary table and Pareto analysis plot.

```bash
python tests/model-evaluation/model_comparison.py \
    --results-dir tests/model-evaluation/results
```

Output:

- `validation_summary.tsv` -- side-by-side comparison of all models (speed + accuracy)
- `pareto_plot.png` -- scatter plot of total throughput vs F1 with Pareto frontier

Use `--no-plot` to generate TSV only (no matplotlib dependency required).

## Target models

```plain
deepseek-r1:8b
deepseek-r1:32b
gemma3:4b
gemma3:12b
gemma3:27b
gpt-oss:20b
llama3.1:8b
phi4:14b
qwen3:4b
qwen3:8b
qwen3:32b
```

## Fixed Ollama parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Structured JSON extraction requires deterministic output |
| `seed` | 0 | Reproducibility |
| `OLLAMA_KV_CACHE_TYPE` | q8_0 | ~1/2 VRAM of f16, negligible quality impact |
| `OLLAMA_FLASH_ATTENTION` | 1 | Required for KV cache quantization |
| `OLLAMA_SCHED_SPREAD` | 1 | Multi-GPU load distribution |

## Related docs

- [docs/benchmarking.md](../../docs/benchmarking.md) -- Benchmarking methodology, metrics, PerformanceSummary reference
- [docs/select-mode.md](../../docs/select-mode.md) -- Select pipeline (3-stage: NER -> ontology search -> LLM selection)
- [docs/configuration.md](../../docs/configuration.md) -- Ollama environment variables reference
