# Benchmarking

How to read the performance and accuracy data embedded in the result JSON. For the schema of each field, see [Data Formats](data-formats.md#performancesummary).

## Evaluation Axes

| Axis | Metrics | Scope |
|---|---|---|
| **Performance** | tokens/sec, latency, wall-clock time | Extract and Select modes. |
| **Accuracy** | precision, recall, F1, accuracy | Select mode only, comparing `term_id` against `mapping answer ID`. |

The `extraction answer` column in the mapping TSV is an auxiliary annotation (originally from MetaSRA) and is not used for evaluation; only `mapping answer ID` is treated as ground truth. Accuracy is therefore only meaningful for Select mode.

## Why tokens/sec, not GPU utilization

`nvidia-smi` reports SM (Streaming Multiprocessor) occupancy. LLM inference is memory-bandwidth-bound, so a GPU can show 5% SM utilisation while being completely saturated on memory bandwidth. `tokens_per_sec = eval_count / eval_duration` directly measures generation rate and is the right metric for:

- Comparing pipeline configurations (parallelism, batch size).
- Detecting GPU saturation.
- Estimating wall-clock time for a given workload.

## LLM Timing Fields

Each LLM call records nanosecond-precision timings via `LlmTimingFields` (see [Data Formats](data-formats.md#llmtimingfields)). Useful identities:

```
total_duration ≈ load_duration + prompt_eval_duration + eval_duration + (internal overhead)
latency_sec    = (total_duration - load_duration) / 1e9
tokens_per_sec = eval_count / (eval_duration / 1e9)
```

`LlmTimingSummary` (in `PerformanceSummary.ner_llm_timing` and `PerformanceSummary.select_llm_timing`) aggregates these across all calls of a stage. Schema: [Data Formats](data-formats.md#llmtimingsummary).

## Diagnosing Execution Time Variance

When two runs of the same workload differ, look at:

- **`load_duration` spikes** -- the model was unloaded between requests. Inspect `mean_load_duration_sec` and `max_load_duration_sec`. Likely Ollama eviction under memory pressure or load timeout.
- **`tokens_per_sec` p99 vs p50 gap** -- intermittent hardware interference, KV cache pressure, or thermal throttling.
- **`sum(total_duration)` << wall-clock** -- requests are spending time in the Ollama queue. Reduce concurrency or `OLLAMA_NUM_PARALLEL`.
- **Stage time imbalance** -- compare `ner_sec`, `ontology_search_sec`, `text2term_sec`, and `llm_select_sec` in `stage_timings[]` to find the bottleneck. `ontology_search_sec` should stay sub-second; a spike there means an index rebuild, not ontology content.
- **`text2term_sec` without `disk_io.text2term_cache_build_sec`** -- the text2term cache acronym was not registered before per-batch calls, so `map_terms()` paid a per-call cache miss. Verify `build_text2term_cache()` ran at startup.
- **`asyncio.gather` tail** -- `llm_select_sec` is the maximum of all concurrent Stage 3 calls per batch, so one slow call dominates.
- **Shared-cluster noise** -- on NIG Slurm, other jobs compete for GPU, network, and storage. Compare runs at different times.

## Detecting GPU Saturation

Run the same workload at several concurrency levels (e.g. 1, 4, 16, 64, 256) and read `mean_tokens_per_sec` (`T_N`) from each result. Compute `N * T_N`:

| Observation | Interpretation |
|---|---|
| `N * T_N` increases linearly | GPU is underutilised. Increase concurrency. |
| `N * T_N` plateaus | GPU is saturated. Optimal concurrency reached. |
| `N * T_N` decreases | Contention overhead. Reduce concurrency. |
| `T_N` drops sharply at some N | Queue pressure. Use the previous N. |

## Reproducibility

- **Warm-up.** Cold starts inflate `load_duration`. Send a few dummy requests before timing.
- **Multiple runs.** Report median +- IQR over at least 3 runs. Mean is sensitive to outliers.
- **Normalise.** Wall-clock time depends on the token budget of the input. Compare `tokens_per_sec` across runs to factor that out.

## Reading PerformanceSummary

| Field | What to check |
|---|---|
| `performance.total_wall_sec` | End-to-end wall-clock time. |
| `performance.total_input_entries` / `completed_count` | Did every entry complete? |
| `performance.ner_llm_timing.mean_tokens_per_sec` | NER GPU throughput. |
| `performance.select_llm_timing.mean_tokens_per_sec` | Stage 3 GPU throughput (Select only). |
| `performance.ner_llm_timing.mean_load_duration_sec` / `max_load_duration_sec` | Warm-up effectiveness. |
| `performance.ner_llm_timing.p50_latency_sec` vs `p99_latency_sec` | Tail latency. |
| `performance.stage_timings[].{ner_sec, ontology_search_sec, text2term_sec, llm_select_sec, resume_write_sec}` | Per-batch breakdown of the Select pipeline. |
| `performance.disk_io.text2term_cache_build_sec` (length and total) | First-run cost of building the text2term cache (Select only). |
| `performance.disk_io.index_cache_load_sec` vs `index_build_from_file_sec` | Cache hits vs rebuilds (Select only). |
| `evaluation.{accuracy, precision, recall, f1}` | Accuracy regression check (Select only, 0-1 ratios). |

Field definitions live in [Data Formats](data-formats.md#performancesummary); this page is the interpretation guide only.

## Multi-Model Bench Script

`scripts/run_model_bench.sh` runs Select mode against `tests/data/eval_biosample.json` (600 entries) with `tests/data/eval_gold_standard.tsv` as the gold standard, sweeping a fixed list of Ollama models under `--no-reasoning --batch-size 128`. Outputs:

- `bench-logs/{run_name}.log` / `.err` per model.
- `bench-logs/summary.tsv` -- `model`, `run_name`, `wall_sec`, `result_json`, `status`.
- `bsllmner2-results/select/select_{run_name}.json` per model.

The script intentionally does not set `set -e`; one model failing does not abort the sweep.
