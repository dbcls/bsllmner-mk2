# Benchmarking

## Evaluation axes

Benchmarking covers two orthogonal axes:

| Axis | Metrics | Scope |
|------|---------|-------|
| **Performance** | tokens/sec, latency, wall-clock time | Both Extract and Select modes |
| **Accuracy** | precision, recall, F1 vs human-curated gold standard | **Select mode only** |

### Why accuracy evaluation is Select-mode only

The mapping TSV `extraction_answer` column is the output of a previous tool (MetaSRA), not a human-curated ground truth. Therefore it cannot serve as a reliable gold standard for Extract evaluation.

The `mapping_answer_id` column, on the other hand, is human-curated and compares the pipeline's final ontology term selection against a known-correct answer. Accuracy evaluation is performed exclusively via Select mode using this column.

## Why tokens/sec, not GPU utilization

`nvidia-smi` reports SM (Streaming Multiprocessor) utilization, which measures compute occupancy. LLM inference, however, is memory-bandwidth-bound: the bottleneck is moving weights from VRAM to the compute units, not the compute itself. A GPU can show 5% SM utilization while being completely saturated on memory bandwidth.

**tokens/sec** (`eval_count / eval_duration`) directly measures how fast the model is generating output tokens. This is the correct metric for:

- Comparing pipeline configurations (parallelism, batch size)
- Detecting GPU saturation
- Estimating wall-clock time for a given workload

## LLM timing fields

Each `ExtractEntry` persists an `LlmTimingFields` object with the subset of Ollama timing data needed for benchmarking. `ChatResponse` objects are kept in memory during the run for aggregate statistics but are not saved to disk.

Every `ChatResponse` from Ollama contains nanosecond-precision timing data:

```
total_duration = load_duration + prompt_eval_duration + eval_duration + internal overhead
```

| Field | Unit | Description |
|-------|------|-------------|
| `total_duration` | ns | Wall-clock time inside Ollama for this request |
| `load_duration` | ns | Model load/unload time (high on cold start) |
| `prompt_eval_count` | tokens | Number of prompt tokens evaluated |
| `prompt_eval_duration` | ns | Time spent on prompt evaluation |
| `eval_count` | tokens | Number of generated tokens |
| `eval_duration` | ns | Time spent on token generation |

## Diagnosing execution time variance

Execution time varies between runs even with identical inputs. The BenchmarkSummary JSON provides data to isolate the cause.

### Layer 1: LLM internal

- **`load_duration` spikes**: Model was unloaded from GPU between requests. Indicates Ollama evicted the model (memory pressure, timeout). Check `max_load_duration_sec` and `mean_load_duration_sec`.
- **`eval_duration / eval_count` variance**: Per-token generation speed fluctuates. May indicate KV cache pressure or Ollama queuing.
- **`total_duration` vs wall-clock gap**: If `sum(total_duration)` is much less than wall-clock time, requests are spending time in the Ollama queue.

### Layer 2: GPU / hardware

- Thermal throttling reduces clock speed under sustained load.
- Memory bandwidth contention from other processes sharing the GPU.
- Compare `tokens_per_sec` p50 vs p99: a large gap suggests intermittent hardware interference.

### Layer 3: Pipeline structure

- `asyncio.gather` takes the maximum of all coroutine durations, so one slow request dominates batch time.
- Resume file writes (`resume_write_sec`) grow linearly with accumulated outputs.

### Layer 4: OS / environment

- On shared clusters (e.g., NIG Slurm), other jobs compete for GPU, network, and storage.
- Compare runs at different times to isolate environmental noise.

## Detecting GPU saturation

GPU saturation means adding more parallelism no longer increases total throughput.

### Method

1. Run the same workload with different concurrency levels (e.g., 1, 4, 16, 64, 256).
2. From BenchmarkSummary, compute:
   - `T_N`: mean per-request tokens/sec at concurrency N
   - `N * T_N`: estimated total throughput

### Interpretation

| Observation | Meaning |
|-------------|---------|
| `N * T_N` increases linearly | GPU is underutilized; increase concurrency |
| `N * T_N` plateaus | GPU is saturated; optimal concurrency reached |
| `N * T_N` decreases | Contention overhead; reduce concurrency |
| `T_N` drops sharply at some N | Queue pressure; consider the previous N as optimal |

## Ensuring reproducibility

### Warm-up

Cold starts inflate `load_duration`. Before benchmarking, send a few dummy requests to ensure the model is loaded and the KV cache is initialized.

### Multiple runs

Report **median +- IQR** (interquartile range) over at least 3 runs. Mean is sensitive to outliers; median is not.

### Normalize by tokens/sec

Wall-clock time depends on the number of tokens generated, which varies with input. Use `tokens_per_sec` to compare across different inputs.

## Reading BenchmarkSummary JSON

Saved to `bsllmner2-results/benchmarks/{run_name}_benchmark.json`.

### Top-level fields

| Field | What to check |
|-------|---------------|
| `total_wall_sec` | End-to-end wall-clock time |
| `total_entries` / `completed_count` | Did all entries complete? |
| `select_accuracy` / `select_matched_entries` | Select accuracy regression check (Select mode only) |
| `select_precision` / `select_recall` / `select_f1` | Select precision/recall/f1 (%) (Select mode only) |

### `ner_llm_timing` / `select_llm_timing`

| Field | What to check |
|-------|---------------|
| `mean_tokens_per_sec` | GPU throughput indicator |
| `p50_tokens_per_sec` | Typical throughput |
| `mean_load_duration_sec` | Warm-up effectiveness |
| `max_load_duration_sec` | Worst-case model load |
| `p50_latency_sec` vs `p99_latency_sec` | Tail latency (outlier detection) |
| `total_prompt_tokens` / `total_eval_tokens` | Token budget |

### `stage_timings[]`

Per-batch breakdown. Compare `ner_sec`, `ontology_search_sec`, `text2term_sec`, `llm_select_sec` to identify the bottleneck stage.

### `disk_io`

| Field | What to check |
|-------|---------------|
| `index_cache_load_sec` | Cache hit speed |
| `index_cache_save_sec` | First-run overhead |
| `resume_write_sec` | I/O growth over batches |
