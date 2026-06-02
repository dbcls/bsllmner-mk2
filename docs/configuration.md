# Configuration

Reference for the environment variables bsllmner-mk2 reads and the Ollama tuning knobs set in `compose.yml`.

## Environment Variables

### Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. Inside `compose.yml`, `app` sets this to `http://bsllmner-mk2-ollama:11434`. Can also be overridden by `--ollama-host`. |
| `BSLLMNER2_OLLAMA_CONCURRENCY` | `256` | Maximum number of concurrent in-flight LLM calls from `OllamaBackend` to the Ollama server. Tune downwards (e.g. `8`) when the server is the bottleneck or to reduce queueing during debugging. Invalid or non-positive values fall back to `256` with a WARNING. |

### CLI / Runtime

| Variable | Default | Description |
|---|---|---|
| `BSLLMNER2_DEBUG` | unset | Truthy values (`true`/`1`/`yes`/`on`, case-insensitive) enable DEBUG-level logging. Equivalent to `--debug`. |

### Directories

| Variable | Default | Description |
|---|---|---|
| `BSLLMNER2_RESULT_DIR` | `$PWD/bsllmner2-results` | Root for result and resume files. Subdirectories `extract/` and `select/` are created on demand. |
| `BSLLMNER2_TMP_DIR` | `<tempfile.gettempdir()>/bsllmner2-<uid>` | Scratch directory. Resolved via Python's `tempfile.gettempdir()` (typically `/tmp` on Linux) with the process UID appended. Reserved for future use; the current implementation does not write anything inside it. |

### Cache

| Variable | Default | Description |
|---|---|---|
| `BSLLMNER2_INDEX_CACHE_DIR` | `ontology/index_cache` | Serialised `OntologyIndex` cache, one file per ontology (`{owl_name}_nofilter_v2.pkl`). |
| `BSLLMNER2_TEXT2TERM_CACHE_DIR` | `ontology/text2term_cache` | text2term prebuilt cache, registered under acronym `{owl_stem}_nofilter`. |

Cache layout and cleanup are documented in [Ontology Preparation](ontology.md#cache-layout).

## Ollama Performance Tuning (Docker Compose)

Set on the `ollama` service in `compose.yml`. Override via shell exports before `docker compose up`, or by editing the file directly.

| Variable | Value | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | In-container bind address. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV cache quantisation. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Enable Flash Attention. |
| `OLLAMA_NUM_PARALLEL` | `16` | Parallel inference slots. Shell-overrideable via `OLLAMA_NUM_PARALLEL=8 docker compose up`. |
| `OLLAMA_MAX_QUEUE` | `1024` | Request queue cap. |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed inside the container. |
| `OLLAMA_SCHED_SPREAD` | `1` | Spread inference work across GPUs. |
| `OLLAMA_LOAD_TIMEOUT` | `30m` | Model load timeout. |

References:

- [Ollama FAQ: KV cache quantisation](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-set-the-quantization-type-for-the-kv-cache)
- [Ollama FAQ: Flash Attention](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-enable-flash-attention)

### num_ctx and Ollama >= 0.15.5

Ollama 0.15.5 introduced tiered default context lengths based on available VRAM:

| VRAM | Default `num_ctx` |
|---|---|
| < 24 GB | 4,096 |
| 24-48 GB | 32,768 |
| >= 48 GB | 262,144 |

On a 48 GB GPU the implicit default is 262,144 -- combined with `OLLAMA_NUM_PARALLEL=16`, the KV cache footprint (`num_ctx * NUM_PARALLEL`) can exhaust VRAM and collapse throughput. Always pass `--num-ctx` explicitly; `4096` is sufficient for typical BioSample NER workloads.

References:

- [Tiered context length can exhaust VRAM (GitHub #14116)](https://github.com/ollama/ollama/issues/14116)
- [New default context lengths will break (GitHub #14073)](https://github.com/ollama/ollama/issues/14073)
