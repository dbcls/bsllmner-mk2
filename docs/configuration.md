# Configuration

Reference for environment variables and configuration values.

## Ollama

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Can also be overridden with the `--ollama-host` CLI option.

## CLI

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_DEBUG` | `false` | Debug mode (enabled with `true`/`1`/`yes`/`on`) |

Can also be enabled with the `--debug` CLI option.

## Directories

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_RESULT_DIR` | `$PWD/bsllmner2-results` | Root directory for extract/select result files |
| `BSLLMNER2_TMP_DIR` | `$TMPDIR/bsllmner2-$UID` | Temporary directory for progress files |

## Cache

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_INDEX_CACHE_DIR` | `ontology/index_cache` | Ontology index cache directory |

## Metrics

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_CONTAINER_NAME` | `bsllmner-mk2-ollama` | Docker container name for metrics collection |

## Ollama Performance Tuning (Docker Compose)

Configured in the ollama service of `compose.yml` / `compose.front.yml`.

| Environment Variable | Value | Description |
|---------------------|-------|-------------|
| `OLLAMA_HOST` | `0.0.0.0:11434` | In-container bind address |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV cache quantization type |
| `OLLAMA_FLASH_ATTENTION` | `1` | Enable Flash Attention |
| `OLLAMA_NUM_PARALLEL` | `16` | Parallel inference threads |
| `OLLAMA_MAX_QUEUE` | `1024` | Maximum queue size |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPU devices to use |
| `OLLAMA_SCHED_SPREAD` | `1` | Spread inference load across GPUs |

### `num_ctx` and Ollama >= 0.15.5

Ollama 0.15.5 introduced tiered default context lengths based on available VRAM:

| VRAM | Default `num_ctx` |
|------|-------------------|
| < 24 GB | 4,096 |
| 24-48 GB | 32,768 |
| >= 48 GB | 262,144 |

When `num_ctx` is not explicitly specified, Ollama auto-selects a value from the table above. On high-VRAM GPUs (e.g., RTX 6000 Ada with 48 GB), this results in a very large context length. Combined with `OLLAMA_NUM_PARALLEL`, the KV cache allocation (`num_ctx * NUM_PARALLEL`) can exhaust VRAM and severely degrade throughput.

**Always specify `--num-ctx` explicitly** to avoid this issue. A value of 4096 is sufficient for typical BioSample NER workloads.

References:

- [Ollama FAQ: KV cache](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-set-the-quantization-type-for-the-kv-cache)
- [Ollama FAQ: Flash Attention](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-enable-flash-attention)
- [Tiered context length can exhaust VRAM (GitHub #14116)](https://github.com/ollama/ollama/issues/14116)
- [New default context lengths will break (GitHub #14073)](https://github.com/ollama/ollama/issues/14073)

## Slurm Configuration

`init-slurm.sh` generates a Slurm job script. For details, see [NIG Slurm - Generate slurm.sh](nig-slurm.md#2-generate-slurmsh).

## Deprecated

> The Backend API and Frontend are not actively maintained. The environment variables below are kept for reference only and may be removed in a future release.

### Backend API Environment Variables

Loaded by `get_config()` in `bsllmner2/config.py`.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_API_HOST` | `127.0.0.1` | API server bind address |
| `BSLLMNER2_API_PORT` | `8000` | API server port |
| `BSLLMNER2_API_URL_PREFIX` | `""` (empty) | FastAPI `root_path` (for reverse proxy) |

### Frontend Environment Variables

Referenced in `front/vite.config.ts`. Embedded at build time via Vite's `define`.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_FRONT_HOST` | `0.0.0.0` | Dev server bind address |
| `BSLLMNER2_FRONT_PORT` | `3000` | Dev server port |
| `BSLLMNER2_FRONT_EXTERNAL_URL` | `http://localhost:3000` | Frontend URL as seen by the browser |
| `BSLLMNER2_API_INTERNAL_URL` | `http://bsllmner-mk2-api:8000` | Inter-container API URL (for Vite proxy) |
| `BSLLMNER2_OLLAMA_URL` | `http://bsllmner-mk2-ollama:11434` | Inter-container Ollama URL (for Vite proxy) |
| `BSLLMNER2_FRONT_BASE` | `/` | Vite base path |

#### Vite Proxy Configuration

The dev server proxies the following paths to internal URLs:

| Path | Forwarded To |
|------|-------------|
| `/api` | `BSLLMNER2_API_INTERNAL_URL` |
| `/ollama` | `BSLLMNER2_OLLAMA_URL` |
