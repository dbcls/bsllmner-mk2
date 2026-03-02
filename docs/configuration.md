# Configuration

Reference for environment variables and configuration values.

## Ollama

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Can also be overridden with the `--ollama-host` CLI option.

## Directories

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_RESULT_DIR` | `$PWD/bsllmner2-results` | Root directory for extract/select result files |
| `BSLLMNER2_TMP_DIR` | `$TMPDIR/bsllmner2-$UID` | Temporary directory for progress files |

## Cache

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BSLLMNER2_INDEX_CACHE_DIR` | `/app/ontology/index_cache` | Ontology index cache directory |

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

References:

- [Ollama FAQ: KV cache](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-set-the-quantization-type-for-the-kv-cache)
- [Ollama FAQ: Flash Attention](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-enable-flash-attention)

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
| `BSLLMNER2_DEBUG` | `false` | Debug mode (enabled with `true`/`1`/`yes`/`on`) |

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
