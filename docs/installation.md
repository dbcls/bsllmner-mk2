# Installation

## Docker Compose (Recommended)

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for faster inference)
- At least 40GB disk space for LLM model storage

### Setup

```bash
git clone https://github.com/dbcls/bsllmner-mk2.git
cd bsllmner-mk2

# Build and start containers
docker compose up -d --build
```

### GPU Configuration

The `compose.yml` reserves all available NVIDIA GPUs by default. To restrict which GPUs are used, edit the `CUDA_VISIBLE_DEVICES` environment variable:

```yaml
# compose.yml (ollama service)
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
```

For Ollama performance tuning options, see [Configuration - Ollama Performance Tuning](configuration.md#ollama-performance-tuning-docker-compose).

## uv (Local Development)

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Ollama server running locally or remotely

### Setup

```bash
uv sync

# Install with test/development dependencies
uv sync --all-extras
```

If the Ollama server is running on a different host, set the `OLLAMA_HOST` environment variable:

```bash
export OLLAMA_HOST=http://<ollama-host>:11434
```

## Verify Installation

For Docker Compose:

```bash
# Check containers are running
docker compose ps

# Verify Ollama is accessible
docker compose exec ollama ollama list
```

For uv (local):

```bash
# Check the CLI is available
uv run bsllmner2_extract --help
```
