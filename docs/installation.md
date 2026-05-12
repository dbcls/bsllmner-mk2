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

`compose.yml` reserves every visible NVIDIA GPU on the host (`deploy.resources.reservations.devices.count: all`) and the ollama service narrows the active set via `CUDA_VISIBLE_DEVICES`. The committed default (`0,1`) targets the DBCLS GPU server (`dbcls-ai01`, RTX 6000 Ada × 2). On a single-GPU host you should edit `compose.yml`:

```yaml
# compose.yml (ollama service)
environment:
  - CUDA_VISIBLE_DEVICES=0  # use only GPU 0
```

If you need to run on a host with a different topology, change both the `count` field under `deploy` and the `CUDA_VISIBLE_DEVICES` value to match.

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
```

For uv (local):

```bash
# Check the CLI is available
uv run bsllmner2_extract --help
```
