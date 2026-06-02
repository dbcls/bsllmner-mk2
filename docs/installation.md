# Installation

## System Requirements

| Component | Why it is needed |
|---|---|
| Docker + docker compose | Runs the `app` and `ollama` services defined in `compose.yml`. |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Exposes host GPUs to the `ollama` container. |
| At least 40 GB free disk | LLM weights (e.g. ~40 GB for the 70 B Llama). |
| [`obolibrary/robot:latest`](https://github.com/ontodev/robot) Docker image | Required by [Ontology Preparation](ontology.md) (auto-pulled on first use). |
| `gawk` on the host | Used by the Plant Ontology preprocess step in `scripts/build_subset_ontologies.sh`. The ROBOT image ships `mawk`, which lacks `gensub()`. Install with `apt-get install gawk` (Debian/Ubuntu) or `brew install gawk` (macOS). |
| Python 3.10+ and [uv](https://docs.astral.sh/uv/) | Only for local (non-Docker) development. |

## Docker Compose (Recommended)

### Setup

```bash
git clone https://github.com/dbcls/bsllmner-mk2.git
cd bsllmner-mk2
docker compose up -d --build
```

This builds the `app` image, pulls `ollama/ollama:0.17.7`, and starts both containers on the `bsllmner-mk2-network` bridge. Named volumes:

- `bsllmner-mk2_venv` -- the project's `.venv` inside the `app` container.
- `bsllmner-mk2_ollama-data` -- Ollama's model cache (`/root/.ollama`).

The host's project directory is bind-mounted at `/app`, so source edits take effect without rebuilding.

### GPU Configuration

`compose.yml` reserves every visible NVIDIA GPU (`deploy.resources.reservations.devices.count: all`) and the `ollama` service narrows the active set with `CUDA_VISIBLE_DEVICES`. The committed default is `0,1` (two GPUs). On a single-GPU host:

```yaml
# compose.yml (ollama service)
environment:
  - CUDA_VISIBLE_DEVICES=0
```

For Ollama performance tuning (KV cache, flash attention, context length tiers), see [Configuration](configuration.md#ollama-performance-tuning-docker-compose).

## uv (Local Development)

For running tests, type checks, or the CLI directly on the host:

```bash
uv sync --all-extras
```

If the Ollama server is on a different host, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://<ollama-host>:11434
```

## Verify Installation

```bash
# Docker Compose
docker compose ps
docker compose exec app bsllmner2_extract --help

# uv
uv run bsllmner2_extract --help
```
