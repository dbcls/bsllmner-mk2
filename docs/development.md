# Development

## Prerequisites

- Python 3.10+
- Docker, Docker Compose
- NVIDIA GPU + CUDA driver (for Ollama)
- Node.js 24+ (frontend development only)

## CLI Entry Points

Defined in `pyproject.toml` under `[project.scripts]`:

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `bsllmner2_extract` | `bsllmner2.cli_extract:run_cli_extract` | Extract mode CLI |
| `bsllmner2_select` | `bsllmner2.cli_select:run_cli_select` | Select mode CLI |
| `bsllmner2_api` | `bsllmner2.api:run_api` | FastAPI server |
| `bsllmner2_metrics` | `bsllmner2.metrics:main` | Metrics computation |

## Local Development Setup

### Python Package

```bash
uv sync --all-extras
```

### Docker Environment

```bash
mkdir -m 777 ollama-data
docker network create bsllmner-mk2-network
docker compose up -d --build
docker compose exec app bash
```

## Running Tests

```bash
# Tests
uv run pytest

# Type checking
uv run mypy

# Linter
uv run ruff check bsllmner2/ tests/ scripts/

# Formatter
uv run ruff format bsllmner2/ tests/ scripts/

# Format check (for CI)
uv run ruff format --check bsllmner2/ tests/ scripts/
```

For details on test structure, mutation testing, and model evaluation, see [Testing](testing.md).

## scripts/ Utility Scripts

| Script | Description |
|--------|-------------|
| `download_ontology_files.py` | Download ontology files |
| `ncbi_gene_to_owl.py` | Convert NCBI Gene data to OWL |
| `prepare_bs_entries.py` | Prepare BioSample entries for ChIP-Atlas |
| `print_select_result.py` | Display and analyze select results |
| `list_unmapped.py` | List unmapped entries |
| `select-config.json` | General select config (7 fields) |
| `select-config-hg38.json` | Human (hg38) select config |
| `select-config-mm10.json` | Mouse (mm10) select config |
