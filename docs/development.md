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

## Local Development Setup

### Python Package

```bash
uv sync --all-extras
```

### Docker Environment

```bash
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

## Release Process

Version is managed via git tags using [hatch-vcs](https://github.com/ofek/hatch-vcs). No manual version editing in `pyproject.toml` is required.

1. Merge PR to `main`
2. Create and push a version tag:
   ```bash
   git tag X.Y.Z
   git push origin X.Y.Z
   ```
3. The tag push triggers `.github/workflows/release.yml`:
   - Build and push Docker image to `ghcr.io/dbcls/bsllmner-mk2`
   - Create GitHub Release with auto-generated notes

## scripts/ Utility Scripts

| Script | Description |
|--------|-------------|
| `download_ontology_files.py` | Download ontology files |
| `ncbi_gene_to_owl.py` | Convert NCBI Gene data to OWL |
| `prepare_bs_entries.py` | Prepare BioSample entries for ChIP-Atlas |
| `inspect_select_result.py` | Inspect a SelectResult JSON (summary / show / find subcommands) |
| `select-config-hg38.json` | Human (hg38) select config |
| `select-config-mm10.json` | Mouse (mm10) select config |
| `select-config-plants.json` | Plants (PO) select config (tissue / cell_type) |

## Debug tools

### `inspect_select_result.py`

Inspect a `SelectResult` JSON produced by `bsllmner2_select`. All three
subcommands emit plain text by default and JSON with `--json`.

```bash
# Run-wide overview: mapping rate per field, NOT_FOUND top values,
# LLM timing, evaluation metrics.
uv run python scripts/inspect_select_result.py summary \
  bsllmner2-results/select/select_<run>.json

# Override how many NOT_FOUND values to print per field (default 10).
uv run python scripts/inspect_select_result.py summary \
  bsllmner2-results/select/select_<run>.json --top-nf 30

# Entry-level detail for a specific BioSample accession.
uv run python scripts/inspect_select_result.py show \
  bsllmner2-results/select/select_<run>.json --accession SAMN00000001

# Only entries that contain at least one unmapped value.
uv run python scripts/inspect_select_result.py show \
  bsllmner2-results/select/select_<run>.json --unmapped-only --limit 20

# Locate every entry that extracted a particular (field, value) pair.
uv run python scripts/inspect_select_result.py find \
  bsllmner2-results/select/select_<run>.json --field cell_line --value HeLa
```

`show` and `find` tag each resolved value with its source:

- `[exact]` - an exact match was found during ontology search
- `[llm]` - the LLM selected the term from multiple candidates
- `[text2term]` - text2term similarity picked the top candidate
