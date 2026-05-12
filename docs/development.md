# Development

## Entry Points

Defined in `pyproject.toml` under `[project.scripts]`:

| Command | Entry Point |
|---|---|
| `bsllmner2_extract` | `bsllmner2.cli_extract:run_cli_extract` |
| `bsllmner2_select` | `bsllmner2.cli_select:run_cli_select` |

For options, see [CLI Reference](cli.md).

## Local Development Setup

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), Docker (for end-to-end runs against Ollama), and an NVIDIA GPU if you intend to exercise the LLM path.

```bash
uv sync --all-extras           # install runtime + test dependencies
docker compose up -d --build    # bring up app + ollama containers
docker compose exec app bash    # interactive shell inside app container
```

For test commands, lint, type check, and mutation testing, see [Testing](testing.md).

## scripts/ Reference

| File | Purpose |
|---|---|
| `download_ontology_files.py` | Fetch upstream OWL/OBO sources into `ontology/`. |
| `preprocess_cellosaurus.py` | Filter Cellosaurus OBO per NCBI taxonomy ID and synthesise `def:` lines. |
| `ncbi_gene_to_owl.py` | Convert NCBI `gene_info` TSV into per-taxon OWL. |
| `build_subset_ontologies.sh` | Build CL / UBERON / MONDO / ChEBI / PO subset OWLs via SPARQL + ROBOT. |
| `prepare_chipatlas_bs_entries.py` | Build ChIP-Atlas `bs_entries.jsonl` from `experimentList.tab` + DDBJ Bulk API. |
| `collect_rnaseq_biosample.py` | Collect RNA-Seq BioSample entries via the DDBJ Search API. |
| `inspect_select_result.py` | Debug tool for SelectResult JSON files (`summary` / `show` / `find`). |
| `run_model_bench.sh` | Run Select mode against the 600-entry evaluation set across a fixed list of Ollama models (defined inline in the script). Pull failures are recorded as `pull_failed` in `summary.tsv` and the sweep continues. |
| `select-config-hg38.json` | Human (hg38) select config. |
| `select-config-mm10.json` | Mouse (mm10) select config. |
| `select-config-plants.json` | Plant Ontology select config (tissue / cell_type). |

## Debug Tools

### inspect_select_result.py

`scripts/inspect_select_result.py` parses a SelectResult JSON file and exposes three subcommands. All emit human-readable text by default and JSON with `--json`.

```bash
# Run-wide overview: mapping rate per field, NOT_FOUND top values,
# LLM timing, evaluation metrics.
uv run python scripts/inspect_select_result.py summary \
  bsllmner2-results/select/select_<run>.json

# Adjust how many NOT_FOUND values to print per field (default 10).
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

- `[exact]` -- an exact match was found during ontology search.
- `[llm]` -- the LLM selected the term from multiple candidates.
- `[text2term]` -- text2term similarity picked the top candidate.

## Release Process

Versions are managed via git tags using [hatch-vcs](https://github.com/ofek/hatch-vcs); no manual edit to `pyproject.toml` is required.

```bash
git tag X.Y.Z
git push origin X.Y.Z
```

The tag push triggers `.github/workflows/release.yml`, which builds and publishes the Docker image to `ghcr.io/dbcls/bsllmner-mk2` and creates a GitHub Release with auto-generated notes.
