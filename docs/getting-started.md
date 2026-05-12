# Getting Started

This guide takes you from a fresh clone to your first Extract and Select runs.

For setup details (Docker Compose, uv, host requirements, GPU configuration), see [Installation](installation.md). For per-option reference, see [CLI](cli.md).

## Prerequisites

- Docker / docker compose with the NVIDIA Container Toolkit. Verify GPU visibility with `docker compose exec ollama nvidia-smi`.
- A writable `ontology/` directory at the repo root (bind-mounted into both containers; created automatically).
- Additional host packages and Docker images required by the ontology pipeline; see [Installation](installation.md#system-requirements).
- An LLM model. Pre-pull or rely on Ollama's auto-pull at first use; see [Step 3](#3-optional-pre-pull-llm-model).

## 1. Start the Service

```bash
docker compose up -d --build
```

The `app` and `ollama` containers come up under the `bsllmner-mk2-network` bridge. `app` mounts the project root at `/app`, so `git pull` on the host is immediately visible inside the container.

## 2. Prepare Ontology Files

Select mode reads pre-built OWL files from `ontology/`. Follow [Ontology Preparation](ontology.md) end-to-end on the first setup; subsequent runs reuse the generated files. The full pipeline:

1. Download upstream OWL/OBO sources.
2. Preprocess Cellosaurus per species and convert to OWL.
3. Build CL / UBERON / MONDO / ChEBI / PO subset OWLs.
4. Generate per-species NCBI Gene OWLs.

Extract mode does not need any ontology files; you can skip this step if you only run `bsllmner2_extract`.

## 3. (Optional) Pre-pull LLM Model

Ollama auto-pulls the model on first use. For large models (e.g., 70B ~ 40 GB), pre-pull to keep the first run from blocking on the download:

```bash
docker compose exec ollama ollama pull llama3.1:70b
```

Models live in the `ollama-data` Docker volume.

## 4. Run Extract Mode

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --debug
```

Output: `bsllmner2-results/extract/{run_name}.json`. See [Extract Mode](extract-mode.md) for the pipeline and [CLI](cli.md#bsllmner2_extract) for every option.

## 5. Run Select Mode

```bash
docker compose exec app bsllmner2_select \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --select-config scripts/select-config-hg38.json \
  --debug
```

Output: `bsllmner2-results/select/select_{run_name}.json`. See [Select Mode](select-mode.md) for the three-stage pipeline and [CLI](cli.md#bsllmner2_select) for every option.

## 6. Inspect Results

```bash
ls bsllmner2-results/extract/
ls bsllmner2-results/select/

# Run-wide summary (mapping rate, NOT_FOUND, LLM timing)
docker compose exec app python3 scripts/inspect_select_result.py summary bsllmner2-results/select/select_<run-name>.json

# Per-entry detail (SAMD00123367 is the first entry in tests/data/example_biosample.json)
docker compose exec app python3 scripts/inspect_select_result.py show \
  bsllmner2-results/select/select_<run-name>.json --accession SAMD00123367
```

Result schemas are defined in [Data Formats](data-formats.md). See [Development](development.md#inspect_select_resultpy) for additional `inspect_select_result.py` subcommands.

## Next Steps

- [ChIP-Atlas](chip-atlas.md) -- process ChIP-Atlas data with the hg38 / mm10 select configs.
- [Benchmarking](benchmarking.md) -- read performance metrics in the result JSON.
- [Select Mode](select-mode.md#selectconfig-customization) -- author a custom `select-config.json`.
