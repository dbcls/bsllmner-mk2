# Getting Started

This guide walks you through running bsllmner-mk2 for the first time.

For installation details (Docker Compose, uv, GPU configuration), see [Installation](installation.md).

## 1. Start the Service

```bash
docker compose up -d --build
```

If this is your first time, complete the full setup in [Installation](installation.md) first.

## 2. Download Ontology Files

The ontology files are used by Select mode (Stage 2) for mapping extracted terms to ontology entries. The download script places files inside the container at `/app/ontology/`. Because `compose.yml` mounts `${PWD}:/app`, they are also available on the host at `./ontology/`.

### 2.1 Run Download Script

```bash
docker compose exec app python3 scripts/download_ontology_files.py
```

This downloads the following ontology files to `ontology/`:

- `cellosaurus.obo` - Cell line database
- `cell_ontology.owl` - Cell Ontology
- `uberon.owl` - UBERON (anatomy ontology)
- `mondo.owl` - MONDO (disease ontology)
- `chebi.owl` - ChEBI (chemical entities)

### 2.2 Convert Cellosaurus OBO to OWL

Cellosaurus is downloaded in OBO format and needs to be converted to OWL:

```bash
cd ontology
docker run -v $PWD:/work -w /work --rm -it obolibrary/robot robot convert \
  -i ./cellosaurus.obo \
  -o ./cellosaurus.owl \
  --format owl
cd ..
```

## 3. Pull LLM Model

Download the LLM model to the Ollama server:

```bash
docker compose exec ollama ollama pull llama3.1:70b
```

This may take several minutes depending on your network speed.
The model (~40GB for 70b) is stored in the `ollama-data/` directory.

## 4. Run Extract Mode

Extract mode performs Named Entity Recognition (NER) to extract biological terms from BioSample records.

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --debug
```

For all extract CLI options, see [Extract Mode](extract-mode.md#cli-options).

## 5. Run Select Mode

Select mode extends extract mode by mapping extracted terms to ontology entries.

```bash
docker compose exec app bsllmner2_select \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --select-config scripts/select-config.json \
  --debug
```

For all select CLI options, see [Select Mode](select-mode.md#cli-options).

## 6. Inspect Results

Results are saved in `bsllmner2-results/`:

```bash
# List result files
ls bsllmner2-results/extract/
ls bsllmner2-results/select/
```

View the extract result:

```bash
# Show run metadata
jq '.run_metadata' bsllmner2-results/extract/*.json

# Show extracted values
jq '.output[] | {accession, output}' bsllmner2-results/extract/*.json
```

View the select result:

```bash
# Show mapped ontology terms
jq '.[0].results' bsllmner2-results/select/*.json
```

### Result Structure

- **Extract results**: `bsllmner2-results/extract/{run_name}.json`
  - Contains extracted entities, evaluation metrics (if mapping provided), and metadata

- **Select results**: `bsllmner2-results/select/select_{run_name}.json`
  - Contains ontology-mapped results for each field

For the full result schema, see [Data Formats](data-formats.md).

## Next Steps

- **ChIP-Atlas data processing**: See [chip-atlas.md](chip-atlas.md) for processing ChIP-Atlas data with hg38/mm10
- **Model evaluation**: See [tests/model-evaluation/README.md](../tests/model-evaluation/README.md) for benchmarking different LLM models
- **Custom configuration**: Create your own `select-config.json` to customize field extraction and ontology mapping
