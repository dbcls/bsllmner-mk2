# Quick Start Guide

This guide walks you through setting up and running bsllmner-mk2 for the first time.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for faster inference)
- At least 40GB disk space for LLM model storage
- Git

## 1. Environment Setup

### 1.1 Clone Repository

```bash
git clone https://github.com/dbcls/bsllmner-mk2.git
cd bsllmner-mk2
```

### 1.2 Start Docker Containers

```bash
# Create directory for Ollama data
mkdir -m 777 ollama-data

# Create Docker network
docker network create bsllmner-mk2-network

# Build and start containers
docker compose up -d --build
```

## 2. Download Ontology Files

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

## 3. Run Extract Mode (Simple Example)

Extract mode performs Named Entity Recognition (NER) to extract biological terms from BioSample records.

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --debug
```

### Extract CLI Options

| Option | Required | Description |
|--------|----------|-------------|
| `--bs-entries` | Yes | Path to input JSON/JSONL file containing BioSample entries |
| `--mapping` | No | Path to mapping file (TSV) for evaluation |
| `--prompt` | No | Path to prompt file (YAML). Default: built-in prompt |
| `--format` | No | Path to JSON schema file for output format |
| `--model` | No | LLM model name. Default: `llama3.1:70b` |
| `--thinking` | No | Enable/disable thinking mode (`true`/`false`) |
| `--max-entries` | No | Process only first N entries. Default: all |
| `--ollama-host` | No | Ollama server URL. Default: `http://localhost:11434` |
| `--with-metrics` | No | Enable metrics collection |
| `--debug` | No | Enable verbose logging |
| `--run-name` | No | Custom name for the run |
| `--resume` | No | Resume from last incomplete run |
| `--batch-size` | No | Entries per batch. Default: 1024 |

## 4. Run Select Mode (Simple Example)

Select mode extends extract mode by mapping extracted terms to ontology entries.

```bash
docker compose exec app bsllmner2_select \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --select-config scripts/select-config.json \
  --debug
```

### Select CLI Options

In addition to the extract options above, select mode has:

| Option | Required | Description |
|--------|----------|-------------|
| `--select-config` | Yes | Path to select configuration file (JSON) |
| `--no-reasoning` | No | Disable reasoning step during selection |

## 5. Check Results

Results are saved in `bsllmner2-results/`:

```bash
# List extract results
ls bsllmner2-results/extract/

# List select results
ls bsllmner2-results/select/
```

### Result Structure

- **Extract results**: `bsllmner2-results/extract/{run_name}.json`
  - Contains extracted entities, evaluation metrics (if mapping provided), and metadata

- **Select results**: `bsllmner2-results/select/select_{run_name}.json`
  - Contains ontology-mapped results for each field

## 6. Frontend (Optional)

For a web-based interface, use the frontend compose file:

```bash
# Start frontend environment
docker compose -f compose.front.yml up -d

# Access the UI at http://localhost:3000
```

The frontend provides:

- File upload interface for BioSample data
- Real-time processing status
- Result visualization

## Next Steps

- **ChIP-Atlas data processing**: See [chip-atlas.md](chip-atlas.md) for processing ChIP-Atlas data with hg38/mm10
- **Model evaluation**: See [tests/model-evaluation/README.md](../tests/model-evaluation/README.md) for benchmarking different LLM models
- **Custom configuration**: Create your own `select-config.json` to customize field extraction and ontology mapping
