# bsllmner-mk2

A CLI tool that extracts biological named entities from [BioSample](https://www.ncbi.nlm.nih.gov/biosample/) records with LLMs ([Ollama](https://ollama.com/)) and maps them to ontology terms.

## Capabilities

- **Extract mode** (`bsllmner2_extract`) -- Performs Named Entity Recognition (NER) over BioSample metadata and emits structured JSON.
- **Select mode** (`bsllmner2_select`) -- Runs the same NER pass, searches each extracted value against ontologies (Cellosaurus, Cell Ontology, UBERON, MONDO, ChEBI, NCBI Gene, Plant Ontology), and lets the LLM pick the best ontology term per field.

## Quick Start

```bash
docker compose up -d --build
docker compose exec app bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b --debug
```

A complete walkthrough -- including ontology preparation and Select mode -- is in [Getting Started](getting-started.md).

## Documentation Map

**Basics**

- [Getting Started](getting-started.md) -- First-run walkthrough
- [Installation](installation.md) -- Docker Compose, uv, and host prerequisites

**Modes**

- [Extract Mode](extract-mode.md) -- NER extraction pipeline
- [Select Mode](select-mode.md) -- NER + ontology mapping pipeline

**Reference**

- [CLI](cli.md) -- `bsllmner2_extract` / `bsllmner2_select` options
- [Data Formats](data-formats.md) -- Input/output schemas
- [Configuration](configuration.md) -- Environment variables and Ollama tuning

**Operations**

- [Ontology Preparation](ontology.md) -- Building the OWL files Select mode consumes
- [ChIP-Atlas](chip-atlas.md) -- Processing ChIP-Atlas data (hg38 / mm10)
- [NIG Slurm](nig-slurm.md) -- Running on the NIG Slurm environment

**Contributing**

- [Development](development.md) -- Local development setup
- [Testing](testing.md) -- pytest, mypy, ruff, mutmut, model evaluation
- [Benchmarking](benchmarking.md) -- Reading performance and accuracy data

## License

Released under the MIT License. See [LICENSE](https://github.com/dbcls/bsllmner-mk2/blob/main/LICENSE).
