# bsllmner-mk2

A CLI tool that extracts biological named entities from [BioSample](https://www.ncbi.nlm.nih.gov/biosample/) records with LLMs ([Ollama](https://ollama.com/)) and maps them to ontology terms.

**Documentation:** <https://dbcls.github.io/bsllmner-mk2>

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

A complete walkthrough -- including ontology preparation and Select mode -- is in [Getting Started](https://dbcls.github.io/bsllmner-mk2/getting-started/).

## Documentation

**Basics**

- [Getting Started](https://dbcls.github.io/bsllmner-mk2/getting-started/) -- First-run walkthrough.
- [Installation](https://dbcls.github.io/bsllmner-mk2/installation/) -- Docker Compose, uv, host requirements.

**Modes**

- [Extract Mode](https://dbcls.github.io/bsllmner-mk2/extract-mode/) -- NER pipeline.
- [Select Mode](https://dbcls.github.io/bsllmner-mk2/select-mode/) -- NER + ontology mapping pipeline.

**Reference**

- [CLI](https://dbcls.github.io/bsllmner-mk2/cli/) -- `bsllmner2_extract` / `bsllmner2_select` options.
- [Data Formats](https://dbcls.github.io/bsllmner-mk2/data-formats/) -- Input/output schemas.
- [Configuration](https://dbcls.github.io/bsllmner-mk2/configuration/) -- Environment variables and Ollama tuning.

**Operations**

- [Ontology Preparation](https://dbcls.github.io/bsllmner-mk2/ontology/) -- Building the OWL files Select mode consumes.
- [ChIP-Atlas](https://dbcls.github.io/bsllmner-mk2/chip-atlas/) -- Processing ChIP-Atlas data (hg38 / mm10).
- [NIG Slurm](https://dbcls.github.io/bsllmner-mk2/nig-slurm/) -- Running on the NIG Slurm environment.

**Contributing**

- [Development](https://dbcls.github.io/bsllmner-mk2/development/) -- Local development setup.
- [Testing](https://dbcls.github.io/bsllmner-mk2/testing/) -- pytest, mypy, ruff, mutmut, model evaluation.
- [Benchmarking](https://dbcls.github.io/bsllmner-mk2/benchmarking/) -- Reading performance and accuracy data.

## Related Resources

- Original repository: [sh-ikeda/bsllmner](https://github.com/sh-ikeda/bsllmner)
- Related paper: <https://doi.org/10.1101/2025.02.17.638570>

## License

Released under the MIT License. See [LICENSE](https://github.com/dbcls/bsllmner-mk2/blob/main/LICENSE).
