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

A complete walkthrough -- including ontology preparation and Select mode -- is in [Getting Started](docs/getting-started.md).

## Documentation

**Basics**

- [Getting Started](docs/getting-started.md) -- First-run walkthrough.
- [Installation](docs/installation.md) -- Docker Compose, uv, host requirements.

**Modes**

- [Extract Mode](docs/extract-mode.md) -- NER pipeline.
- [Select Mode](docs/select-mode.md) -- NER + ontology mapping pipeline.

**Reference**

- [CLI](docs/cli.md) -- `bsllmner2_extract` / `bsllmner2_select` options.
- [Data Formats](docs/data-formats.md) -- Input/output schemas.
- [Configuration](docs/configuration.md) -- Environment variables and Ollama tuning.

**Operations**

- [Ontology Preparation](docs/ontology.md) -- Building the OWL files Select mode consumes.
- [ChIP-Atlas](docs/chip-atlas.md) -- Processing ChIP-Atlas data (hg38 / mm10).
- [NIG Slurm](docs/nig-slurm.md) -- Running on the NIG Slurm environment.

**Contributing**

- [Development](docs/development.md) -- Local development setup.
- [Testing](docs/testing.md) -- pytest, mypy, ruff, mutmut, model evaluation.
- [Benchmarking](docs/benchmarking.md) -- Reading performance and accuracy data.

## Related Resources

- Original repository: [sh-ikeda/bsllmner](https://github.com/sh-ikeda/bsllmner)
- Related paper: <https://doi.org/10.1101/2025.02.17.638570>

## License

Released under the MIT License. See [LICENSE](./LICENSE).
