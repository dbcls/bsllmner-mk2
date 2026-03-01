# bsllmner-mk2

A tool for extracting biological named entities from [BioSample](https://www.ncbi.nlm.nih.gov/biosample/) records using Large Language Models (LLMs) and mapping them to ontology terms.

**Key capabilities:**

- **Extract mode** -- Performs Named Entity Recognition (NER) to extract terms such as cell line, tissue, and organism from BioSample metadata
- **Select mode** -- Extends extract mode by mapping extracted terms to ontology entries (Cellosaurus, UBERON, Cell Ontology, etc.)

bsllmner-mk2 uses [Ollama](https://ollama.com/) as the LLM inference server.

## Quick Start

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:70b
docker compose exec app bsllmner2_extract \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b --debug
```

For a complete walkthrough including ontology setup and Select mode, see [Getting Started](docs/getting-started.md).

## Documentation

**Basics**

- [Getting Started](docs/getting-started.md) -- First run walkthrough with ontology setup
- [Installation](docs/installation.md) -- Docker Compose, uv, and GPU configuration

**Features**

- [Extract Mode](docs/extract-mode.md) -- NER extraction pipeline and CLI options
- [Select Mode](docs/select-mode.md) -- Ontology mapping pipeline and CLI options
- [Data Formats](docs/data-formats.md) -- Input/output data format specification
- [Configuration](docs/configuration.md) -- Environment variables and settings

**Operations**

- [ChIP-Atlas](docs/chip-atlas.md) -- Processing ChIP-Atlas data (hg38/mm10)
- [NIG Slurm](docs/nig-slurm.md) -- Running on NIG Slurm environment

**Development**

- [Development](docs/development.md) -- Development environment setup
- [Testing](docs/testing.md) -- Unit tests, linting, mutation testing, model evaluation

## Related Resources

- Original repository: [sh-ikeda/bsllmner](https://github.com/sh-ikeda/bsllmner)
- Related paper: [https://doi.org/10.1101/2025.02.17.638570](https://doi.org/10.1101/2025.02.17.638570)

## Other Interfaces

bsllmner-mk2 also includes a FastAPI server (`bsllmner2_api`) and a React-based web UI, but these are not actively maintained and their operation is unverified.

## License

This repository is released under the MIT License.
For details, see the [LICENSE](./LICENSE) file.
