# bsllmner-mk2

## Introduction

This repository contains the code for `bsllmner-mk2`, a collaborative project building upon the original `bsllmner` toolkit.
The goal is to use Large Language Models (LLMs) to perform named entity recognition (NER) of biological terms from BioSample records and to select appropriate ontology terms.

In this mk2 version, we have enhanced the original algorithms and pipeline, refactored the codebase into a more robust library, and introduced several new features:

- Improved NER and ontology mapping algorithms
- Parallel processing for large-scale data extraction
- Development of a user-friendly frontend
- Modular library structure for easier integration

`bsllmner-mk2` is developed in collaboration with the authors of the original repository, aiming to provide a more scalable and maintainable solution for biological sample metadata extraction and ontology mapping.

Related resources:

- Original repository: [sh-ikeda/bsllmner](https://github.com/sh-ikeda/bsllmner)
- Related paper: [https://doi.org/10.1101/2025.02.17.638570](https://doi.org/10.1101/2025.02.17.638570)

## Quick Start

For a step-by-step guide to get started quickly, see **[docs/quick-start.md](docs/quick-start.md)**.

## Installation

`bsllmner-mk2` is provided as a Python package and CLI tool.
To install it locally using pip, run the following command:

```bash
python3 -m pip install .
```

### Recommended Setup

`bsllmner-mk2` is designed to work in combination with an [Ollama](https://ollama.com/).
The recommended way to launch the environment is using Docker, which will start both the Python container (with `bsllmner-mk2` installed) and the Ollama server container.

To start the environment:

```sh
mkdir -m 777 ollama-data
docker network create bsllmner-mk2-network
docker compose up -d --build
```

After starting, you can run CLI commands such as:

```sh
docker compose exec app bsllmner2_extract --help
```

### Model Preparation

Before using bsllmner-mk2, you need to pull the required LLM model with Ollama.
After starting the containers, run the following command to download a model (e.g., `llama3.1:70b`):

```sh
docker compose exec ollama ollama pull llama3.1:70b
```

You can browse available models and their names at [Ollama Model Library](https://ollama.com/library).
Replace `llama3.1:70b` with the model name you want to use.
Make sure the model is fully downloaded before running extraction or API commands.

### Custom Ollama Server

You can also specify a custom Ollama server host using CLI options.
Details about this option are described in a later section.

## CLI Usage

The main CLI tool provided by bsllmner-mk2 is `bsllmner2_extract`.
It performs Named Entity Recognition (NER) of biological terms in BioSample records using LLMs.

To see all available options, run:

```sh
bsllmner2_extract --help
usage: bsllmner2_extract [-h] --bs-entries BS_ENTRIES [--mapping MAPPING]
                         [--prompt PROMPT] [--format FORMAT] [--model MODEL]
                         [--thinking {true,false}] [--max-entries MAX_ENTRIES]
                         [--ollama-host OLLAMA_HOST] [--with-metrics] [--debug]
                         [--run-name RUN_NAME] [--resume] [--batch-size BATCH_SIZE]

Named Entity Recognition (NER) of biological terms in BioSample records using
LLMs, developed as bsllmner-mk2.

options:
  -h, --help            show this help message and exit
  --bs-entries BS_ENTRIES
                        Path to the input JSON or JSONL file containing BioSample
                        entries.
  --mapping MAPPING     Path to the mapping file in TSV format.
  --prompt PROMPT       Path to the prompt file in YAML format. Default is
                        'prompt/prompt_extract.yml' relative to the project root.
  --format FORMAT       Path to the JSON schema file for the output format.
  --model MODEL         LLM model to use for NER.
  --thinking {true,false}
                        Enable or disable thinking mode for the LLM. Use 'true'
                        to enable thinking, 'false' to disable it.
  --max-entries MAX_ENTRIES
                        Process only the first N entries from the input file.
                        Default is -1, which means all entries will be processed.
  --ollama-host OLLAMA_HOST
                        Host URL for the Ollama server (default:
                        http://localhost:11434)
  --with-metrics        Enable collection of metrics during processing.
  --debug               Enable debug mode for more verbose logging.
  --run-name RUN_NAME   Name of the run for identification purposes.
  --resume              Resume from the last incomplete run if possible.
  --batch-size BATCH_SIZE
                        Number of entries to process in each batch (default: 1024).
```

### Main Options

- `--bs-entries` (Required)
  - Path to the input JSON or JSONL file containing BioSample entries.
  - Examples:
    - [`tests/test-data/cell_line_example.biosample.json`](./tests/test-data/cell_line_example.biosample.json) (small dataset)
    - [`tests/zenodo-data/biosample_cellosaurus_mapping_testset.json`](./tests/zenodo-data/biosample_cellosaurus_mapping_testset.json) (large dataset)
- `--mapping` (Optional)
  - Path to the mapping file in TSV format for evaluation.
  - If provided, evaluation metrics (precision, recall, F1) will be calculated.
  - Examples:
    - [`tests/test-data/cell_line_example.mapping.tsv`](./tests/test-data/cell_line_example.mapping.tsv) (small dataset)
    - [`tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv`](./tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv) (large dataset)
- `--prompt`
  - Path to the prompt file in YAML format.
  - Default:
    - [`bsllmner2/prompt/prompt_extract.yml`](./bsllmner2/prompt/prompt_extract.yml)
- `--format`
  - Path to the JSON schema file for the output format.
  - Examples:
    - [`bsllmner2/format/cell_line.schema.json`](./bsllmner2/format/cell_line.schema.json)
- `--run-name`
  - Custom name for the run. Used for result file naming and resume functionality.
  - Default: `{model}_{timestamp}`
- `--resume`
  - Resume processing from the last incomplete run.
  - Requires `--run-name` to match a previous run.
- `--batch-size`
  - Number of entries to process in each batch.
  - Default: 1024
  - Reduce this value if you encounter memory issues.

### Example Command

Here is an example command to run NER extraction on a small dataset:

```sh
$ docker compose exec app bsllmner2_extract \
  --debug \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --mapping tests/test-data/cell_line_example.mapping.tsv \
  --prompt bsllmner2/prompt/prompt_extract.yml \
  --format bsllmner2/format/cell_line.schema.json \
  --model llama3.1:70b \
  --with-metrics

2025-09-02 12:53:51 - bsllmner2 - INFO - Starting bsllmner2 CLI extract mode...
2025-09-02 12:53:51 - bsllmner2 - DEBUG - Config:
{
  "ollama_host": "http://bsllmner-mk2-ollama:11434",
  "debug": true,
  "api_host": "0.0.0.0",
  "api_port": 8000,
  "api_url_prefix": ""
}
2025-09-02 12:53:51 - bsllmner2 - DEBUG - Args:
{
  "bs_entries": "/app/tests/test-data/cell_line_example.biosample.json",
  "mapping": "/app/tests/test-data/cell_line_example.mapping.tsv",
  "prompt": "/app/bsllmner2/prompt/prompt_extract.yml",
  "format": "/app/bsllmner2/format/cell_line.schema.json",
  "model": "llama3.1:70b",
  "thinking": null,
  "max_entries": null,
  "with_metrics": true
}
2025-09-02 12:53:51 - bsllmner2 - DEBUG - Processing entry: SAMD00123367
...(omitted for brevity)

2025-09-02 12:54:11 - bsllmner2 - INFO - Processing complete. Result saved to /app/bsllmner2-results/llama3.1:70b_20250902_125351.json
```

The results will be saved in JSON format under the `./bsllmner2-results` directory.
The structure of the output JSON follows the `Result` schema defined in [`bsllmner2/schema.py`](./bsllmner2/schema.py).

## Frontend

The frontend environment consists of:

- A REST API server (based on `bsllmner-mk2`)
- A React-based frontend UI

You can launch both the API server and the frontend using Docker with the provided [`compose.front.yml`](./compose.front.yml) file.

```sh
mkdir -m 777 ollama-data
docker network create bsllmner-mk2-network
docker compose -f compose.front.yml up -d
```

**How it works:**

- The frontend server acts as a proxy for both the API and Ollama server.
- Only port `3000` is exposed; all communication with the API and Ollama happens internally.
- To access the UI, open your browser and navigate to [http://localhost:3000](http://localhost:3000).
- If you are running in a remote environment, ensure port forwarding for `3000` is set up.

This setup allows you to interact with `bsllmner-mk2` via a user-friendly web interface, while all backend processing and LLM inference are handled within the Docker network.

## Ollama server Behavior and Configuration

The behavior and configuration of the Ollama server are defined within each Docker Compose file (`compose.yml`, `compose.front.yml`).

Several environment variables are set to optimize LLM inference and resource usage:

- `OLLAMA_KV_CACHE_TYPE`:
  - The type of key-value cache used for model inference.
  - Ref.: [GitHub - ollama/ollama - FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-set-the-quantization-type-for-the-kv-cache)
- `OLLAMA_FLASH_ATTENTION`:
  - Enables or disables flash attention optimization.
  - Ref.: [GitHub - ollama/ollama - FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-enable-flash-attention)
- `OLLAMA_NUM_PARALLEL`:
  - Sets the number of parallel inference threads.
  - **Note:** The optimal value depends on your GPU performance and available memory. If you are using a large model and encounter out-of-memory errors, try reducing this value.
- `OLLAMA_MAX_QUEUE`:
  - Defines the maximum number of requests that can be queued.
  - **Note:** Actual queuing is handled internally by the `bsllmner-mk2` library, so this parameter is mostly a formality and rarely needs adjustment.
- `CUDA_VISIBLE_DEVICES`:
  - Specifies which GPU devices are available to Ollama.
  - **Note:** In our development environment, we used two GPUs (`0,1`). Please set this value according to your own hardware configuration.
- `OLLAMA_SCHED_SPREAD`:
  - Enables scheduling strategies for spreading inference load across available resources.

All of these settings can be customized by editing the relevant Compose file.  
For more details, refer directly to the configuration in [`compose.yml`](./compose.yml) or [`compose.front.yml`](./compose.front.yml).

## Select

Select mode extends extract mode by mapping extracted terms to ontology entries.

```bash
docker compose exec app bsllmner2_select \
  --debug \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --select-config ./scripts/select-config.json
```

### Select Options

In addition to the common options from extract mode, `bsllmner2_select` has:

- `--select-config` (Required)
  - Path to the select configuration file in JSON format.
  - Defines which fields to extract and their corresponding ontology files.
  - Examples:
    - [`scripts/select-config.json`](./scripts/select-config.json) - Full configuration with gene fields
    - [`scripts/select-config-hg38.json`](./scripts/select-config-hg38.json) - Human (hg38) configuration
    - [`scripts/select-config-mm10.json`](./scripts/select-config-mm10.json) - Mouse (mm10) configuration
- `--no-reasoning`
  - Disable reasoning step during selection.
  - When enabled, the LLM will not provide explanations for its selections.

### Select Configuration

The select configuration file defines which fields to extract and map to ontologies:

```json
{
  "fields": {
    "cell_line": {
      "ontology_file": "/app/ontology/cellosaurus.owl",
      "prompt_description": "Cell line is a group of cells...",
      "ontology_filter": { "hasDbXref": "NCBI_TaxID:9606" }
    },
    "tissue": {
      "ontology_file": "/app/ontology/uberon.owl",
      "prompt_description": "Tissue is a group of cells..."
    }
  }
}
```

For detailed configuration options, see [docs/chip-atlas.md](docs/chip-atlas.md#select-configuration).

## bsllmner-mk2 with ChIP-Atlas

[ChIP-Atlas](https://chip-atlas.org) is a data-mining suite for exploring epigenomic landscapes by fully integrating ChIP-seq, ATAC-seq, and Bisulfite-seq experiments.
Each SRX entry in ChIP-Atlas contains human-curated metadata values.
Since SRX and BioSample entries are in a one-to-one relationship, we can provide BioSample JSON records to `bsllmner-mk2` and obtain LLM-predicted values for comparison against the human-curated values from ChIP-Atlas.

This enables benchmarking and evaluation of LLM-based NER and ontology mapping against expert-curated annotations.

For detailed instructions on processing ChIP-Atlas data with hg38 and mm10 genome assemblies, see **[docs/chip-atlas.md](docs/chip-atlas.md)**.

## Download Ontology Files for Searching

### Using the Download Script

To download all required ontology files, run the following command:

```bash
docker compose exec app python3 scripts/download_ontology_files.py
```

This downloads:

- `cellosaurus.obo` - Cell line database (needs conversion to OWL)
- `cell_ontology.owl` - Cell Ontology
- `uberon.owl` - UBERON (anatomy ontology)
- `mondo.owl` - MONDO (disease ontology)
- `chebi.owl` - ChEBI (chemical entities)

### Converting Cellosaurus OBO to OWL

Cellosaurus is downloaded in OBO format and must be converted to OWL:

```bash
cd ontology
docker run -v $PWD:/work -w /work --rm -it obolibrary/robot robot convert \
  -i ./cellosaurus.obo \
  -o ./cellosaurus.owl \
  --format owl
cd ..
```

## License

This repository is released under the MIT License.

You are free to use, modify, and distribute this software in accordance with the terms of the MIT License.  
For details, see the [LICENSE](./LICENSE) file included in this repository.
