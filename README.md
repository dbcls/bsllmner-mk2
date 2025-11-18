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
usage: bsllmner2_extract [-h] --bs-entries BS_ENTRIES --mapping MAPPING
                         [--prompt PROMPT] [--format FORMAT] [--model MODEL]
                         [--thinking {true,false}] [--max-entries MAX_ENTRIES]
                         [--ollama-host OLLAMA_HOST] [--with-metrics] [--debug]

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
```

### Main Options

- `--bs-entries`
  - Path to the input JSON or JSONL file containing BioSample entries.
  - Examples:
    - [`tests/test-data/cell_line_example.biosample.json`](./tests/test-data/cell_line_example.biosample.json) (small dataset)
    - [`tests/zenodo-data/biosample_cellosaurus_mapping_testset.json`](./tests/zenodo-data/biosample_cellosaurus_mapping_testset.json) (large dataset)
- `--mapping`
  - Path to the mapping file in TSV format.
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

```
docker compose exec app bsllmner2_select \
  --debug \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --model llama3.1:70b \
  --select-config ./scripts/select-config.json
```

## bsllmner-mk2 with ChIP-Atlas

[ChIP-Atlas](https://chip-atlas.org) is a data-mining suite for exploring epigenomic landscapes by fully integrating ChIP-seq, ATAC-seq, and Bisulfite-seq experiments.
Each SRX entry in ChIP-Atlas contains human-curated metadata values.
Since SRX and BioSample entries are in a one-to-one relationship, we can provide BioSample JSON records to `bsllmner-mk2` and obtain LLM-predicted values for comparison against the human-curated values from ChIP-Atlas.

This enables benchmarking and evaluation of LLM-based NER and ontology mapping against expert-curated annotations.

To facilitate this experiment, the repository provides a script:

- [`scripts/chip_atlas_batch.py`](./scripts/chip_atlas_batch.py)

### Usage

This script automates batch extraction and comparison between LLM-predicted and human-curated values for a set of SRX/BioSample entries.

Usage example:

```sh
$ python3 ./scripts/chip_atlas_batch.py --help
...
$ python3 ./scripts/chip_atlas_batch.py \
  --predict-field cell_type \
  --model llama3.1:70b \
  --num-lines 100
```

During execution, the script automatically downloads various external resources (e.g., ChIP-Atlas tab files, NCBI SRA Accessions.tab, BioSample entry JSON files). All intermediate and output files are stored in the `./tmp-data` directory.

After running the script, the contents of `./tmp-data` may look like:

```sh
$ ls -l tmp-data/
total 31193280
-rw-r--r-- 1 root root 29418575866 Sep  2 11:22 SRA_Accessions.tab
drwxr-xr-x 2 root root     2285568 Sep  2 22:21 bs_entries
-rw-r--r-- 1 root root      106919 Sep  2 21:04 chip_atlas_ner_results_llama3.1:70b_cell_type.json
-rw-r--r-- 1 root root        5816 Sep  2 21:04 chip_atlas_ner_results_llama3.1:70b_cell_type.tsv
-rw-r--r-- 1 root root   344940779 Sep  2 10:04 experimentList.tab
-rw-r--r-- 1 root root   644734357 Sep  2 12:19 experiments.json
-rw-r--r-- 1 root root       63177 Sep  2 12:42 meta_field_keys.txt
-rw-r--r-- 1 root root  1530128296 Sep  2 12:02 srx_to_biosample.json
```

Among these, the main result files are:

- `chip_atlas_ner_results_llama3.1:70b_cell_type.json`
- `chip_atlas_ner_results_llama3.1:70b_cell_type.tsv`

These files contain the LLM-predicted and human-curated values for each entry, and can be used for downstream analysis or benchmarking.

## Download Ontology Files for Searching

You can download ontology files in OBO format file from [Cellosaurus](https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo).

```sh
mkdir -p ontology
cd ontology
curl -O https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo
docker run -v $PWD:/work -w /work --rm -it obolibrary/robot robot convert -i ./cellosaurus.obo -o ./cellosaurus.owl --format owl
```

## License

This repository is released under the MIT License.

You are free to use, modify, and distribute this software in accordance with the terms of the MIT License.  
For details, see the [LICENSE](./LICENSE) file included in this repository.
