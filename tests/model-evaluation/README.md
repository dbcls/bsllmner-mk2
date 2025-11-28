# Model Evaluation

This document describes the evaluation workflow for ontology mapping performance using various LLM models running on Ollama.  
The evaluation measures both **runtime performance** (extract/select stages) and **ontology mapping accuracy** (cell line ontology predictions).

## Dataset

Two datasets are used:

- Input BioSample dataset (600 entries)
  - [`../tests/zenodo-data/biosample_cellosaurus_mapping_testset.json`](../tests/zenodo-data/biosample_cellosaurus_mapping_testset.json)
  - This file contains 600 BioSample entries used as input for extraction and selection.
- Human-curated ontology mapping
  - [`../tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv`](../tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv)
  - This file contains the correct ontology IDs for the `cell_line` field and is used to compute evaluation metrics.

## Models Evaluated

The following models are evaluated:

```plain
deepseek-r1:8b
deepseek-r1:32b
gemma3:4b
gemma3:12b
gemma3:27b
gpt-oss:20b
llama3.1:8b
phi4:14b
qwen3:4b
qwen3:8b
qwen3:32b
```

All models must be pulled in advance.

## Pulling All Models (Host Machine)

Pull all required models into the Ollama instance:

```bash
for model in \
  "deepseek-r1:8b" \
  "deepseek-r1:32b" \
  "gemma3:4b" \
  "gemma3:12b" \
  "gemma3:27b" \
  "gpt-oss:20b" \
  "llama3.1:8b" \
  "phi4:14b" \
  "qwen3:4b" \
  "qwen3:8b" \
  "qwen3:32b"; do
  docker compose exec ollama ollama pull $model
done
```

## Extraction and Selection Configuration

The extraction/selection workflow is performed using [`../../scripts/select-config.json`](../../scripts/select-config.json):

The following fields are processed:

- `cell_line`
- `cell_type`
- `tissue`
- `disease`
- `drug`

These fields are extracted and then ontology candidates are selected.

## Running the Batch Evaluation

The batch script:

```bash
bash run-model-evaluation-batch.sh
```

Executes, for each model:

```bash
bsllmner2_select \
    --bs-entries "$BS_ENTRIES" \
    --select-config "$SELECT_CONFIG" \
    --run-name "$run_name" \
    --resume \
    --thinking false \
    --no-reasoning \
    --model "$model"
```

### Notes

- Both **thinking mode** and **reasoning mode** are disabled.
- Ollama is configured to run with `OLLAMA_NUM_PARALLEL=16` allowing **16 parallel requests per model**.
- Logs are written under `./model-evaluation-batch-logs/`
- Extraction/selection JSON results are stored under `<repository_root>/bsllmner2-results/`

## Running the Aggregated Evaluation

Once all models complete, run:

```bash
python tests/model-evaluation/evaluation_results.py
```

This script:

- Parses logs to measure extraction/selection runtime.
- Loads predicted ontology mappings.
- Loads human-curated ontology mappings.
- Computes evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - Accuracy
- Retrieves model metadata from Ollama:
  - parameter size
  - context length
  - embedding length
- Produces a consolidated table in TSV format `evaluation_results.tsv` under the current directory.

## Saving Extract/Select JSON Outputs

Extraction/selection JSON files are copied for archival:

```bash
mkdir -p ./model-evaluation-results/extract
mkdir -p ./model-evaluation-results/select

cp -r /app/bsllmner2-results/extract/models-with-large-dataset-* ./model-evaluation-results/extract/
cp -r /app/bsllmner2-results/select/select_models-with-large-dataset-* ./model-evaluation-results/select/
```

Directory structure:

```bash
model-evaluation-results/
├── extract
│   ├── models-with-large-dataset-deepseek-r1_32b.json
│   ├── models-with-large-dataset-deepseek-r1_8b.json
│   ├── models-with-large-dataset-gemma3_12b.json
│   ├── ...
│   └── models-with-large-dataset-qwen3_8b.json
└── select
    ├── select_models-with-large-dataset-deepseek-r1_32b.json
    ├── select_models-with-large-dataset-deepseek-r1_8b.json
    ├── select_models-with-large-dataset-gemma3_12b.json
    ├── ...
    └── select_models-with-large-dataset-qwen3_8b.json
```

These files can be used for debugging or manual analysis.

## Evaluation Criteria

### Runtime Metrics

Measured from logs:

- Extract stage time (sec)
- Selection stage time (sec)
- Total LLM compute time = extract + selection

### Ontology Accuracy Metrics

Evaluated for the `cell_line` field (600 samples):

- **Precision (%)**
  Correct predicted ontology strings / all predicted ontology strings
- **Recall (%)**
  Correct predictions / all human-curated ontology strings
- **F1-score**
  Harmonic mean of precision and recall
- **Accuracy (%)**
  Exact matches (including None)

## Final Output

The final summary file [`evaluation_results.tsv`](evaluation_results.tsv) contains a table like the following:

includes, for each model:

- Parameter size
- Context length
- Embedding length
- Accuracy / Precision / Recall / F1-score
- Extraction & selection time
- Extracted / selected / final field counts

This table enables a clear comparison of both **LLM runtime characteristics** and **ontology mapping accuracy** across all evaluated models.
