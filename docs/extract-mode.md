# Extract Mode

Performs Named Entity Recognition (NER) on BioSample records using an LLM to extract biological information in specified categories.

## Overview

```
BioSample JSON
      |
      v
+------------+
| Load       |
| bs_entries |
+------------+
      |
      v
+------------+
| Build      |
| messages   |
+------------+
      |
      v
+------------+
| Ollama     |
| chat()     |
+------------+
      |
      v
+------------+
| Parse JSON |
| response   |
+------------+
      |
      v
+------------+
| Evaluate   |
| (optional) |
+------------+
```

1. Load BioSample entries from JSON/JSONL
2. Apply prompt (YAML) and format schema (JSON Schema)
3. Send batch requests to Ollama
4. Extract and parse JSON from responses
5. Evaluate accuracy if a mapping file is provided

## CLI Options

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bs-entries` | Path to the input JSON or JSONL file containing BioSample entries (required) | -- |
| `--mapping` | Path to the mapping file in TSV format (for evaluation) | `None` |
| `--model` | LLM model to use for NER | `llama3.1:70b` |
| `--thinking BOOL` | Enable or disable thinking mode for the LLM (`true`/`false`) | `None` |
| `--max-entries` | Process only the first N entries (`-1` for all) | `-1` |
| `--ollama-host` | Host URL for the Ollama server | `http://localhost:11434` |
| `--with-metrics` | Enable collection of metrics during processing. Requires Docker environment; collects container resource usage (CPU, memory, network, block I/O) via `docker stats` and GPU metrics via `nvidia-smi` from the Ollama container (`bsllmner-mk2-ollama`). | `false` |
| `--debug` | Enable debug mode for more verbose logging | `false` |
| `--run-name` | Name of the run for identification purposes | `{model}_{timestamp}` |
| `--resume` | Resume from the last incomplete run | `false` |
| `--batch-size` | Number of entries to process in each batch | `1024` |

### Extract-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Path to the prompt file in YAML format | `bsllmner2/prompt/prompt_extract.yml` |
| `--format` | Path to the JSON schema file for the output format | `None` |

## Usage Examples

```bash
bsllmner2_extract \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --mapping tests/test-data/cell_line_example.mapping.tsv \
  --prompt bsllmner2/prompt/prompt_extract.yml \
  --format bsllmner2/format/cell_line.schema.json \
  --model llama3.1:70b \
  --with-metrics \
  --debug
```

With Docker:

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/test-data/cell_line_example.biosample.json \
  --mapping tests/test-data/cell_line_example.mapping.tsv \
  --model llama3.1:70b
```

## Prompt Specification

Prompts are defined as a YAML list where each element has `role` and `content`.

```yaml
- role: system
  content: |-
    You are a smart curator of biological data
- role: user
  content: |-
    I will input JSON formatted metadata of a sample...
    Here is the input metadata:
```

At runtime, the BioSample entry JSON is appended to the `content` of the last message.

### Customization

1. Copy the built-in prompt (`bsllmner2/prompt/prompt_extract.yml`)
2. Edit the category descriptions and output rules
3. Specify the file with `--prompt`

## Output Format Specification

When a JSON Schema is specified with `--format`, the Ollama structured output feature (`format` parameter) constrains the output format.

Built-in schema (`bsllmner2/format/cell_line.schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "cell_line": { "type": ["string", "null"] }
  },
  "required": ["cell_line"],
  "additionalProperties": true
}
```

If `--format` is omitted, the LLM responds in free form. The last JSON object/array is extracted from the response using a regex and parsed.

## Resume

When `--resume` is specified, processing continues from the previous interruption. The resume file is automatically deleted after successful completion.

The same `--run-name` must be specified as the original run. If the original run used the auto-generated name (`{model}_{timestamp}`), you need to find it from the resume file in `bsllmner2-results/extract/`.

## Result Files

See [Data Formats](data-formats.md) for the full result schema.

| File | Description |
|------|-------------|
| `bsllmner2-results/extract/{run_name}.json` | Complete result |
| `bsllmner2-results/extract/{run_name}_resume.json` | Resume intermediate file (during processing only) |

The default `run_name` is `{model}_{YYYYMMDD_HHMMSS}` (UTC).

## Evaluation

When a mapping file is provided, the extract result is compared against the ground truth to calculate accuracy.

- `evaluation[].match`: Exact match between the extracted value (`output.cell_line`) and the ground truth (`mapping[accession].extraction_answer`)
- `run_metadata.accuracy`: `matched_entries / total_entries * 100` (%)
