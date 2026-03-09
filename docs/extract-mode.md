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
```

1. Load BioSample entries from JSON/JSONL
2. Apply prompt (YAML) and format schema (JSON Schema)
3. Send batch requests to Ollama
4. Extract and parse JSON from responses

## CLI Options

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bs-entries` | Path to the input JSON or JSONL file containing BioSample entries (required) | -- |
| `--model` | LLM model to use for NER | `llama3.1:70b` |
| `--thinking BOOL` | Enable or disable thinking mode for the LLM (`true`/`false`) | `false` |
| `--max-entries` | Process only the first N entries (`-1` for all) | `-1` |
| `--ollama-host` | Host URL for the Ollama server | `http://localhost:11434` |
| `--debug` | Enable debug mode for more verbose logging | `false` |
| `--run-name` | Name of the run for identification purposes | `{model}_{timestamp}` |
| `--resume` | Resume from the last incomplete run | `false` |
| `--batch-size` | Number of entries to process in each batch | `1024` |
| `--num-ctx` | Context length for Ollama | `4096` |

### Extract-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Path to the prompt file in YAML format | `bsllmner2/prompt/prompt_extract.yml` |
| `--format` | Path to the JSON schema file for the output format | `None` |

## Usage Examples

```bash
bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
  --prompt bsllmner2/prompt/prompt_extract.yml \
  --format bsllmner2/format/cell_line.schema.json \
  --model llama3.1:70b \
  --debug
```

With Docker:

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
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
