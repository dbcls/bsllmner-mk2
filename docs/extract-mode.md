# Extract Mode

`bsllmner2_extract` performs Named Entity Recognition (NER) over BioSample records using an LLM and emits structured JSON.

## Overview

```
BioSample JSON/JSONL
         |
         v
+--------------------+
| Filter keys        |  bsllmner2/biosample.py
| (filter_keys.json) |
+--------------------+
         |
         v
+--------------------+
| Build messages     |  Prompt YAML + entry JSON appended
+--------------------+
         |
         v
+--------------------+
| Ollama chat()      |  optional JSON Schema via format=
+--------------------+
         |
         v
+--------------------+
| Parse last JSON    |  extracted + raw_output captured
+--------------------+
         |
         v
   ExtractResult JSON
```

Input is loaded as JSON (a list of objects) or JSONL (one object per line). Each entry is normalised with `construct_llm_input_json()` before being appended to the prompt's final user message:

- Keys listed in `bsllmner2/filter_keys.json` are dropped.
- Non-EBI entries pass through unchanged (minus filtered keys).
- EBI-style entries (top-level `characteristics: dict`) are flattened to `{key: characteristics[key][0]["text"]}`. Attributes whose value is **not** a non-empty list of objects containing a `text` key are silently dropped.

## CLI

See [CLI Reference](cli.md#bsllmner2_extract) for the full option table. Extract-specific options at a glance:

| Option | Default | Purpose |
|---|---|---|
| `--prompt PATH` | `bsllmner2/prompt/prompt_extract.yml` | Prompt YAML to use. |
| `--format PATH` | (none) | JSON Schema enforced via Ollama's `format=` parameter. |

Example:

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
  --prompt bsllmner2/prompt/prompt_extract.yml \
  --format bsllmner2/format/cell_line.schema.json \
  --model llama3.1:70b \
  --debug
```

## Prompt YAML

A prompt file is a YAML list of `{role, content}` messages. The built-in `bsllmner2/prompt/prompt_extract.yml` extracts `cell_line`, `tissue`, and `organism`:

```yaml
- role: system
  content: |-
    You are a smart curator of biological data
- role: user
  content: |-
    I will input JSON formatted metadata of a sample for a biological experiment.
    ...
    Here is the input metadata:
```

At runtime, the entry's filtered JSON is appended (with a leading newline) to the `content` of the last message. To customise: copy the built-in YAML, edit categories and output rules, then pass it with `--prompt`.

Note: when extraction runs as the first stage of [Select Mode](select-mode.md#stage-1-ner-extraction), the prompt is synthesised in code from the select config -- `--prompt` is not used in that flow.

## Output Format Schema

Pass a JSON Schema via `--format` to enforce structured output. Ollama's `format=` parameter applies the schema to the model output. Built-in example (`bsllmner2/format/cell_line.schema.json`):

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

If `--format` is omitted the LLM responds in free form. `bsllmner2` then scans the message text and parses the **last** valid JSON object/array it finds; both the parsed value (`extracted`) and the raw JSON substring (`raw_output`) are preserved.

## Output

Result file: `bsllmner2-results/extract/{run_name}.json` (and a per-batch `{run_name}_resume.json` while running). For the full `ExtractResult` schema, see [Data Formats](data-formats.md#extractresult).
