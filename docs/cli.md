# CLI Reference

This page lists every option of `bsllmner2_extract` and `bsllmner2_select`. Default values are taken from the implementation; see [`bsllmner2/cli_common.py`](https://github.com/dbcls/bsllmner-mk2/blob/main/bsllmner2/cli_common.py).

## bsllmner2_extract

Runs Named Entity Recognition (NER) over BioSample records. Output: `bsllmner2-results/extract/{run_name}.json` (see [Data Formats](data-formats.md#extractresult)).

### Common Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--bs-entries PATH` | path | (required) | Input BioSample JSON or JSONL file. |
| `--model STR` | string | `llama3.1:70b` | Ollama model identifier. |
| `--thinking BOOL` | bool | `false` | Enable the model's thinking mode. Accepts `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off`. |
| `--max-entries INT` | int | `-1` | Process only the first N entries. `-1` means all. |
| `--ollama-host URL` | string | `http://localhost:11434` | Ollama server URL. Usually inherited from the `OLLAMA_HOST` environment variable. |
| `--debug` | flag | off | Enable DEBUG-level logging. |
| `--run-name STR` | string | `{model}_{YYYYmmdd_HHMMSS}` | Identifier used in output and resume file names. |
| `--resume` | flag | off | Resume from an incomplete run. See [Resume](#resume). |
| `--batch-size INT` | int | `1024` | Entries per batch. Each batch dumps a resume file. |
| `--num-ctx INT` | int | `4096` | Ollama context length (`num_ctx`). |

### Extract-Specific Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--prompt PATH` | path | `bsllmner2/prompts/extract.yml` | YAML prompt file (list of `{role, content}` messages). |
| `--format PATH` | path | (none) | JSON Schema file passed as the Ollama `format` option. Example: `bsllmner2/format/cell_line.schema.json`. |

## bsllmner2_select

Extract + ontology mapping. Output: `bsllmner2-results/select/select_{run_name}.json` (see [Data Formats](data-formats.md#selectresult)).

### Common Options

Same as [bsllmner2_extract](#common-options).

### Select-Specific Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--select-config PATH` | path | (required) | Select configuration JSON. Schema: see [Data Formats](data-formats.md#selectconfig). |
| `--mapping PATH` | path | (none) | Mapping TSV used to compute evaluation metrics against a gold standard. Format: see [Data Formats](data-formats.md#mapping-tsv-for-evaluation). |
| `--no-reasoning` | flag | reasoning on | Disable reasoning in the LLM selection step. When omitted, the LLM emits a `reasoning` field alongside each picked term ID. |

`bsllmner2_select` reuses `bsllmner2_extract`'s NER stage with a prompt and JSON Schema dynamically built from `--select-config`. See [Select Mode](select-mode.md#stage-1-ner-extraction) for the Stage 1 prompt construction details.

## Resume

`--resume` continues a prior run that ended with `status != "completed"`. The same `--run-name` must be supplied so the CLI can locate the existing resume files.

Files written atomically per batch under `BSLLMNER2_RESULT_DIR`:

| File | Written by | Contents |
|---|---|---|
| `extract/{run_name}_resume.json` | `bsllmner2_extract`, `bsllmner2_select` | `list[ExtractEntry]` of completed extract entries. |
| `select/select_{run_name}_resume.json` | `bsllmner2_select` | `list[SelectEntry]` of completed select entries. |

Resume behaviour:

- `bsllmner2_extract` -- accessions present in the extract resume file are skipped; remaining entries are processed.
- `bsllmner2_select` -- the CLI cross-checks both resume files:
    - Accessions present in both files are skipped.
    - **Orphans** (in extract, missing in select) are re-run through the select stage before normal batch processing resumes.
    - If the select resume file references an accession the extract resume file does not contain, the run aborts with `ResumeDataError` to prevent silent data corruption.

When a run finishes with `status == "completed"`, the corresponding resume files are removed automatically.

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Run finished with `status == "completed"` and zero batch errors. |
| `1` | Run was aborted or one or more batches reported errors. The result JSON is still written. |
