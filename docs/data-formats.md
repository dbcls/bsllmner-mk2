# Data Formats

## BioSample JSON Input (bs_entries)

A list of BioSample entries. Supports JSON array or JSONL (one JSON object per line) format.

Each entry must have an `accession` field.

```json
[
  {
    "accession": "SAMN00000001",
    "title": "HeLa cell RNA-seq",
    "characteristics": {
      "cell_line": "HeLa",
      "organism": "Homo sapiens"
    }
  }
]
```

JSONL format:

```
{"accession": "SAMN00000001", "title": "HeLa cell RNA-seq", ...}
{"accession": "SAMN00000002", "title": "HEK293 cell ChIP-seq", ...}
```

## Mapping TSV (for evaluation)

A TSV file used for evaluating Select accuracy. A header row is required.

> **Note:** The `extraction answer` column is the output of a previous tool (MetaSRA), not a human-curated ground truth. It is not used for evaluation. Only `mapping answer ID` (human-curated) is used as the gold standard for Select mode evaluation.

| Column | Description |
|--------|-------------|
| `BioSample ID` | BioSample accession |
| `Experiment type` | Experiment type |
| `extraction answer` | Previous tool output (not used for evaluation) |
| `mapping answer ID` | Human-curated ground truth mapping ID (used for Select evaluation) |
| `mapping answer label` | Ground truth mapping label |

```tsv
BioSample ID Experiment type extraction answer mapping answer ID mapping answer label
SAMN00000001 RNA-seq HeLa CVCL_0030 HeLa
SAMN00000002 RNA-seq HEK293 CVCL_0045 HEK293
```

## Extract Result JSON (ExtractResult)

Saved to `bsllmner2-results/extract/{run_name}.json`.

```json
{
  "entries": [
    {
      "accession": "SAMN00000001",
      "extracted": { "cell_line": "HeLa" },
      "raw_output": "{\"cell_line\": \"HeLa\"}",
      "llm_timing": {
        "total_duration": 1000000000,
        "load_duration": 100000000,
        "eval_count": 50,
        "eval_duration": 500000000,
        "prompt_eval_count": 100
      }
    }
  ],
  "run_metadata": {
    "run_name": "llama3.1:70b_20250101_120000",
    "model": "llama3.1:70b",
    "thinking": false,
    "start_time": "2025-01-01T12:00:00Z",
    "end_time": "2025-01-01T12:10:00Z",
    "status": "completed",
    "processing_time_sec": 600.0,
    "total_entries": 1
  },
  "performance": null,
  "errors": []
}
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `entries[].accession` | `string` | BioSample accession |
| `entries[].extracted` | `dict \| list \| null` | Parsed extraction result |
| `entries[].raw_output` | `string \| null` | Raw JSON string from LLM |
| `entries[].llm_timing` | `LlmTimingFields` | Lightweight timing data (nanoseconds) |
| `run_metadata.run_name` | `string` | Run identifier |
| `run_metadata.model` | `string` | Model name |
| `run_metadata.start_time` | `datetime` | ISO 8601 UTC start time |
| `run_metadata.end_time` | `datetime \| null` | ISO 8601 UTC end time |
| `run_metadata.status` | `"running" \| "completed" \| "failed"` | Run status |
| `run_metadata.processing_time_sec` | `float \| null` | Processing time (seconds) |
| `run_metadata.total_entries` | `int \| null` | Total processed entries |
| `errors` | `list[ErrorLog]` | Error information |

### LlmTimingFields

Lightweight timing fields extracted from `ChatResponse` (nanoseconds). Replaces the full `ChatResponse` in persisted output.

| Field | Type | Description |
|-------|------|-------------|
| `total_duration` | `int` | Total duration (ns) |
| `load_duration` | `int` | Model load duration (ns) |
| `eval_count` | `int` | Number of tokens generated |
| `eval_duration` | `int` | Token generation duration (ns) |
| `prompt_eval_count` | `int` | Number of prompt tokens |

## Select Result JSON (SelectResult)

Saved to `bsllmner2-results/select/select_{run_name}.json`.

```json
{
  "entries": [
    {
      "extract": {
        "accession": "SAMN00000001",
        "extracted": { "cell_line": "HeLa", "tissue": "cervix" },
        "raw_output": "{\"cell_line\": \"HeLa\", \"tissue\": \"cervix\"}",
        "llm_timing": { "total_duration": 0, "load_duration": 0, "eval_count": 0, "eval_duration": 0, "prompt_eval_count": 0 }
      },
      "search_results": {
        "cell_line": {
          "HeLa": [
            {
              "term_uri": "http://purl.obolibrary.org/obo/CVCL_0030",
              "term_id": "CVCL:0030",
              "prop_uri": "http://www.w3.org/2000/01/rdf-schema#label",
              "value": "HeLa",
              "label": "HeLa",
              "exact_match": true,
              "text2term_score": null,
              "reasoning": null,
              "definitions": null,
              "comments": ["Disease: Cervical adenocarcinoma"]
            }
          ]
        }
      },
      "text2term_results": {},
      "select_timings": {
        "cell_line": {
          "HeLa": { "total_duration": 500000000, "load_duration": 0, "eval_count": 20, "eval_duration": 200000000, "prompt_eval_count": 50 }
        }
      },
      "results": {
        "cell_line": [
          {
            "value": "HeLa",
            "term_id": "CVCL:0030",
            "term_uri": "http://purl.obolibrary.org/obo/CVCL_0030",
            "label": "HeLa",
            "exact_match": true,
            "reasoning": "Exact match found for HeLa"
          }
        ]
      }
    }
  ],
  "run_metadata": {
    "run_name": "llama3.1:70b_20250101_120000",
    "model": "llama3.1:70b",
    "thinking": false,
    "start_time": "2025-01-01T12:00:00Z",
    "end_time": "2025-01-01T12:15:00Z",
    "status": "completed",
    "processing_time_sec": 900.0,
    "total_entries": 1
  },
  "evaluation": null,
  "performance": null,
  "errors": []
}
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `entries[].extract` | `ExtractEntry` | Embedded extract result for this entry |
| `entries[].search_results` | `dict[field, dict[value, list[SearchResult]]]` | Stage 2a ontology search results |
| `entries[].text2term_results` | `dict[field, dict[value, list[SearchResult]]]` | Stage 2b text2term results |
| `entries[].search_results.*.[].definitions` | `list[str] \| null` | `obo:IAO_0000115` values collected from the subset OWL. Passed to the Stage 3 LLM as term-level context |
| `entries[].search_results.*.[].comments` | `list[str] \| null` | `rdfs:comment` values. In the default subset OWLs only ChEBI populates this (with `has_role` info as `"{role_type}: {role_label}"`); most other ontologies leave it null |
| `entries[].select_timings` | `dict[field, dict[value, LlmTimingFields]]` | Per-field LLM timing |
| `entries[].results` | `dict[field, list[ResolvedValue]]` | Final mapping results |
| `evaluation` | `EvaluationMetrics \| null` | Evaluation metrics (independent from RunMetadata). All ratio fields (`accuracy`, `precision`, `recall`, `f1`) are stored as 0–1 ratios, not percentages. |
| `errors` | `list[ErrorLog]` | Error information |

### ResolvedValue

Unified result type for Select mode output.

| Field | Type | Description |
|-------|------|-------------|
| `value` | `string` | Original extracted value |
| `term_id` | `string \| null` | Matched ontology term ID |
| `term_uri` | `string \| null` | Matched ontology term URI |
| `label` | `string \| null` | Ontology term label |
| `exact_match` | `bool \| null` | Whether it was an exact match |
| `reasoning` | `string \| null` | LLM reasoning for selection |

## Select Config JSON

Configuration file for Select mode. Defines the ontology file, prompt, and filter for each field.

```json
{
  "fields": {
    "cell_line": {
      "ontology_file": "/app/ontology/cellosaurus.owl",
      "prompt_description": "Cell line is a group of cells that are genetically identical...",
      "ontology_filter": { "hasDbXref": "NCBI_TaxID:9606" },
      "value_type": "string"
    },
    "drug": {
      "ontology_file": "/app/ontology/chebi_subset.owl",
      "prompt_description": "Drug is a chemical or biological substance...",
      "value_type": "array"
    },
    "knockout_gene": {
      "ontology_file": "/app/ontology/ncbi_gene_human.owl",
      "prompt_description": "Knockout gene refers to a gene that has been rendered completely non-functional...",
      "value_type": "array"
    }
  }
}
```

`ontology_filter` is only needed for Cellosaurus; the other ontologies are delivered as pre-subsetted OWLs (`{cl,uberon}_{human,mouse}_subset.owl`, `chebi_subset.owl`, `mondo_human_subset.owl`) generated by `scripts/build_subset_ontologies.sh`.

For the full specification of each field, see [Select Mode - Select Config Customization](select-mode.md#select-config-customization).

## Prompt YAML

Prompts are defined in YAML as a list of `role` and `content`.

```yaml
- role: system
  content: |-
    You are a smart curator of biological data
- role: user
  content: |-
    I will input JSON formatted metadata of a sample...
    Here is the input metadata:
```

`role` must be one of `"system"`, `"user"`, or `"assistant"`.

## Format JSON Schema

A JSON Schema that controls the LLM output format. Passed to the Ollama `format` parameter.

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

In Select mode, the schema is dynamically generated from the SelectConfig field definitions (`build_extract_schema_for_select`). For `value_type: "array"`, it is generated as `{"type": ["array", "null"], "items": {"type": "string"}}`. The generated schema always includes `"additionalProperties": false`.

## PerformanceSummary

Performance data is embedded in the `performance` field of `ExtractResult` and `SelectResult`. There is no separate benchmark file; all data lives inside the result JSON.

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `performance.total_input_entries` | `int` | Total input entries |
| `performance.completed_count` | `int` | Entries that completed processing |
| `performance.total_wall_sec` | `float \| null` | Total wall-clock time (seconds) |
| `performance.stage_timings[]` | `StageTimings[]` | Per-batch stage breakdown |
| `performance.ner_llm_timing` | `LlmTimingSummary \| null` | Aggregated NER LLM timing stats |
| `performance.select_llm_timing` | `LlmTimingSummary \| null` | Aggregated Select LLM timing stats (Select mode only) |
| `performance.disk_io` | `DiskIoTimings` | Disk I/O timing breakdown (Select mode only) |

Accuracy metrics (`accuracy`, `precision`, `recall`, `f1`) are in `SelectResult.evaluation`, not in `PerformanceSummary`.

### LlmTimingSummary Fields

| Field | Description |
|-------|-------------|
| `call_count` | Number of LLM calls |
| `total_duration_sec` | Sum of `total_duration` across all calls |
| `mean_latency_sec` | Mean latency per call (`total_duration - load_duration`) |
| `p50/p95/p99_latency_sec` | Latency percentiles |
| `mean_tokens_per_sec` | Mean generation speed (`eval_count / eval_duration`) |
| `p50/p95_tokens_per_sec` | tokens/sec percentiles |
| `mean_load_duration_sec` | Mean model load time (high = cold start) |
| `max_load_duration_sec` | Max model load time |
| `total_prompt_tokens` | Total prompt tokens processed |
| `total_eval_tokens` | Total tokens generated |

For interpretation guidance, see [benchmarking.md](benchmarking.md).
