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

## Extract Result JSON (Result)

Saved to `bsllmner2-results/extract/{run_name}.json`.

```json
{
  "input": {
    "bs_entries": [...],
    "prompt": [
      { "role": "system", "content": "..." },
      { "role": "user", "content": "..." }
    ],
    "model": "llama3.1:70b",
    "thinking": null,
    "format": { ... },
    "config": {
      "ollama_host": "http://localhost:11434",
      "debug": false
    },
    "cli_args": null
  },
  "output": [
    {
      "accession": "SAMN00000001",
      "output": { "cell_line": "HeLa" },
      "output_full": "{\"cell_line\": \"HeLa\"}",
      "characteristics": null,
      "taxId": null,
      "chat_response": { ... }
    }
  ],
  "run_metadata": {
    "run_name": "llama3.1:70b_20250101_120000",
    "model": "llama3.1:70b",
    "thinking": null,
    "username": null,
    "start_time": "20250101_120000",
    "end_time": "20250101_121000",
    "status": "completed",
    "processing_time": 600.0,
    "matched_entries": null,
    "total_entries": 1,
    "accuracy": null,
    "completed_count": null
  },
  "error_log": null
}
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `input.bs_entries` | `List[Dict]` | Input BioSample entries |
| `input.prompt` | `List[Prompt]` | Prompt used |
| `input.model` | `string` | Model name |
| `input.thinking` | `bool \| null` | Thinking mode |
| `input.format` | `JsonSchemaValue \| null` | Output schema |
| `input.config` | `Config` | Runtime configuration |
| `output[].accession` | `string` | BioSample accession |
| `output[].output` | `any \| null` | Parsed extraction result |
| `output[].output_full` | `string \| null` | Raw JSON string |
| `output[].chat_response` | `ChatResponse` | Full Ollama response |
| `run_metadata.status` | `"running" \| "completed" \| "failed"` | Run status |
| `run_metadata.accuracy` | `float \| null` | Accuracy (%) |
| `error_log` | `ErrorLog \| null` | Error information |

## Select Result JSON (SelectResult)

Saved to `bsllmner2-results/select/select_{run_name}.json`.

```json
[
  {
    "accession": "SAMN00000001",
    "extract_output": { "cell_line": "HeLa", "tissue": "cervix" },
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
            "reasoning": null
          }
        ]
      }
    },
    "text2term_results": { ... },
    "llm_chat_response": { ... },
    "results": {
      "cell_line": {
        "HeLa": {
          "term_uri": "http://purl.obolibrary.org/obo/CVCL_0030",
          "term_id": "CVCL:0030",
          "prop_uri": "http://www.w3.org/2000/01/rdf-schema#label",
          "value": "HeLa",
          "label": "HeLa",
          "exact_match": true,
          "text2term_score": null,
          "reasoning": "Exact match found for HeLa"
        }
      }
    }
  }
]
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `accession` | `string` | BioSample accession |
| `extract_output` | `any \| null` | Stage 1 NER result |
| `search_results` | `Dict[field, Dict[value, List[SearchResult]]]` | Stage 2a ontology search results |
| `text2term_results` | `Dict[field, Dict[value, List[SearchResult]]]` | Stage 2b text2term results |
| `llm_chat_response` | `Dict[field, Dict[value, ChatResponse \| null]]` | Stage 3 LLM responses |
| `results` | `Dict[field, Dict[value, SearchResult \| null] \| any]` | Final mapping results |

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
      "ontology_file": "/app/ontology/chebi.owl",
      "prompt_description": "Drug is a chemical or biological substance...",
      "value_type": "array"
    },
    "gene_perturbation": {
      "prompt_description": "Experimental perturbation applied to the target gene...",
      "value_type": "array"
    }
  }
}
```

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

In Select mode, the schema is dynamically generated from the SelectConfig field definitions (`build_extract_schema_for_select`). For `value_type: "array"`, it is generated as `{"type": ["array", "null"], "items": {"type": "string"}}`.

## Benchmark Summary JSON

Saved to `bsllmner2-results/benchmarks/{run_name}_benchmark.json`. Generated automatically on every Extract/Select run.

```json
{
  "run_name": "llama3.1:70b_20250101_120000",
  "model": "llama3.1:70b",
  "thinking": null,
  "total_entries": 100,
  "completed_count": 100,
  "total_wall_sec": 342.5,
  "stage_timings": [
    {
      "batch_idx": 0,
      "batch_size": 50,
      "ner_sec": 120.3,
      "ontology_search_sec": null,
      "text2term_sec": null,
      "llm_select_sec": null,
      "resume_write_sec": 0.05
    }
  ],
  "ner_llm_timing": {
    "call_count": 100,
    "total_duration_sec": 280.0,
    "mean_latency_sec": 2.7,
    "p50_latency_sec": 2.5,
    "p95_latency_sec": 4.1,
    "p99_latency_sec": 5.8,
    "mean_tokens_per_sec": 45.2,
    "p50_tokens_per_sec": 46.0,
    "p95_tokens_per_sec": 38.1,
    "mean_load_duration_sec": 0.001,
    "max_load_duration_sec": 0.85,
    "total_prompt_tokens": 50000,
    "total_eval_tokens": 5000
  },
  "select_llm_timing": null,
  "disk_io": {
    "index_cache_load_sec": [],
    "index_cache_save_sec": [],
    "index_build_from_file_sec": [],
    "resume_write_sec": []
  },
  "select_accuracy": null,
  "select_precision": null,
  "select_recall": null,
  "select_f1": null,
  "select_matched_entries": null
}
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `run_name` | `string` | Run identifier |
| `model` | `string` | Ollama model name |
| `thinking` | `bool \| null` | Whether thinking mode was enabled |
| `total_entries` | `int` | Total input entries |
| `completed_count` | `int` | Entries that completed processing |
| `total_wall_sec` | `float \| null` | Total wall-clock time (seconds) |
| `stage_timings[]` | `StageTimings[]` | Per-batch stage breakdown |
| `ner_llm_timing` | `LlmTimingSummary \| null` | Aggregated NER LLM timing stats |
| `select_llm_timing` | `LlmTimingSummary \| null` | Aggregated Select LLM timing stats (Select mode only) |
| `disk_io` | `DiskIoTimings` | Disk I/O timing breakdown (Select mode only) |
| `select_accuracy` | `float \| null` | Select accuracy (%) (Select mode only) |
| `select_precision` | `float \| null` | Select precision (%) (Select mode only) |
| `select_recall` | `float \| null` | Select recall (%) (Select mode only) |
| `select_f1` | `float \| null` | Select F1-score (%) (Select mode only) |
| `select_matched_entries` | `int \| null` | Number of correctly matched entries (Select mode only) |

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
