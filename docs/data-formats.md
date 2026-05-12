# Data Formats

This page is the canonical schema reference for every file bsllmner-mk2 reads or writes.

## BioSample Input (bs_entries)

A list of BioSample entries. Accepted as either a JSON array or JSONL (one object per line). Each entry must contain an `accession`.

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

JSONL:

```
{"accession": "SAMN00000001", "title": "HeLa cell RNA-seq", ...}
{"accession": "SAMN00000002", "title": "HEK293 cell ChIP-seq", ...}
```

EBI-style entries (those whose `characteristics` is a dict mapping each key to a list of `{text, ...}` objects) are flattened by `construct_llm_input_json()` before being passed to the LLM; the first element's `text` is taken. Keys listed in `bsllmner2/filter_keys.json` are dropped.

## Mapping TSV (for evaluation)

A TSV with the header row below. Only `mapping answer ID` (human-curated) is used as the gold standard by `bsllmner2_select --mapping`; the other columns are informational.

| Column | Description |
|---|---|
| `BioSample ID` | BioSample accession. |
| `Experiment type` | Experiment type. |
| `extraction answer` | Auxiliary annotation. |
| `mapping answer ID` | Ground-truth ontology term ID. |
| `mapping answer label` | Ground-truth label. |

```tsv
BioSample ID	Experiment type	extraction answer	mapping answer ID	mapping answer label
SAMN00000001	RNA-seq	HeLa	CVCL_0030	HeLa
SAMN00000002	RNA-seq	HEK293	CVCL_0045	HEK293
```

Evaluation compares predicted `term_id` against `mapping answer ID` per accession for the `cell_line` field. See `bsllmner2/pipeline.evaluate_select_output()` and [Benchmarking](benchmarking.md) for how the metrics are computed and read.

## ExtractResult

Written to `bsllmner2-results/extract/{run_name}.json`.

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

### Schema

| Path | Type | Description |
|---|---|---|
| `entries[].accession` | `string` | BioSample accession. |
| `entries[].extracted` | `dict \| list \| null` | Parsed JSON value from the LLM. |
| `entries[].raw_output` | `string \| null` | The last JSON substring extracted from the raw response text. |
| `entries[].llm_timing` | `LlmTimingFields` | Per-call timing (nanoseconds). See below. |
| `run_metadata.run_name` | `string` | Run identifier. |
| `run_metadata.model` | `string` | Ollama model. |
| `run_metadata.thinking` | `bool` | Whether thinking mode was on. |
| `run_metadata.start_time` | `datetime` | ISO 8601 UTC. |
| `run_metadata.end_time` | `datetime \| null` | ISO 8601 UTC. |
| `run_metadata.status` | `"running" \| "completed" \| "failed"` | Run status. |
| `run_metadata.processing_time_sec` | `float \| null` | `end_time - start_time` in seconds. |
| `run_metadata.total_entries` | `int \| null` | Number of entries in the result. |
| `performance` | `PerformanceSummary \| null` | See [PerformanceSummary](#performancesummary). |
| `errors` | `list[ErrorLog]` | Captured errors. |

### LlmTimingFields

Subset of the Ollama `ChatResponse` timing fields, in nanoseconds.

| Field | Type | Description |
|---|---|---|
| `total_duration` | `int` | Total duration (ns). |
| `load_duration` | `int` | Model load duration (ns). |
| `eval_count` | `int` | Tokens generated. |
| `eval_duration` | `int` | Generation duration (ns). |
| `prompt_eval_count` | `int` | Prompt tokens. |

## SelectResult

Written to `bsllmner2-results/select/select_{run_name}.json`.

```json
{
  "entries": [
    {
      "extract": {
        "accession": "SAMN00000001",
        "extracted": { "cell_line": "HeLa", "tissue": "cervix" },
        "raw_output": "{\"cell_line\": \"HeLa\", \"tissue\": \"cervix\"}",
        "llm_timing": { "total_duration": 2000000000, "load_duration": 100000000, "eval_count": 80, "eval_duration": 1000000000, "prompt_eval_count": 250 }
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
            "reasoning": "Exact match on rdfs:label"
          }
        ]
      }
    }
  ],
  "run_metadata": { "...": "as in ExtractResult" },
  "evaluation": null,
  "performance": null,
  "errors": []
}
```

### Schema

| Path | Type | Description |
|---|---|---|
| `entries[].extract` | `ExtractEntry` | Embedded extract result for this entry. |
| `entries[].search_results` | `dict[field, dict[value, list[SearchResult]]]` | Stage 2a word-combination candidates. See [SearchResult](#searchresult). |
| `entries[].text2term_results` | `dict[field, dict[value, list[SearchResult]]]` | Stage 2b text2term candidates. See [SearchResult](#searchresult). |
| `entries[].select_timings` | `dict[field, dict[value, LlmTimingFields]]` | Per `(field, value)` Stage 3 LLM call timing. |
| `entries[].results` | `dict[field, list[ResolvedValue]]` | Final per-field selections. See [ResolvedValue](#resolvedvalue). |
| `evaluation` | `EvaluationMetrics \| null` | Set when `--mapping` is supplied. `accuracy`, `precision`, `recall`, `f1` are stored as 0-1 ratios (not percentages). |
| `errors` | `list[ErrorLog]` | Captured errors. |

### SearchResult

One candidate returned by Stage 2 ontology search (Stage 2a word-combination index and Stage 2b text2term). Stored under `entries[].search_results[field][value][]` and `entries[].text2term_results[field][value][]`.

| Field | Type | Description |
|---|---|---|
| `term_uri` | `string` | Ontology term URI. |
| `term_id` | `string` | Normalised term ID (e.g. `CVCL:0030`). |
| `prop_uri` | `string \| null` | URI of the property that matched the value (e.g. `http://www.w3.org/2000/01/rdf-schema#label`). |
| `value` | `string` | Property value that produced the match. |
| `label` | `string \| null` | Preferred label of the term (`rdfs:label` / `skos:prefLabel`). |
| `exact_match` | `bool` | `true` when the query and `value` are equal after NFKC normalisation. |
| `text2term_score` | `float \| null` | text2term similarity score. `null` for Stage 2a hits. |
| `reasoning` | `string \| null` | Human-readable provenance: `"Exact match on <prop>"` for Stage 2a exact hits, `"text2term score: ..."` for Stage 2b hits, otherwise `null`. |
| `definitions` | `list[str] \| null` | `obo:IAO_0000115` definitions from the OWL. Surfaced to the Stage 3 LLM as context only. |
| `comments` | `list[str] \| null` | `rdfs:comment` values. Populated mainly by ChEBI; see [Ontology Preparation](ontology.md). |

### ResolvedValue

One final per-field pick. Stored under `entries[].results[field][]`.

| Field | Type | Description |
|---|---|---|
| `value` | `string` | The extracted value. |
| `term_id` | `string \| null` | Picked ontology term ID. |
| `term_uri` | `string \| null` | Picked ontology term URI. |
| `label` | `string \| null` | Picked term label. |
| `exact_match` | `bool \| null` | Whether the pick came from an exact word-combination match. |
| `reasoning` | `string \| null` | Stage 3 LLM reasoning, or `"Exact match on <prop>"` for Stage 2a picks, or `"text2term score: ..."` for Stage 2b picks. |

## SelectConfig

```json
{
  "fields": {
    "cell_line": {
      "ontology_file": "ontology/cellosaurus_human.owl",
      "prompt_description": "Cell line is a group of cells that are genetically identical...",
      "value_type": "string"
    },
    "drug": {
      "ontology_file": "ontology/chebi_subset.owl",
      "prompt_description": "Drug is a chemical or biological substance...",
      "value_type": "array"
    }
  }
}
```

| Property | Type | Default | Description |
|---|---|---|---|
| `fields` | `dict[str, SelectConfigField]` | (required) | Field name to field config. |
| `fields[name].ontology_file` | `string \| null` | `null` | Path to an OWL or TSV/CSV. `null` skips Stage 2/3 mapping and returns the extracted value as-is. |
| `fields[name].prompt_description` | `string \| null` | `null` | Description injected into the Stage 1 prompt. |
| `fields[name].value_type` | `"string" \| "array"` | `"string"` | `"array"` extracts a list of values for the field. |

Ontology files are built per [Ontology Preparation](ontology.md). See [Select Mode](select-mode.md#selectconfig-customization) for usage notes.

## Prompt YAML

```yaml
- role: system
  content: |-
    You are a smart curator of biological data
- role: user
  content: |-
    I will input JSON formatted metadata of a sample...
    Here is the input metadata:
```

`role` must be `"system"`, `"user"`, or `"assistant"`. At runtime, the entry JSON is appended to the last message's `content`.

## Format JSON Schema

Passed to Ollama's `format=` parameter.

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

In Select mode, the schema is synthesised from the select config by `build_extract_schema_for_select()`. For `value_type: "array"`, the property becomes `{"type": ["array", "null"], "items": {"type": "string"}}`. The generated schema sets `additionalProperties: false`.

## PerformanceSummary

Embedded in the `performance` field of `ExtractResult` and `SelectResult`. There is no separate benchmark file -- all timing data lives inside the result JSON.

| Path | Type | Description |
|---|---|---|
| `performance.total_input_entries` | `int` | Total input entries. |
| `performance.completed_count` | `int` | Entries that completed processing. |
| `performance.total_wall_sec` | `float \| null` | Total wall-clock time (seconds). |
| `performance.stage_timings[]` | `list[StageTimings]` | Per-batch stage breakdown. |
| `performance.ner_llm_timing` | `LlmTimingSummary \| null` | Aggregated NER LLM timing. |
| `performance.select_llm_timing` | `LlmTimingSummary \| null` | Aggregated Stage 3 LLM timing (Select only). |
| `performance.disk_io` | `DiskIoTimings` | Disk I/O timings (Select only). |

Accuracy metrics live in `SelectResult.evaluation`, not in `PerformanceSummary`.

### LlmTimingSummary

| Field | Description |
|---|---|
| `call_count` | Number of LLM calls. |
| `total_duration_sec` | Sum of `total_duration` across calls. |
| `mean_latency_sec` | Mean of `(total_duration - load_duration)`. |
| `p50/p95/p99_latency_sec` | Latency percentiles. |
| `mean_tokens_per_sec` | Mean `eval_count / eval_duration`. |
| `p50/p95_tokens_per_sec` | tokens/sec percentiles. |
| `mean_load_duration_sec` | Mean model load time. |
| `max_load_duration_sec` | Max model load time. |
| `total_prompt_tokens` | Sum of `prompt_eval_count`. |
| `total_eval_tokens` | Sum of `eval_count`. |

### StageTimings

One entry per processed batch.

| Field | Type | Description |
|---|---|---|
| `batch_idx` | `int` | Zero-based batch index. |
| `batch_size` | `int` | Entries in this batch. |
| `ner_sec` | `float \| null` | Stage 1 NER wall-clock time. |
| `ontology_search_sec` | `float \| null` | Stage 2a word-combination search time. |
| `text2term_sec` | `float \| null` | Stage 2b `text2term.map_terms()` time. |
| `llm_select_sec` | `float \| null` | Stage 3 LLM selection time (max-across-fields under `asyncio.gather`). |
| `resume_write_sec` | `float \| null` | Per-batch resume-file write time. |

### DiskIoTimings

Run-wide. Each list grows by one entry per operation, so `len(list)` is the operation count.

| Field | Type | Description |
|---|---|---|
| `index_cache_load_sec` | `list[float]` | `OntologyIndex` cache load time (cache hit). |
| `index_cache_save_sec` | `list[float]` | `OntologyIndex` cache save time (after rebuild). |
| `index_build_from_file_sec` | `list[float]` | `OntologyIndex` rebuild from OWL/TSV (cache miss). |
| `text2term_cache_build_sec` | `list[float]` | `text2term.cache_ontology()` build time (first run per OWL). |
| `text2term_cache_load_sec` | `list[float]` | `text2term.cache_exists()` check time (cache-hit path). |
| `resume_write_sec` | `list[float]` | Per-batch resume write time. |

For interpretation guidance see [Benchmarking](benchmarking.md).
