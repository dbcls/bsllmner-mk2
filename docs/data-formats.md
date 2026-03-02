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

A TSV file used for evaluating Extract accuracy. A header row is required.

| Column | Description |
|--------|-------------|
| `BioSample ID` | BioSample accession |
| `Experiment type` | Experiment type |
| `extraction answer` | Ground truth extraction value |
| `mapping answer ID` | Ground truth mapping ID |
| `mapping answer label` | Ground truth mapping label |

```tsv
BioSample ID	Experiment type	extraction answer	mapping answer ID	mapping answer label
SAMN00000001	RNA-seq	HeLa	CVCL_0030	HeLa
SAMN00000002	RNA-seq	HEK293	CVCL_0045	HEK293
```

## Extract Result JSON (Result)

Saved to `bsllmner2-results/extract/{run_name}.json`.

```json
{
  "input": {
    "bs_entries": [...],
    "mapping": { "SAMN00000001": { ... } },
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
  "evaluation": [
    {
      "accession": "SAMN00000001",
      "expected": "HeLa",
      "actual": "HeLa",
      "match": true
    }
  ],
  "metrics": null,
  "run_metadata": {
    "run_name": "llama3.1:70b_20250101_120000",
    "model": "llama3.1:70b",
    "thinking": null,
    "username": null,
    "start_time": "20250101_120000",
    "end_time": "20250101_121000",
    "status": "completed",
    "processing_time": 600.0,
    "matched_entries": 1,
    "total_entries": 1,
    "accuracy": 100.0,
    "completed_count": null
  },
  "error_log": null
}
```

### Key Fields

| Path | Type | Description |
|------|------|-------------|
| `input.bs_entries` | `List[Dict]` | Input BioSample entries |
| `input.mapping` | `Dict[str, MappingValue] \| null` | Mapping for evaluation |
| `input.prompt` | `List[Prompt]` | Prompt used |
| `input.model` | `string` | Model name |
| `input.thinking` | `bool \| null` | Thinking mode |
| `input.format` | `JsonSchemaValue \| null` | Output schema |
| `input.config` | `Config` | Runtime configuration |
| `output[].accession` | `string` | BioSample accession |
| `output[].output` | `any \| null` | Parsed extraction result |
| `output[].output_full` | `string \| null` | Raw JSON string |
| `output[].chat_response` | `ChatResponse` | Full Ollama response |
| `evaluation[].match` | `bool` | Whether it matched the ground truth |
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
