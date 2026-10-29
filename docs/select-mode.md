# Select Mode

`bsllmner2_select` runs a three-stage pipeline: NER extraction, ontology search (word combinations + text2term), and LLM selection from the resulting candidate set.

## Overview

```
BioSample JSON/JSONL
        |
        v
+--------------------------------------------+
| Stage 1: NER Extraction                    |
| dynamic prompt + JSON Schema from          |
| SelectConfig                               |
+--------------------------------------------+
        |
        v
+--------------------------------------------+
| Stage 2: Ontology Search                   |
| 2a. Word-combination index lookup          |
| 2b. text2term similarity fallback (OWL)    |
+--------------------------------------------+
        |
        v
+--------------------------------------------+
| Stage 3: LLM Selection                     |
| pick best term_id per (field, value)       |
+--------------------------------------------+
        |
        v
   SelectResult
```

Select mode is self-contained; there is no need to run `bsllmner2_extract` separately.

## CLI

See [CLI Reference](cli.md#bsllmner2_select) for the full option table. Select-specific options at a glance:

| Option | Default | Purpose |
|---|---|---|
| `--select-config PATH` | (required) | Field-to-ontology mapping (see [SelectConfig Customization](#selectconfig-customization)). |
| `--mapping PATH` | (none) | Gold-standard TSV for evaluation metrics. |
| `--no-reasoning` | reasoning on | Strip the `reasoning` field from the Stage 3 schema. |

Example:

```bash
docker compose exec app bsllmner2_select \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --select-config scripts/select-config-hg38.json \
  --debug
```

## Stage 1: NER Extraction

The same `ner()` function used by Extract mode runs here, but its prompt and JSON Schema are synthesised from the select config at runtime:

- `build_extract_schema_for_select()` -- maps each field to a JSON Schema property (`value_type: "string"` -> `["string", "null"]`, `"array"` -> `["array", "null"]` of strings). All fields are `required`; `additionalProperties` is `false`.
- `build_extract_prompt_for_select()` -- emits a two-message prompt embedding the field list, `prompt_description`s, and the rules below.

### Category Assignment Rules

The Stage 1 user message enforces these domain-agnostic constraints:

- **Output rules** -- JSON-only output, per-`value_type` value handling, prefer exact mentions, no hallucination.
- **Category assignment rules** -- each extracted value belongs to **at most one** category; classify by biological meaning rather than the attribute key (e.g. an attribute labelled `drug` containing `HeLa` is extracted as `cell_line`); experimental control terms (`negative control`, `NC`, `vehicle`, `mock`, `empty vector`, `scramble`, `non-targeting`, `shControl`, `siControl`, ...) are never extracted into any category.

These rules ship as-is so the same prompt builder works for arbitrary select configs, and they stay field-agnostic on purpose. An extraction rule that only fits one field belongs in that field's `prompt_description` instead: instructions shaped for `cell_line` -- strip the prose and the engineering annotations a record wraps a name in -- measurably suppress extraction for `tissue` and `cell_type`, where the recorded value legitimately reads as a description.

## Stage 2: Ontology Search

Two caches are warmed once per process; per-batch work is then just lookups.

- `build_index_map()` -- loads or rebuilds an `OntologyIndex` (word-combination index) per `ontology_file` and persists it under `ontology/index_cache/`.
- `build_text2term_cache()` -- registers each OWL with text2term via `text2term.cache_ontology()` so later `map_terms()` calls skip OWL parsing. Cache location: `ontology/text2term_cache/`.

Both cache directories are configurable via environment variables (see [Configuration](configuration.md#cache)).

### Stage 2a: Word-Combination Search

`OntologyIndex` builds from OWL (via `owlready2`) or TSV/CSV files. It indexes `rdfs:label`, `skos:prefLabel`, and the standard synonym properties (`oboInOwl:hasExactSynonym`, `hasRelatedSynonym`, `hasBroadSynonym`, `hasNarrowSynonym`, `skos:altLabel`, `skos:hiddenLabel`). Additional per-term metadata is collected but not used for matching:

- `obo:IAO_0000115` (textual definition) -- surfaced to Stage 3 as `definitions`.
- `rdfs:comment` -- surfaced to Stage 3 as `comments`. Populated mainly by ChEBI (`has_role` info is injected at build time).

For each extracted value, `build_word_combinations()` generates lower-cased n-grams (NFKC normalised, CamelCase split, alpha/digit boundary split, joined by space and any of `-/_+` present in the query). Each candidate found in the index becomes a `SearchResult`.

Resolution of these candidates is delegated to `_resolve_search_candidates`, which returns one of three verdicts:

| verdict | trigger | downstream effect |
|---|---|---|
| `single` | the exact hits all point to one ontology `term_id` | auto-pick: the term is appended to `results[field]`; Stage 2b and Stage 3 are skipped for that `(field, value)` |
| `ambiguous` | the exact hits span **multiple distinct `term_id`s**, or there are no exact hits at all (only non-exact n-gram subset matches) | the `value` is added to `entry.ambiguous_fields[field]`; Stage 3 must disambiguate |
| `no_match` | no candidates at all | Stage 2b / Stage 3 take over |

An exact hit decides the term. Candidates are filtered down to the exact matches first and only the survivors are checked for agreement, so a non-exact subset hit pointing elsewhere (`PC-3` also surfacing `CVCL:UU13`) does not push the value into Stage 3. Curator-validated labels and synonyms (`rdfs:label`, `skos:prefLabel`, `oboInOwl:has*Synonym`) are authoritative identifiers: `MCF-7` vs `MCF7` or `H9 ESC` vs `WA09` resolve without invoking the LLM. Only when several terms register the same name (`H9` belongs to both `CVCL:1240` and `WA09`) is there nothing to pick, and the verdict falls back to `ambiguous`.

Exactness is measured on the **whole** value -- `exact_match` is `_normalize_key(query) == _normalize_key(annotation_value)`, and word-combination search always looks up the normalised full query alongside its n-grams -- so a value is never auto-picked on the strength of a fragment of itself.

The accepted cost is the parent/derivative case: extraction follows the source text, so a value matching a registered name exactly can still be the parent of the derivative the record actually used (`BLaER` read off a record describing a `BLaER1` derivative). Stage 3 cannot repair that, because the derivative's own name never entered the candidate list; the exact hit is taken and the LLM call is saved.

All species / hierarchy filtering happens at ontology build time -- per-species Cellosaurus OWLs and the CL / UBERON / MONDO / ChEBI subsets are pre-filtered. There is no runtime filter applied to the index.

### Stage 2b: text2term Candidate Supply

For OWL-backed fields, `text2term.map_terms(target_ontology=<acronym>, use_cache=True, cache_folder=BSLLMNER2_TEXT2TERM_CACHE_DIR)` is consulted to broaden the candidate pool with fuzzy (TF-IDF / embedding) matches. The acronym is `{ontology_file.stem}_nofilter` and matches the cache key used in Stage 2a. TSV/CSV ontologies are skipped (text2term operates only on OWL).

Only values left unresolved by Stage 2a are queried, and **resolution is tracked per value, not per field**. An `array` field holds one independent `(value, term_id)` pair per extracted value, so a value auto-picked by the `single` verdict must not stop its siblings from receiving fuzzy candidates -- those siblings are precisely the values that need them, having produced no word-combination hit of their own.

**text2term hits are candidate supply only — they never auto-pick a term, even if the mapping score is 1.0.** A fuzzy matcher cannot tell whether a "perfect" match is genuinely the right term or a coincidental string overlap with an unrelated one, and short cell-line names (`H9`, `NB4`, `697`) attract exactly those neighbours, so the decision is always escalated to Stage 3.

Withholding auto-pick here costs no exact match. A text2term row is dropped unless its mapped label already exists in the word-combination index, so any hit reproducing the value exactly is one Stage 2a has necessarily resolved already.

Failed `text2term` calls are memoised as empty lists so repeated queries do not re-hit the failing call.

## Stage 3: LLM Selection

For every `(field, value)` not resolved by the Stage 2a `single` verdict, candidates from word-combination search **and** text2term are merged (deduplicated by `term_id`, preferring label-property hits). The LLM is asked to pick a single `term_id` from this list (or return `null`) under a strict JSON Schema. Calls are issued via `asyncio.gather` with a 256-way semaphore over the Ollama client (the limit is hard-coded as `OllamaBackend(semaphore_limit=256)` and is not exposed as a CLI flag).

The prompt is engineered to disambiguate by **biological description, not by string similarity**:

- The system message frames the task as comparing each candidate's `label` + `comments` + `definitions` against the BioSample metadata, and instructs the LLM to ignore string resemblance between the input value and a candidate's label.
- `null` is an allowed answer when no candidate is consistent with the BioSample metadata.
- Outside knowledge and term popularity are forbidden; only the provided context and candidate descriptions may inform the decision.
- The prompt contains no field-specific vocabulary (no `cell_line` / `disease` / `tissue` references), so the same instructions apply unchanged when new fields are added.

Each candidate carries `exact_match`, recording whether the input value matched that term's registered label or synonym exactly. Stage 2a already resolves the case where the exact hits agree, so a flagged candidate reaches Stage 3 only when several terms claim the same name -- the flag is what separates those competing identities (`WA09`, which records `H9` as a synonym) from the fuzzy candidates text2term supplies alongside them (`H9 RM`, whose label merely contains the string). The `text2term` similarity score is deliberately withheld, so no other signal of that kind is available. The prompt frames the flag as **evidence, not a decision rule**: a candidate without it can still be correct when the metadata points there.

`--no-reasoning` removes the `reasoning` property from the schema and skips the reasoning instructions in the prompt.

### `ambiguous_fields` (audit trail)

Every `SelectEntry` carries an `ambiguous_fields` map shaped as `{field_name: {ambiguous_value: [distinct_term_ids]}}`. Whenever Stage 2a detects ambiguity for a `(field, value)`, the full list of competing `term_id`s seen by `_resolve_search_candidates` is recorded (sorted for stable diffs).

The picked term lives in `results[field]` together with the LLM's reasoning, so a reviewer can read a single entry and see: which alternatives the LLM had to choose between, which one it picked, and why. Example:

```json
"ambiguous_fields": {
  "cell_line": {"H9": ["CVCL:1240", "CVCL:9773", "CVCL:E9X7"]}
},
"results": {
  "cell_line": [{"value": "H9", "term_id": "CVCL:9773", "reasoning": "..."}]
}
```

## Reselect Mode

`bsllmner2_reselect` repeats Stage 2 and Stage 3 for values that an earlier `bsllmner2_select` run left without a term, and leaves everything else in that run untouched. Use it when the mapping code or an ontology file changed and you want the unresolved values to get a second chance without extracting again.

The input is the `SelectResult` file of the source run (`--from-result`), the BioSample records that run consumed (`--bs-entries`, the Stage 3 prompt needs them as context) and the select config (`--select-config`). `--fields` limits the work to the named fields; by default every field with an `ontology_file` is taken.

For each entry of the source run and each selected field, the target values are the values of `extract.extracted[field]` that do not appear in `results[field]`. An entry without target values is skipped and does not appear in the output. The target values go through the same `select()` pipeline as a normal run, with two restrictions: the select config is cut down to the selected fields, and the `ExtractEntry` holds only the target values. Stage 1 does not run; the extracted values are taken from the source run as they are.

The output is a `SelectResult` whose `entries` are the processed entries in full. For each target `(field, value)` the entry's `search_results`, `text2term_results`, `select_timings` and `ambiguous_fields` are replaced by what the new pass produced, and a resolved value is appended to `results[field]` when Stage 2a or Stage 3 picked a term. Values and fields that were not targets keep their original candidates and results. `run_metadata.reselect` records the source file name, its SHA-256, the source run name and the selected fields (see [Data Formats](data-formats.md#selectresult)).

`bsllmner2_reselect` has no `--resume`. A reselect run is small compared with the source run, so a failed run is repeated from the start. Values that got candidates but no term from the LLM are unresolved again after the run; a second reselect run would ask the LLM about them once more.

```bash
docker compose exec app bsllmner2_reselect \
  --from-result bsllmner2-results/select/select_example.json \
  --bs-entries tests/data/example_biosample.json \
  --select-config scripts/select-config-hg38.json \
  --fields knockout_gene knockdown_gene overexpressed_gene chip_antigen \
  --model llama3.1:70b --run-name example_reselect
```

## SelectConfig Customization

Pre-built configs live in `scripts/`:

| File | Purpose |
|---|---|
| `scripts/select-config-hg38.json` | 9 fields: `cell_line`, `cell_type`, `tissue`, `disease`, `drug`, `knockout_gene`, `knockdown_gene`, `overexpressed_gene`, `chip_antigen` (human ontologies). |
| `scripts/select-config-mm10.json` | Same 9 fields with mouse-specific ontologies. `disease` reuses `mondo_human_subset.owl`. |
| `scripts/select-config-plants.json` | 2 fields: `tissue`, `cell_type` (Plant Ontology). |

To author a custom config:

```json
{
  "fields": {
    "your_field_name": {
      "ontology_file": "ontology/your_subset.owl",
      "prompt_description": "Description used in the Stage 1 prompt.",
      "value_type": "string"
    }
  }
}
```

| Property | Type | Default | Description |
|---|---|---|---|
| `ontology_file` | `string` or `null` | `null` | Path to an OWL or TSV/CSV ontology. If `null`, the extracted value is recorded as-is (no Stage 2/3 mapping). |
| `prompt_description` | `string` or `null` | `null` | Per-field description injected into the Stage 1 prompt. |
| `value_type` | `"string"` or `"array"` | `"string"` | `"array"` produces a list of values for the field. |

See [Data Formats](data-formats.md#selectconfig) for the canonical schema.

## Output

Result file: `bsllmner2-results/select/select_{run_name}.json` (and per-batch resume files while running). For the full `SelectResult` schema, see [Data Formats](data-formats.md#selectresult).
