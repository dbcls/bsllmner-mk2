# Getting Started

This guide walks you through running bsllmner-mk2 for the first time.

For installation details (Docker Compose, uv, GPU configuration), see [Installation](installation.md).

## 1. Start the Service

```bash
docker compose up -d --build
```

If this is your first time, complete the full setup in [Installation](installation.md) first.

## 2. Prepare Ontology Files

Select mode (Stage 2) uses **pre-subsetted** OWL files that expose only the properties the LLM needs (`rdfs:label`, various synonyms, `obo:IAO_0000115` definition, and `rdfs:comment` for ChEBI `has_role`). The `ontology/` directory is bind-mounted into the container.

### 2.1 Download Upstream OWL Sources

```bash
docker compose exec app python3 scripts/download_ontology_files.py
```

This fetches the following upstream files to `ontology/`:

- `cellosaurus.obo` - Cell line database (preprocessed per species in §2.2)
- `cl.owl` - full Cell Ontology (input for CL human/mouse subset)
- `efo.owl` - Experimental Factor Ontology (merged into the CL subset as additional cell types under `EFO:0000324`)
- `uberon.owl` - full UBERON anatomy ontology (input for UBERON human/mouse subset)
- `mondo.owl` - full MONDO disease ontology
- `chebi.owl` - full ChEBI chemical entities
- `po.owl` - full Plant Ontology (input for PO plant tissue / cell subsets)

### 2.2 Preprocess Cellosaurus (per species)

Cellosaurus arrives in OBO format. Run the preprocessor once per taxon; it filters
terms by `NCBI_TaxID`, preserves `Disease` / `derived_from` as `rdfs:comment`, and
synthesizes a one-line `def:` (IAO_0000115 textual definition) from Category /
Sex / Species of origin / Disease / Derived from so the Stage 3 LLM has richer
context for each cell-line candidate.

```bash
docker compose exec app python3 scripts/preprocess_cellosaurus.py --taxid 9606    # -> ontology/cellosaurus_human.mod.obo
docker compose exec app python3 scripts/preprocess_cellosaurus.py --taxid 10090   # -> ontology/cellosaurus_mouse.mod.obo
```

Then convert each `.mod.obo` to OWL with ROBOT:

```bash
docker run -v $PWD/ontology:/work -w /work --rm obolibrary/robot \
    robot convert -i cellosaurus_human.mod.obo -o cellosaurus_human.owl --format owl
docker run -v $PWD/ontology:/work -w /work --rm obolibrary/robot \
    robot convert -i cellosaurus_mouse.mod.obo -o cellosaurus_mouse.owl --format owl
```

### 2.3 Build Subset Ontologies

Generate subset OWL files for CL / UBERON / ChEBI / MONDO / PO. This runs `sh-ikeda/ontology-constructor-for-bsllmner` SPARQL templates through ROBOT (Docker), then applies a post-processing step to add `rdf:type owl:Class` so `owlready2` can load them.

```bash
bash scripts/build_subset_ontologies.sh
```

Outputs (all under `ontology/`):

- `cl_human_subset.owl`, `cl_mouse_subset.owl` (CL `{human,mouse}_subset` merged with EFO cell types under `EFO:0000324`)
- `uberon_human_subset.owl`, `uberon_mouse_subset.owl`
- `chebi_subset.owl` (has-role info injected into `rdfs:comment`; the ChEBI update step needs `ROBOT_JAVA_ARGS="-Xmx24g"` because the upstream `chebi.owl` is large)
- `mondo_human_subset.owl` (mm10 reuses the same subset — mouse-model diseases are overwhelmingly human diseases, so no separate `mondo_mouse_subset.owl` is built)
- `po_tissue_subset.owl`, `po_cell_subset.owl` (PO plant tissue / cell subtrees; `po.owl` is preprocessed with `po_edit.awk` to strip German/Japanese/Spanish synonyms and `owl:Axiom` blocks before the SPARQL CONSTRUCT)

### 2.4 Generate NCBI Gene OWL

`scripts/ncbi_gene_to_owl.py` converts NCBI's `gene_info` TSV into a per-taxon OWL file, storing the gene description (9th column) as `obo:IAO_0000115`:

```bash
# gene_info must be downloaded separately from https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
docker compose exec app python3 scripts/ncbi_gene_to_owl.py --taxid 9606   # -> ontology/ncbi_gene_human.owl
docker compose exec app python3 scripts/ncbi_gene_to_owl.py --taxid 10090  # -> ontology/ncbi_gene_mouse.owl
```

### 2.5 (Optional) Clear Stale Caches

Select mode uses two on-disk caches:

- **`ontology/index_cache/`** — word-combination search index (serialized `OntologyIndex`). Files are named `{ontology_file_name}_nofilter_v2.pkl`. The runtime ontology filter feature has been retired; the constant `_nofilter` suffix is kept so on-disk cache names remain stable with past runs. The `_v2` suffix indicates the on-disk format version; older entries are simply ignored when the format changes.
- **`ontology/text2term_cache/`** — text2term prebuilt ontology cache. Each OWL is cached under an acronym of the form `{ontology_file_stem}_nofilter`, so per-batch `text2term.map_terms()` can reuse the parsed ontology across runs instead of re-parsing the OWL. Override the location with `BSLLMNER2_TEXT2TERM_CACHE_DIR` (see [configuration.md](configuration.md#cache)).

Stale entries are ignored automatically when the key changes. If disk usage is a concern, remove them manually:

```bash
rm -rf ontology/index_cache/ ontology/text2term_cache/
```

## 3. (Optional) Pre-pull LLM Model

The LLM model is automatically downloaded on first use via the Ollama API. No manual pull is required.

To pre-download the model before running (recommended for large models like 70b):

```bash
docker compose exec ollama ollama pull llama3.1:70b
```

The model (~40GB for 70b) is stored in the `ollama-data/` directory.

## 4. Run Extract Mode

Extract mode performs Named Entity Recognition (NER) to extract biological terms from BioSample records.

```bash
docker compose exec app bsllmner2_extract \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --debug
```

For all extract CLI options, see [Extract Mode](extract-mode.md#cli-options).

## 5. Run Select Mode

Select mode extends extract mode by mapping extracted terms to ontology entries.

```bash
docker compose exec app bsllmner2_select \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --select-config scripts/select-config-hg38.json \
  --debug
```

For all select CLI options, see [Select Mode](select-mode.md#cli-options).

## 6. Inspect Results

Results are saved in `bsllmner2-results/`:

```bash
# List result files
ls bsllmner2-results/extract/
ls bsllmner2-results/select/
```

View the extract result:

```bash
# Show run metadata
jq '.run_metadata' bsllmner2-results/extract/*.json

# Show extracted values
jq '.output[] | {accession, output}' bsllmner2-results/extract/*.json
```

View the select result:

```bash
# Show mapped ontology terms
jq '.[0].results' bsllmner2-results/select/*.json
```

### Result Structure

- **Extract results**: `bsllmner2-results/extract/{run_name}.json`
  - Contains extracted entities and metadata

- **Select results**: `bsllmner2-results/select/select_{run_name}.json`
  - Contains ontology-mapped results for each field

For the full result schema, see [Data Formats](data-formats.md).

## Next Steps

- **ChIP-Atlas data processing**: See [chip-atlas.md](chip-atlas.md) for processing ChIP-Atlas data with hg38/mm10
- **Model evaluation**: See [tests/model-evaluation/README.md](../tests/model-evaluation/README.md) for benchmarking different LLM models
- **Custom configuration**: Create your own select config JSON to customize field extraction and ontology mapping
