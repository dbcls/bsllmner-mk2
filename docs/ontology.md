# Ontology Preparation

[Select Mode](select-mode.md) consumes pre-built ontology OWLs from `ontology/`. This page walks through every step from downloading upstream sources to clearing stale caches.

## Overview

The pipeline produces the OWLs listed below. All commands assume the project root as the working directory and run inside the `app` container (`docker compose exec app ...`). Subset ontologies are built once and reused across runs.

| Output | Source | Generator |
|---|---|---|
| `ontology/cellosaurus_human.owl` | `cellosaurus.obo` (taxid 9606) | `preprocess_cellosaurus.py` + ROBOT |
| `ontology/cellosaurus_mouse.owl` | `cellosaurus.obo` (taxid 10090) | `preprocess_cellosaurus.py` + ROBOT |
| `ontology/cl_human_subset.owl` | `cl.owl` + `efo.owl` | `build_subset_ontologies.sh` |
| `ontology/cl_mouse_subset.owl` | `cl.owl` + `efo.owl` | `build_subset_ontologies.sh` |
| `ontology/uberon_human_subset.owl` | `uberon.owl` | `build_subset_ontologies.sh` |
| `ontology/uberon_mouse_subset.owl` | `uberon.owl` | `build_subset_ontologies.sh` |
| `ontology/mondo_human_subset.owl` | `mondo.owl` | `build_subset_ontologies.sh` |
| `ontology/chebi_subset.owl` | `chebi.owl` | `build_subset_ontologies.sh` |
| `ontology/po_tissue_subset.owl` | `po.owl` | `build_subset_ontologies.sh` |
| `ontology/po_cell_subset.owl` | `po.owl` | `build_subset_ontologies.sh` |
| `ontology/ncbi_gene_human.owl` | `gene_info` (taxid 9606) | `ncbi_gene_to_owl.py` |
| `ontology/ncbi_gene_mouse.owl` | `gene_info` (taxid 10090) | `ncbi_gene_to_owl.py` |

Prerequisites: see [Installation](installation.md). In particular, `build_subset_ontologies.sh` requires Docker access to `obolibrary/robot:latest` and a host-side `gawk` for the Plant Ontology preprocess step.

## 1. Download Upstream OWL

```bash
docker compose exec app python3 scripts/download_ontology_files.py
```

Fetches `cellosaurus.obo`, `cl.owl`, `efo.owl`, `uberon.owl`, `mondo.owl`, `chebi.owl`, and `po.owl` into `ontology/`. Existing files are skipped.

## 2. Preprocess Cellosaurus

`preprocess_cellosaurus.py` filters Cellosaurus to one NCBI taxonomy ID and synthesizes a single-line `def:` annotation (from Category / Sex / Species / Disease / Derived from) so ROBOT emits an `IAO_0000115` textual definition.

```bash
docker compose exec app python3 scripts/preprocess_cellosaurus.py --taxid 9606    # human
docker compose exec app python3 scripts/preprocess_cellosaurus.py --taxid 10090   # mouse
```

Output: `ontology/cellosaurus_{human,mouse}.mod.obo`.

Convert OBO to OWL with the ROBOT Docker image:

```bash
docker run --rm -v $PWD/ontology:/work -w /work obolibrary/robot \
  robot convert -i cellosaurus_human.mod.obo -o cellosaurus_human.owl --format owl
docker run --rm -v $PWD/ontology:/work -w /work obolibrary/robot \
  robot convert -i cellosaurus_mouse.mod.obo -o cellosaurus_mouse.owl --format owl
```

## 3. Build Subset Ontologies (CL / UBERON / MONDO / ChEBI / PO)

```bash
bash scripts/build_subset_ontologies.sh           # build only what is missing
bash scripts/build_subset_ontologies.sh --force   # regenerate everything
```

The script clones [`sh-ikeda/ontology-constructor-for-bsllmner`](https://github.com/sh-ikeda/ontology-constructor-for-bsllmner) into `work/` and runs SPARQL CONSTRUCT queries through `obolibrary/robot:latest`. Notes:

- `chebi_subset.owl` runs ROBOT with a 24 GB Java heap (`ROBOT_JAVA_ARGS="-Xmx24g"`); ensure the host has that much RAM available to Docker.
- The ChEBI build (`build_chebi_subset` in `scripts/build_subset_ontologies.sh`) folds each term's `has_role` chain into `rdfs:comment` via the `chebi_update.rq` / `chebi_construct.rq` SPARQL queries in `ontology-constructor-for-bsllmner/chebi/`. [Select Mode](select-mode.md#stage-2a-word-combination-search) surfaces this comment text to the Stage 3 LLM as additional context.
- The Plant Ontology preprocess uses GNU awk's `gensub()` and is run on the host (`gawk -f work/ontology-constructor-for-bsllmner/po/po_edit.awk`). The ROBOT image ships only `mawk`, which lacks `gensub()`.
- `mondo_human_subset.owl` is reused for the mm10 select config; there is no `mondo_mouse_subset.owl`.
- `po_tissue_subset.owl` is consumed by the `tissue` field and `po_cell_subset.owl` by `cell_type` in `select-config-plants.json`.

## 4. Generate NCBI Gene OWL

```bash
# Fetch gene_info (one-time)
curl -L -o ontology/gene_info.gz https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
gunzip ontology/gene_info.gz

# Build per-species OWL
docker compose exec app python3 scripts/ncbi_gene_to_owl.py --taxid 9606    # ncbi_gene_human.owl
docker compose exec app python3 scripts/ncbi_gene_to_owl.py --taxid 10090   # ncbi_gene_mouse.owl
```

Each gene becomes an `owl:Class` with `rdfs:label` (symbol), `oboInOwl:hasExactSynonym` (pipe-separated synonyms), and `obo:IAO_0000115` (description).

## Cache Layout

Select mode caches expensive parses to avoid re-loading large OWLs every batch. Locations are controlled by environment variables (see [Configuration](configuration.md#cache)).

| Directory (default) | Purpose | File naming |
|---|---|---|
| `ontology/index_cache/` | Pickled `OntologyIndex` per ontology. | `{ontology_file.name}_nofilter_v2.pkl` |
| `ontology/text2term_cache/` | text2term ontology cache. | acronym `{ontology_file.stem}_nofilter` |

The `_nofilter` suffix and `_v2` format tag are constants in the current implementation. When an OWL is rebuilt, delete its cache entries so they are regenerated on the next run.

## Cache Cleanup

```bash
rm -rf ontology/index_cache ontology/text2term_cache
```

Both directories are rebuilt automatically on the next `bsllmner2_select` run. Building the text2term cache is the slowest step (proportional to the OWL size); rebuilds only happen for ontologies whose cache entry is missing.
