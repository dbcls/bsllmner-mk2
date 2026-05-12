# ChIP-Atlas Data Processing

How to fetch ChIP-Atlas experiments and run Select mode against them for human (hg38) and mouse (mm10) assemblies.

## Overview

[ChIP-Atlas](https://chip-atlas.org) is a data-mining suite covering ChIP-seq, ATAC-seq, DNase-seq, and Bisulfite-seq experiments. Each SRX entry maps one-to-one to a BioSample record. ChIP-Atlas itself provides human-curated metadata, which makes it a useful benchmark target for the LLM-based extraction in `bsllmner2_select`.

## Prerequisites

1. The Docker environment is up (see [Installation](installation.md)).
2. Ontology files have been prepared (see [Ontology Preparation](ontology.md)).

## Data Preparation

`scripts/prepare_bs_entries.py` downloads ChIP-Atlas `experimentList.tab`, builds an SRX-to-BioSample mapping from NCBI `SRA_Accessions.tab`, and fetches the corresponding BioSample entries through the DDBJ Search Bulk API.

```bash
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly <GENOME>
```

| Option | Description |
|---|---|
| `--genome-assembly` | Filter experiments by genome assembly (e.g. `hg38`, `mm10`). |
| `--force` | Re-download files even if cached copies exist. |

Output under `chip-atlas-data/`:

| File | Description |
|---|---|
| `experimentList.tab` | Raw ChIP-Atlas metadata. |
| `experimentList.json` | Parsed `ChipAtlasExperiment` list. |
| `SRA_Accessions.tab` | NCBI SRA accession mapping source. |
| `srx_to_biosample.json` | SRX -> BioSample ID mapping. |
| `bs_entries.jsonl` | BioSample entries (one JSON per line). |
| `bs_entries/{prefix}/{accession}.json` | Per-accession cache. |

DDBJ Bulk API calls are batched in groups of 1000 and retried up to 3 times with exponential backoff. Cached entries are reused on re-runs unless `--force` is given.

## Provided Select Configs

| File | Taxonomy | Fields |
|---|---|---|
| `scripts/select-config-hg38.json` | 9606 (human) | `cell_line`, `cell_type`, `tissue`, `disease`, `drug`, `knockout_gene`, `knockdown_gene`, `overexpressed_gene` |
| `scripts/select-config-mm10.json` | 10090 (mouse) | Same 8 fields, with mouse ontologies. |

For the SelectConfig schema itself, see [Select Mode](select-mode.md#selectconfig-customization). To customise: copy one of the files above and edit the field list / ontology paths.

### Field Comparison

| Field | hg38 ontology | mm10 ontology |
|---|---|---|
| `cell_line` | `cellosaurus_human.owl` | `cellosaurus_mouse.owl` |
| `cell_type` | `cl_human_subset.owl` | `cl_mouse_subset.owl` |
| `tissue` | `uberon_human_subset.owl` | `uberon_mouse_subset.owl` |
| `disease` | `mondo_human_subset.owl` | `mondo_human_subset.owl` (reused) |
| `drug` | `chebi_subset.owl` | `chebi_subset.owl` |
| `knockout_gene` / `knockdown_gene` / `overexpressed_gene` | `ncbi_gene_human.owl` | `ncbi_gene_mouse.owl` |

## Processing hg38

```bash
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly hg38

docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-hg38.json \
  --run-name hg38-full \
  --debug
```

Result: `bsllmner2-results/select/select_hg38-full.json`.

## Processing mm10

```bash
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly mm10

docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-mm10.json \
  --run-name mm10-full \
  --debug
```

`prepare_bs_entries.py` overwrites `chip-atlas-data/bs_entries.jsonl` and the related JSON files. To keep the previous assembly's output, rename the files before the second run (e.g. `mv chip-atlas-data/bs_entries.jsonl chip-atlas-data/bs_entries_hg38.jsonl`).

## Tips for Large-Scale Runs

- **Test first**: use `--max-entries 100 --run-name <run>-test` to validate the pipeline end-to-end before launching the full run.
- **Sample**: `awk 'NR % 350 == 1' chip-atlas-data/bs_entries.jsonl > chip-atlas-data/bs_entries.small.jsonl` gives ~500 entries.
- **Resume**: keep the same `--run-name` and add `--resume`. See [CLI Reference](cli.md#resume).
- **Batch size**: lower `--batch-size` (default 1024) if VRAM is tight.

Approximate data volumes (as observed at the time of writing; ChIP-Atlas keeps growing):

| Assembly | Experiments | BioSample entries |
|---|---|---|
| hg38 | ~200,000+ | ~150,000+ |
| mm10 | ~188,000 | ~140,000 |

## Troubleshooting

**Out of memory.** Reduce `--batch-size` (e.g. 128), drop `OLLAMA_NUM_PARALLEL` in `compose.yml`, or run a smaller model. See [Configuration](configuration.md#ollama-performance-tuning-docker-compose).

**Bulk API failures.** Re-run with `--force` to retry. The script retries each batch up to 3 times with exponential backoff; cached entries are reused across attempts.

**Missing ontology files.** Build them following [Ontology Preparation](ontology.md). For mm10 specifically you need `cellosaurus_mouse.owl`, `cl_mouse_subset.owl`, `uberon_mouse_subset.owl`, and `ncbi_gene_mouse.owl`; `mondo_human_subset.owl` is reused.
