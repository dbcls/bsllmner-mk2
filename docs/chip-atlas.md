# ChIP-Atlas Data Processing

This guide explains how to process ChIP-Atlas data using bsllmner-mk2 for both human (hg38) and mouse (mm10) genome assemblies.

## Overview

### What is ChIP-Atlas?

[ChIP-Atlas](https://chip-atlas.org) is a comprehensive data-mining suite for exploring epigenomic landscapes by fully integrating ChIP-seq, ATAC-seq, DNase-seq, and Bisulfite-seq experiments.

### Relationship with bsllmner-mk2

- Each SRX (experiment) entry in ChIP-Atlas is linked to a BioSample record
- SRX and BioSample have a one-to-one relationship
- ChIP-Atlas provides human-curated metadata values for each experiment
- bsllmner-mk2 can extract and map values from BioSample records using LLM
- This enables benchmarking LLM-based NER against human-curated annotations

## Prerequisites

Before processing ChIP-Atlas data, ensure:

1. Docker environment is running (see [getting-started.md](getting-started.md))
2. Ontology files are downloaded and converted (see [Quick Start §2](getting-started.md#2-download-ontology-files))

## Data Preparation

### scripts/prepare_bs_entries.py

This script downloads and prepares BioSample entries from ChIP-Atlas:

```bash
# Inside Docker container
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly <GENOME>
```

**Options:**

- `--genome-assembly`: Filter by genome assembly (e.g., `hg38`, `mm10`)
- `--force`: Re-download files even if they already exist

**Output Files** (in `chip-atlas-data/`):

| File | Description |
|------|-------------|
| `experimentList.tab` | Raw metadata from ChIP-Atlas |
| `experimentList.json` | Parsed experiment metadata |
| `SRA_Accessions.tab` | SRX to BioSample mapping source |
| `srx_to_biosample.json` | SRX to BioSample ID mapping |
| `bs_entries.jsonl` | BioSample entries (one per line) |
| `bs_entries/{prefix}/{accession}.json` | Cached individual BioSample files |

## Select Configuration

### Configuration Structure

The select config JSON file defines which fields to extract and map to ontologies:

```json
{
  "fields": {
    "field_name": {
      "ontology_file": "/app/ontology/example.owl",
      "prompt_description": "Description for LLM",
      "ontology_filter": {
        "hasDbXref": "NCBI_TaxID:9606"
      },
      "value_type": "string"
    }
  }
}
```

For field property details, see [Select Mode - Select Config Customization](select-mode.md#select-config-customization).

### Provided Configuration Files

| File | TaxID | Fields | Use Case |
|------|-------|--------|----------|
| `select-config-hg38.json` | 9606 | 8 (cell_line, cell_type, tissue, disease, drug, knockout_gene, knockdown_gene, overexpressed_gene) | Human ChIP-Atlas evaluation |
| `select-config-mm10.json` | 10090 | 8 (cell_line, cell_type, tissue, disease, drug, knockout_gene, knockdown_gene, overexpressed_gene) | Mouse ChIP-Atlas evaluation |

**Note:** You should customize the select-config based on your specific needs.

### Key Differences Between hg38 and mm10 Configs

| Aspect | hg38 | mm10 |
|--------|------|------|
| `cell_line` ontology | `cellosaurus.owl` + `ontology_filter: NCBI_TaxID:9606` | `cellosaurus.owl` + `ontology_filter: NCBI_TaxID:10090` |
| `cell_type` ontology | `cl_human_subset.owl` (CL human_subset + EFO cell types) | `cl_mouse_subset.owl` (CL mouse_subset + EFO cell types) |
| `tissue` ontology | `uberon_human_subset.owl` | `uberon_mouse_subset.owl` |
| `disease` ontology | `mondo_human_subset.owl` (`MONDO:0700096` subtree) | `mondo.owl` (full MONDO — no mouse subset query upstream) |
| `drug` ontology | `chebi_subset.owl` | `chebi_subset.owl` |
| `knockout/down/overexpressed_gene` ontology | `ncbi_gene_human.owl` | `ncbi_gene_mouse.owl` |

For CL / UBERON / ChEBI / MONDO the subset OWLs encode the species / hierarchy filter at build time, so no runtime `ontology_filter` is needed. `ontology_filter` is retained only for Cellosaurus (per-species tax ID).

## Processing hg38 (Human)

### 1. Prepare Data

```bash
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly hg38
```

This downloads and processes human experiments from ChIP-Atlas.

### 2. Run Select Mode

```bash
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-hg38.json \
  --run-name hg38-full \
  --debug
```

### 3. Check Results

```bash
ls bsllmner2-results/extract/
ls bsllmner2-results/select/
```

## Processing mm10 (Mouse)

### 1. Backup Existing Data (if switching from hg38)

```bash
# Optional: backup hg38 data before overwriting
mv chip-atlas-data/bs_entries.jsonl chip-atlas-data/bs_entries_hg38.jsonl
mv chip-atlas-data/experimentList.json chip-atlas-data/experimentList_hg38.json
mv chip-atlas-data/srx_to_biosample.json chip-atlas-data/srx_to_biosample_hg38.json
```

### 2. Prepare Data

```bash
docker compose exec app python3 scripts/prepare_bs_entries.py --genome-assembly mm10
```

### 3. Run Select Mode

```bash
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-mm10.json \
  --run-name mm10-full \
  --debug
```

## Large-Scale Processing Tips

### Test with Limited Entries

Before processing the full dataset, test with a subset:

```bash
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --select-config ./scripts/select-config-hg38.json \
  --model llama3.1:70b \
  --max-entries 100 \
  --run-name hg38-test
```

### Resume Interrupted Processing

If processing is interrupted, resume from where it left off:

```bash
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --select-config ./scripts/select-config-hg38.json \
  --model llama3.1:70b \
  --run-name hg38-full \
  --resume
```

### Adjust Batch Size

If you encounter memory issues, reduce the batch size:

```bash
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --select-config ./scripts/select-config-hg38.json \
  --model llama3.1:70b \
  --batch-size 256 \
  --run-name hg38-full
```

Default batch size is 1024.

### Sample Data for Quick Testing

Create a smaller dataset by sampling:

```bash
# Sample every 350th entry (reduces ~188k to ~500 entries)
awk 'NR % 350 == 1' chip-atlas-data/bs_entries.jsonl > chip-atlas-data/bs_entries.small.jsonl

# Run on sampled data
docker compose exec app bsllmner2_select \
  --bs-entries ./chip-atlas-data/bs_entries.small.jsonl \
  --select-config ./scripts/select-config-mm10.json \
  --model llama3.1:70b \
  --run-name mm10-test-small
```

## Troubleshooting

### Out of Memory Errors

**Symptom:** Container crashes during processing

**Solutions:**

1. Reduce `--batch-size` (e.g., `--batch-size 128`)
2. Reduce Ollama parallel requests in `compose.yml`:

   ```yaml
   environment:
     - OLLAMA_NUM_PARALLEL=8  # reduce from 16
   ```

3. Use a smaller model (e.g., `llama3.1:8b` instead of `llama3.1:70b`)

### Data Download Failures

**Symptom:** Network errors during `prepare_bs_entries.py`

The script uses the DDBJ Search Bulk API to fetch BioSample entries in batches of 1000. If a batch fails, it retries up to 3 times with exponential backoff. Successfully cached entries are skipped on re-run.

**Solution:**

```bash
# Re-run with --force to retry failed downloads
docker compose exec app python3 scripts/prepare_bs_entries.py \
  --genome-assembly hg38 \
  --force
```

### Missing Ontology Files

**Symptom:** FileNotFoundError for `.owl` files

**Solution:**

1. Run the download script: `python3 scripts/download_ontology_files.py`
2. Convert Cellosaurus OBO to OWL (see Prerequisites section)

## Data Volume Reference

Approximate data sizes by genome assembly:

| Assembly | Experiments | BioSample Entries |
|----------|-------------|-------------------|
| hg38 | ~200,000+ | ~150,000+ |
| mm10 | ~188,000 | ~140,000 |

Processing times vary based on:

- Model size (larger models = slower but more accurate)
- GPU performance
- Number of parallel requests (`OLLAMA_NUM_PARALLEL`)
- Batch size
