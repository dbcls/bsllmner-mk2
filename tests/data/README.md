# Test Data

## License

All data in this directory is licensed under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## Files

### CLI example data (10 entries)

| File | Description |
|------|-------------|
| `example_biosample.json` | 10 BioSample entries (subset of the 600-entry evaluation dataset) |
| `example_gold_standard.tsv` | 10-row human-curated gold standard (subset of `eval_gold_standard.tsv`) |

Used in CLI examples throughout the documentation (README, Getting Started, Extract Mode, Select Mode, NIG Slurm).

### Evaluation dataset (600 entries)

| File | Description |
|------|-------------|
| `eval_biosample.json` | 600 BioSample entries for model evaluation |
| `eval_gold_standard.tsv` | 600-row human-curated gold standard for cell line → Cellosaurus ontology mapping |

Used by `tests/model-evaluation/` for benchmarking LLM models.

Source: [Zenodo 14881142](https://zenodo.org/records/14881142), [Zenodo 14643285](https://zenodo.org/records/14643285)

### Test fixtures

| File | Description |
|------|-------------|
| `test.owl` | Minimal OWL ontology with 3 classes (Alpha, Beta, Gamma Cell) for unit testing `build_index_from_owl` |

## Gold standard TSV format

Both `example_gold_standard.tsv` and `eval_gold_standard.tsv` share the same 5-column TSV format:

| Column | Description |
|--------|-------------|
| `BioSample ID` | BioSample accession (e.g., `SAMD00011704`) |
| `Experiment type` | Experiment category (e.g., `Histone`, `ATAC-Seq`) |
| `extraction answer` | Human-curated extracted entity name (e.g., `HEK293`). Empty if no entity present. |
| `mapping answer ID` | Cellosaurus ontology ID (e.g., `CVCL:0045`). Empty if no mapping exists. |
| `mapping answer label` | Human-readable label for the ontology term (e.g., `HEK293`). Empty if no mapping exists. |

Rows where both `mapping answer ID` and `mapping answer label` are empty indicate BioSample entries where no cell line mapping exists in the ground truth.
