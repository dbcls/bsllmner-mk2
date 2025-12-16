# mm10 memo

```bash
mv chip-atlas-data/bs_entries.jsonl chip-atlas-data/bs_entries_hg38.jsonl
mv chip-atlas-data/experimentList.json chip-atlas-data/experimentList_hg38.json
mv chip-atlas-data/srx_to_biosample.json chip-atlas-data/srx_to_biosample_hg38.json
python3 scripts/download_ontology_files.py --genome-assembly mm10
```

```bash
$ cp ./scripts/select-config.json ./scripts/select-config-mm10.json
$ rm -rf ./ontology/index_cache/cellosaurus.owl.pkl
$ rm -rf ./ontology/index_cache/mondo.owl.pkl
$ ls ./ontology/index_cache/
cell_ontology.owl.pkl  chebi.owl.pkl  ncbi_gene_human.owl.pkl  uberon.owl.pkl
```

```bash
bsllmner2_select \
  --debug \
  --bs-entries ./chip-atlas-data/bs_entries.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-mm10.json \
  --max-entries 500 \
  --run-name mm10-test
```

```bash
$ wc -l bs_entries.jsonl
188122 bs_entries.jsonl
root@45483b996e76:/app/chip-atlas-data# awk 'NR % 350 == 1' bs_entries.jsonl > bs_entries.small.jsonl
root@45483b996e76:/app/chip-atlas-data# wc -l bs_entries.small.jsonl 
538 bs_entries.small.jsonl
```

```bash
bsllmner2_select \
  --debug \
  --bs-entries ./chip-atlas-data/bs_entries.small.jsonl \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-mm10.json \
  --max-entries 500 \
  --run-name mm10-test-small
```
