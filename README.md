# bsllmner-mk2

- [the original](https://github.com/sh-ikeda/bsllmner)
- [paper](https://doi.org/10.1101/2025.02.17.638570)

## Development

```bash
mkdir ollama-data
chmod 777 ollama-data
docker network create bsllmner-mk2-network
docker compose up -d --build
```

```bash
docker compose exec ollama ollama pull llama3.1:70b
docker compose exec api bsllmner2_extract --debug --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv --with-metrics
docker compose exec api bsllmner2_extract --debug --model deepseek-r1:70b --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv

docker compose exec api bsllmner2_extract --debug --model deepseek-r1:70b --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv
```

## Large Data

- tests/zenodo-data/biosample_cellosaurus_mapping_testset.json
- tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv
  - 600サンプル

```bash
docker compose exec api bsllmner2_extract --debug --model qwen3:32b --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv --with-metrics

docker compose exec api bsllmner2_extract --debug --model phi4:14b --bs-entries ./tests/zenodo-data/biosample_cellosaurus_mapping_testset.json --mapping ./tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv

docker compose exec api bsllmner2_extract --debug --model llama3.1:70b --bs-entries ./tests/zenodo-data/biosample_cellosaurus_mapping_testset.json --mapping ./tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv
```

- Test data
  - <https://zenodo.org/records/14881142>
  - <https://zenodo.org/records/14643285>
