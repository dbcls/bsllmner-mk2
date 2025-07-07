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
docker compose exec app bsllmner2_extract --debug --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv --with-metrics
docker compose exec app bsllmner2_extract --debug --model deepseek-r1:70b --bs-entries ./tests/test-data/cell_line_example.biosample.json --mapping ./tests/test-data/cell_line_example.mapping.tsv
```

- Test data
  - <https://zenodo.org/records/14881142>
  - <https://zenodo.org/records/14643285>
