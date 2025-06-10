# bsllmner-mk2

- [the original](https://github.com/sh-ikeda/bsllmner)
- [paper](https://doi.org/10.1101/2025.02.17.638570)

## Development

```bash
mkdir ollama-data
chmod 777 ollama-data
docker compose up -d --build
```

## Memo

- まず mode extract から
  1. BsLlmProcess で、self に格納
  2. self.llm_input_json = self.construct_llm_input_json()
     1. bs json から、filter_key val で key を制限する
