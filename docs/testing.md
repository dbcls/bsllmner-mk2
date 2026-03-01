# Testing

## Running Unit Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/py_tests/test_utils.py

# Exclude slow tests
uv run pytest -m "not slow"

# Run with randomized order (enabled by pytest-randomly)
uv run pytest -p randomly
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Long-running tests, skipped with `-m "not slow"` |

## Type Checking

```bash
uv run mypy
```

Configured in `pyproject.toml` with strict mode and the pydantic plugin.

## Linting and Formatting

```bash
# Lint
uv run ruff check bsllmner2/ tests/ scripts/

# Format
uv run ruff format bsllmner2/ tests/ scripts/

# Format check (for CI)
uv run ruff format --check bsllmner2/ tests/ scripts/
```

## Mutation Testing

[mutmut](https://github.com/boxed/mutmut) validates that tests can detect code mutations.

```bash
# Run mutation testing
uv run mutmut run

# Show results
uv run mutmut results
```

Target modules are configured in `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = "bsllmner2/"
tests_dir = "tests/py_tests/"
```

## Model Evaluation

The `tests/model-evaluation/` directory contains scripts for benchmarking LLM models on ontology mapping accuracy.

**Datasets** (hosted on Zenodo):

- <https://zenodo.org/records/14881142>
- <https://zenodo.org/records/14643285>

600 BioSample entries evaluated against a human-curated gold standard.

**Evaluated models**: deepseek-r1 (8b/32b), gemma3 (4b/12b/27b), gpt-oss (20b), llama3.1 (8b), phi4 (14b), qwen3 (4b/8b/32b)

**Metrics**: Precision, Recall, F1-score, Accuracy (for the `cell_line` field)

For the full evaluation workflow (batch execution, metric computation, result aggregation), see [tests/model-evaluation/README.md](../tests/model-evaluation/README.md).
