# Testing

## Unit Tests

```bash
uv run pytest                          # all tests, with coverage
uv run pytest tests/py_tests/test_io.py
uv run pytest -m "not slow"            # skip slow tests
uv run pytest -p randomly              # explicit randomised order
```

Pytest is configured in `pyproject.toml` with `--cov=bsllmner2 --cov-report=term-missing`, strict markers, and async support. Property-based tests are written with [hypothesis](https://hypothesis.readthedocs.io/).

### Markers

| Marker | Description |
|---|---|
| `slow` | Long-running tests. Skip with `-m "not slow"`. |

## Type Checking

```bash
uv run mypy
```

`mypy` runs in `strict` mode with the `pydantic.mypy` plugin against `bsllmner2/**/*.py` and `tests/**/*.py`. External modules without stubs have their missing-imports ignored via `[[tool.mypy.overrides]]` in `pyproject.toml`.

## Linting and Formatting

```bash
uv run ruff check bsllmner2/ tests/ scripts/
uv run ruff format bsllmner2/ tests/ scripts/
uv run ruff format --check bsllmner2/ tests/ scripts/
```

Ruff is configured with `select = ["ALL"]` and an explicit `ignore` list (see `[tool.ruff.lint]` in `pyproject.toml`). `target-version = "py310"` and `line-length = 120`.

## Mutation Testing

[mutmut](https://github.com/boxed/mutmut) validates that the test suite detects code mutations.

```bash
uv run mutmut run
uv run mutmut results
```

Configured in `pyproject.toml`:

```toml
[tool.mutmut]
paths_to_mutate = ["bsllmner2/"]
tests_dir = ["tests/py_tests/"]
pytest_add_cli_args = ["--no-cov", "-p", "no:randomly"]
```

## Model Evaluation

`tests/model-evaluation/` benchmarks LLM models on the ontology mapping task using the 600-entry evaluation set in `tests/data/eval_biosample.json` with `tests/data/eval_gold_standard.tsv` as the gold standard for the `cell_line` field.

Source datasets (Zenodo):

- <https://zenodo.org/records/14881142>
- <https://zenodo.org/records/14643285>

Metrics produced: accuracy, precision, recall, F1 (see `SelectResult.evaluation` in [Data Formats](data-formats.md#selectresult)). For how to interpret performance numbers in the result JSON, see [Benchmarking](benchmarking.md).

Full workflow (batch execution across models, metric aggregation): see [`tests/model-evaluation/README.md`](https://github.com/dbcls/bsllmner-mk2/blob/main/tests/model-evaluation/README.md).
