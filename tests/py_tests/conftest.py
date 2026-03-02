import atexit
import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

# Patch INDEX_CACHE_DIR before any bsllmner2 module is imported.
# client/ollama.py runs INDEX_CACHE_DIR.mkdir() at import time, which fails
# outside Docker where /app/ontology does not exist.
# conftest.py is evaluated before test modules are collected, so setting
# the env var here ensures the module-level side effect uses a temp dir.
# atexit is used because conftest.py module-level code runs outside the
# pytest fixture lifecycle.
_INDEX_CACHE_TMPDIR_OBJ = tempfile.TemporaryDirectory()
_INDEX_CACHE_TMPDIR = _INDEX_CACHE_TMPDIR_OBJ.name
atexit.register(_INDEX_CACHE_TMPDIR_OBJ.cleanup)
os.environ.setdefault("BSLLMNER2_INDEX_CACHE_DIR", _INDEX_CACHE_TMPDIR)

from bsllmner2.models import LlmOutput  # noqa: E402  # must be after env var setup


def make_chat_response(content: str) -> ChatResponse:
    """Build a minimal ChatResponse with the given assistant content."""
    return ChatResponse(
        model="test-model",
        message=Message(role="assistant", content=content),
        done=True,
        done_reason="stop",
        total_duration=0,
        load_duration=0,
        prompt_eval_count=0,
        prompt_eval_duration=0,
        eval_count=0,
        eval_duration=0,
    )


def make_llm_output(accession: str, output: object = None) -> LlmOutput:
    """Create a LlmOutput with minimal boilerplate for testing."""
    return LlmOutput(
        accession=accession,
        output=output,
        chat_response=make_chat_response('{"cell_line": "Test"}'),
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bs_entries() -> list[dict[str, Any]]:
    """Sample BioSample entries for testing."""
    return [
        {
            "accession": "SAMN00000001",
            "title": "Test Sample 1",
            "characteristics": {"cell_line": "HeLa"},
        },
        {
            "accession": "SAMN00000002",
            "title": "Test Sample 2",
            "characteristics": {"cell_line": "HEK293"},
        },
    ]


@pytest.fixture
def bs_entries_json_file(temp_dir: Path, sample_bs_entries: list[dict[str, Any]]) -> Path:
    """Create a temporary JSON file with BioSample entries."""
    file_path = temp_dir / "bs_entries.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(sample_bs_entries, f)
    return file_path


@pytest.fixture
def bs_entries_jsonl_file(temp_dir: Path, sample_bs_entries: list[dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with BioSample entries."""
    file_path = temp_dir / "bs_entries.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for entry in sample_bs_entries:
            f.write(json.dumps(entry) + "\n")
    return file_path


@pytest.fixture
def sample_prompt() -> list[dict[str, str]]:
    """Sample prompt for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract cell line information."},
    ]


@pytest.fixture
def prompt_file(temp_dir: Path, sample_prompt: list[dict[str, str]]) -> Path:
    """Create a temporary prompt YAML file."""
    file_path = temp_dir / "prompt.yml"
    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(sample_prompt, f)
    return file_path


@pytest.fixture
def sample_format_schema() -> dict[str, Any]:
    """Sample JSON schema for output format."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"cell_line": {"type": ["string", "null"]}},
        "required": ["cell_line"],
        "additionalProperties": False,
    }


@pytest.fixture
def format_schema_file(temp_dir: Path, sample_format_schema: dict[str, Any]) -> Path:
    """Create a temporary JSON schema file."""
    file_path = temp_dir / "format.schema.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(sample_format_schema, f)
    return file_path


@pytest.fixture
def sample_select_config() -> dict[str, Any]:
    """Sample select configuration for testing."""
    return {
        "fields": {
            "cell_line": {
                "ontology_file": None,
                "prompt_description": "Cell line name",
                "value_type": "string",
            },
        },
    }


@pytest.fixture
def select_config_file(temp_dir: Path, sample_select_config: dict[str, Any]) -> Path:
    """Create a temporary select config file."""
    file_path = temp_dir / "select_config.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(sample_select_config, f)
    return file_path


@pytest.fixture
def sample_mapping_tsv() -> str:
    """Sample mapping TSV content."""
    return """BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label
SAMN00000001\tRNA-seq\tHeLa\tCVCL_0030\tHeLa
SAMN00000002\tRNA-seq\tHEK293\tCVCL_0045\tHEK293"""


@pytest.fixture
def mapping_file(temp_dir: Path, sample_mapping_tsv: str) -> Path:
    """Create a temporary mapping TSV file."""
    file_path = temp_dir / "mapping.tsv"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(sample_mapping_tsv)
    return file_path


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after test."""
    env_vars_to_clean = [
        "OLLAMA_HOST",
        "BSLLMNER2_DEBUG",
        "BSLLMNER2_API_HOST",
        "BSLLMNER2_API_PORT",
        "BSLLMNER2_API_URL_PREFIX",
    ]
    original_values = {k: os.environ.get(k) for k in env_vars_to_clean}

    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]

    yield

    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class FakeLlmBackend:
    """In-memory LlmBackend for testing. Returns pre-configured responses."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.host = "http://fake:11434"

    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool | None = None,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse:
        idx = self._call_index
        self._call_index += 1
        if idx >= len(self._responses):
            raise RuntimeError(f"FakeLlmBackend: no response configured for call {idx}")
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return make_chat_response(item)

    async def ensure_model(self, model: str) -> None:
        pass

    def list_models(self) -> list[str]:
        return ["test-model"]
