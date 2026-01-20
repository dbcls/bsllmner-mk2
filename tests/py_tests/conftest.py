import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import patch

import pytest
import yaml


# Patch INDEX_CACHE_DIR before importing bsllmner2 modules
# This is necessary because client/ollama.py creates the directory on import
@pytest.fixture(scope="session", autouse=True)
def patch_index_cache_dir() -> Generator[None, None, None]:
    """Patch INDEX_CACHE_DIR to use a temp directory for tests.

    This is needed because client/ollama.py has a module-level side effect
    that creates INDEX_CACHE_DIR on import, which fails in test environments
    without write access to /app/ontology.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # We need to patch this before the module is imported
        # Since conftest.py is loaded first, we can set up the patch here
        os.environ["BSLLMNER2_INDEX_CACHE_DIR"] = tmpdir
        yield


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bs_entries() -> List[Dict[str, Any]]:
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
def bs_entries_json_file(temp_dir: Path, sample_bs_entries: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSON file with BioSample entries."""
    file_path = temp_dir / "bs_entries.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(sample_bs_entries, f)
    return file_path


@pytest.fixture
def bs_entries_jsonl_file(temp_dir: Path, sample_bs_entries: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with BioSample entries."""
    file_path = temp_dir / "bs_entries.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for entry in sample_bs_entries:
            f.write(json.dumps(entry) + "\n")
    return file_path


@pytest.fixture
def sample_prompt() -> List[Dict[str, str]]:
    """Sample prompt for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract cell line information."},
    ]


@pytest.fixture
def prompt_file(temp_dir: Path, sample_prompt: List[Dict[str, str]]) -> Path:
    """Create a temporary prompt YAML file."""
    file_path = temp_dir / "prompt.yml"
    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(sample_prompt, f)
    return file_path


@pytest.fixture
def sample_format_schema() -> Dict[str, Any]:
    """Sample JSON schema for output format."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "cell_line": {"type": ["string", "null"]}
        },
        "required": ["cell_line"],
        "additionalProperties": False,
    }


@pytest.fixture
def format_schema_file(temp_dir: Path, sample_format_schema: Dict[str, Any]) -> Path:
    """Create a temporary JSON schema file."""
    file_path = temp_dir / "format.schema.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(sample_format_schema, f)
    return file_path


@pytest.fixture
def sample_select_config() -> Dict[str, Any]:
    """Sample select configuration for testing."""
    return {
        "fields": {
            "cell_line": {
                "ontology_file": None,
                "prompt_description": "Cell line name",
                "value_type": "string",
            }
        }
    }


@pytest.fixture
def select_config_file(temp_dir: Path, sample_select_config: Dict[str, Any]) -> Path:
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
