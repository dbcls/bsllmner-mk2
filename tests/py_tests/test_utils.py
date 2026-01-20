"""Tests for utility functions."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

from bsllmner2.errors import ResumeDataError
from bsllmner2.schema import LlmOutput, SelectResult
from bsllmner2.utils import (
    build_extract_prompt_for_select,
    build_extract_schema_for_select,
    dump_extract_resume_file,
    dump_select_resume_file,
    get_now_str,
    load_bs_entries,
    load_extract_resume_file,
    load_format_schema,
    load_mapping,
    load_prompt_file,
    load_select_config,
    load_select_resume_file,
    validate_resume_consistency,
)


class TestLoadBsEntries:
    """Test cases for load_bs_entries function."""

    def test_load_json_file(self, bs_entries_json_file: Path) -> None:
        """Test loading a JSON file."""
        entries = load_bs_entries(bs_entries_json_file)
        assert len(entries) == 2
        assert entries[0]["accession"] == "SAMN00000001"
        assert entries[1]["accession"] == "SAMN00000002"

    def test_load_jsonl_file(self, bs_entries_jsonl_file: Path) -> None:
        """Test loading a JSONL file."""
        entries = load_bs_entries(bs_entries_jsonl_file)
        assert len(entries) == 2
        assert entries[0]["accession"] == "SAMN00000001"
        assert entries[1]["accession"] == "SAMN00000002"

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_bs_entries(Path("/nonexistent/path.json"))

    def test_invalid_json(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for invalid JSON."""
        file_path = temp_dir / "invalid.json"
        file_path.write_text("not valid json {")
        with pytest.raises(ValueError):
            load_bs_entries(file_path)

    def test_empty_jsonl(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for empty JSONL."""
        file_path = temp_dir / "empty.jsonl"
        file_path.write_text("")
        with pytest.raises(ValueError, match="no valid JSON objects"):
            load_bs_entries(file_path)


class TestLoadPromptFile:
    """Test cases for load_prompt_file function."""

    def test_load_valid_prompt(self, prompt_file: Path) -> None:
        """Test loading a valid prompt file."""
        prompts = load_prompt_file(prompt_file)
        assert len(prompts) == 2
        assert prompts[0].role == "system"
        assert prompts[1].role == "user"

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_prompt_file(Path("/nonexistent/prompt.yml"))

    def test_invalid_format(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for invalid format."""
        file_path = temp_dir / "invalid_prompt.yml"
        with file_path.open("w") as f:
            yaml.dump({"not": "a list"}, f)
        with pytest.raises(ValueError, match="must contain a list"):
            load_prompt_file(file_path)


class TestLoadMapping:
    """Test cases for load_mapping function."""

    def test_load_valid_mapping(self, mapping_file: Path) -> None:
        """Test loading a valid mapping file."""
        mapping = load_mapping(mapping_file)
        assert "SAMN00000001" in mapping
        assert mapping["SAMN00000001"].extraction_answer == "HeLa"
        assert mapping["SAMN00000001"].mapping_answer_id == "CVCL_0030"

    def test_invalid_header(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for invalid header."""
        file_path = temp_dir / "invalid_mapping.tsv"
        file_path.write_text("Wrong\tHeader\n")
        with pytest.raises(ValueError, match="Header mismatch"):
            load_mapping(file_path)

    def test_duplicate_accession(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for duplicate accession."""
        file_path = temp_dir / "duplicate.tsv"
        content = """BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label
SAMN00000001\tRNA-seq\tHeLa\tCVCL_0030\tHeLa
SAMN00000001\tRNA-seq\tHEK293\tCVCL_0045\tHEK293"""
        file_path.write_text(content)
        with pytest.raises(ValueError, match="Duplicate accession"):
            load_mapping(file_path)


class TestLoadFormatSchema:
    """Test cases for load_format_schema function."""

    def test_load_valid_schema(self, format_schema_file: Path) -> None:
        """Test loading a valid JSON schema file."""
        schema = load_format_schema(format_schema_file)
        assert schema["type"] == "object"
        assert "cell_line" in schema["properties"]

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_format_schema(Path("/nonexistent/schema.json"))

    def test_invalid_json(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for invalid JSON."""
        file_path = temp_dir / "invalid.schema.json"
        file_path.write_text("not valid json {")
        with pytest.raises(ValueError, match="Invalid JSON schema"):
            load_format_schema(file_path)


class TestLoadSelectConfig:
    """Test cases for load_select_config function."""

    def test_load_valid_config(self, select_config_file: Path) -> None:
        """Test loading a valid select config file."""
        config = load_select_config(select_config_file)
        assert "cell_line" in config.fields
        assert config.fields["cell_line"].value_type == "string"

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_select_config(Path("/nonexistent/config.json"))

    def test_invalid_json(self, temp_dir: Path) -> None:
        """Test that ValueError is raised for invalid JSON."""
        file_path = temp_dir / "invalid_config.json"
        file_path.write_text("not valid json {")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_select_config(file_path)


class TestGetNowStr:
    """Test cases for get_now_str function."""

    def test_format(self) -> None:
        """Test that the output format is correct."""
        result = get_now_str()
        # Should match YYYYMMDD_HHMMSS format
        datetime.strptime(result, "%Y%m%d_%H%M%S")

    def test_returns_current_utc_time(self) -> None:
        """Test that get_now_str returns UTC time close to now."""
        before = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        result = get_now_str()
        after = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Result should be between before and after (or equal)
        assert before <= result <= after

    def test_uses_utc_time(self) -> None:
        """Verify that get_now_str uses UTC time for consistency."""
        utc_now = datetime.now(timezone.utc)
        result = get_now_str()
        result_dt = datetime.strptime(result, "%Y%m%d_%H%M%S")

        # The result should be within 1 second of UTC time
        utc_diff = abs((result_dt - utc_now.replace(microsecond=0, tzinfo=None)).total_seconds())
        assert utc_diff <= 1  # Should be within 1 second of UTC time


class TestResumeFileFunctions:
    """Test cases for resume file load/dump functions."""

    @pytest.fixture
    def mock_extract_result_dir(self, temp_dir: Path) -> Path:
        """Create a mock EXTRACT_RESULT_DIR."""
        return temp_dir

    @pytest.fixture
    def mock_select_result_dir(self, temp_dir: Path) -> Path:
        """Create a mock SELECT_RESULT_DIR."""
        return temp_dir

    @pytest.fixture
    def sample_llm_output(self) -> LlmOutput:
        """Create a sample LlmOutput for testing."""
        from ollama import ChatResponse

        # Create a minimal ChatResponse-like dict
        chat_response: ChatResponse = {  # type: ignore
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "test"},
            "done": True,
        }
        return LlmOutput(
            accession="SAMN00000001",
            output={"cell_line": "HeLa"},
            chat_response=chat_response,
        )

    @pytest.fixture
    def sample_select_result(self) -> SelectResult:
        """Create a sample SelectResult for testing."""
        return SelectResult(
            accession="SAMN00000001",
            extract_output={"cell_line": "HeLa"},
        )

    def test_dump_and_load_extract_resume(
        self,
        temp_dir: Path,
        sample_llm_output: LlmOutput,
    ) -> None:
        """Test dumping and loading extract resume files."""
        with patch("bsllmner2.utils.EXTRACT_RESULT_DIR", temp_dir):
            # Dump
            outputs = [sample_llm_output]
            dump_extract_resume_file(outputs, "test-run")

            # Verify file exists
            resume_file = temp_dir / "test-run_resume.json"
            assert resume_file.exists()

            # Load
            loaded = load_extract_resume_file("test-run")
            assert len(loaded) == 1
            assert loaded[0].accession == "SAMN00000001"
            assert loaded[0].output == {"cell_line": "HeLa"}

    def test_load_extract_resume_missing_file(self, temp_dir: Path) -> None:
        """Test that loading missing resume file returns empty list."""
        with patch("bsllmner2.utils.EXTRACT_RESULT_DIR", temp_dir):
            loaded = load_extract_resume_file("nonexistent-run")
            assert loaded == []

    def test_dump_and_load_select_resume(
        self,
        temp_dir: Path,
        sample_select_result: SelectResult,
    ) -> None:
        """Test dumping and loading select resume files."""
        with patch("bsllmner2.utils.SELECT_RESULT_DIR", temp_dir):
            # Dump
            results = [sample_select_result]
            dump_select_resume_file(results, "test-run")

            # Verify file exists
            resume_file = temp_dir / "select_test-run_resume.json"
            assert resume_file.exists()

            # Load
            loaded = load_select_resume_file("test-run")
            assert len(loaded) == 1
            assert loaded[0].accession == "SAMN00000001"

    def test_load_select_resume_missing_file(self, temp_dir: Path) -> None:
        """Test that loading missing resume file returns empty list."""
        with patch("bsllmner2.utils.SELECT_RESULT_DIR", temp_dir):
            loaded = load_select_resume_file("nonexistent-run")
            assert loaded == []


class TestBuildExtractSchemaForSelect:
    """Test cases for build_extract_schema_for_select function."""

    def test_string_field(self, select_config_file: Path) -> None:
        """Test schema generation for string field."""
        config = load_select_config(select_config_file)
        schema = build_extract_schema_for_select(config)

        assert schema["type"] == "object"
        assert "cell_line" in schema["properties"]
        assert schema["properties"]["cell_line"]["type"] == ["string", "null"]

    def test_array_field(self, temp_dir: Path) -> None:
        """Test schema generation for array field."""
        config_data = {
            "fields": {
                "diseases": {
                    "value_type": "array",
                    "prompt_description": "List of diseases",
                }
            }
        }
        config_file = temp_dir / "array_config.json"
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_select_config(config_file)
        schema = build_extract_schema_for_select(config)

        assert "diseases" in schema["properties"]
        assert schema["properties"]["diseases"]["type"] == ["array", "null"]
        assert schema["properties"]["diseases"]["items"]["type"] == "string"


class TestBuildExtractPromptForSelect:
    """Test cases for build_extract_prompt_for_select function."""

    def test_prompt_generation(self, select_config_file: Path) -> None:
        """Test prompt generation for select mode."""
        config = load_select_config(select_config_file)
        prompts = build_extract_prompt_for_select(config)

        assert len(prompts) == 2
        assert prompts[0].role == "system"
        assert prompts[1].role == "user"
        assert "cell_line" in prompts[1].content


class TestValidateResumeConsistency:
    """Test cases for validate_resume_consistency function."""

    @pytest.fixture
    def make_llm_output(self) -> callable:
        """Factory to create LlmOutput instances."""
        from ollama import ChatResponse

        def _make(accession: str) -> LlmOutput:
            chat_response: ChatResponse = {  # type: ignore
                "model": "test-model",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "test"},
                "done": True,
            }
            return LlmOutput(
                accession=accession,
                output={"cell_line": "Test"},
                chat_response=chat_response,
            )
        return _make

    @pytest.fixture
    def make_select_result(self) -> callable:
        """Factory to create SelectResult instances."""
        def _make(accession: str) -> SelectResult:
            return SelectResult(
                accession=accession,
                extract_output={"cell_line": "Test"},
            )
        return _make

    def test_consistent_data_returns_done_ids(
        self, make_llm_output: callable, make_select_result: callable
    ) -> None:
        """Test that consistent data returns done_ids and empty orphans."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
            make_llm_output("SAMN003"),
        ]
        select_results = [
            make_select_result("SAMN001"),
            make_select_result("SAMN002"),
            make_select_result("SAMN003"),
        ]

        done_ids, orphan_ids = validate_resume_consistency(
            extract_outputs, select_results, "test-run"
        )

        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}
        assert orphan_ids == set()

    def test_orphan_entries_detected(
        self, make_llm_output: callable, make_select_result: callable
    ) -> None:
        """Test that orphan entries (extract only) are detected."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
            make_llm_output("SAMN003"),
            make_llm_output("SAMN004"),  # Orphan
        ]
        select_results = [
            make_select_result("SAMN001"),
            make_select_result("SAMN002"),
            make_select_result("SAMN003"),
        ]

        done_ids, orphan_ids = validate_resume_consistency(
            extract_outputs, select_results, "test-run"
        )

        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}
        assert orphan_ids == {"SAMN004"}

    def test_multiple_orphans_detected(
        self, make_llm_output: callable, make_select_result: callable
    ) -> None:
        """Test that multiple orphan entries are detected."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
            make_llm_output("SAMN003"),
        ]
        select_results = [
            make_select_result("SAMN001"),
        ]

        done_ids, orphan_ids = validate_resume_consistency(
            extract_outputs, select_results, "test-run"
        )

        assert done_ids == {"SAMN001"}
        assert orphan_ids == {"SAMN002", "SAMN003"}

    def test_invalid_data_raises_error(
        self, make_llm_output: callable, make_select_result: callable
    ) -> None:
        """Test that select entries without extract raise ResumeDataError."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
        ]
        select_results = [
            make_select_result("SAMN001"),
            make_select_result("SAMN002"),
            make_select_result("SAMN003"),  # No extract for this
        ]

        with pytest.raises(ResumeDataError) as exc_info:
            validate_resume_consistency(extract_outputs, select_results, "test-run")

        assert "test-run" in str(exc_info.value)
        assert "SAMN003" in str(exc_info.value)

    def test_empty_inputs(self) -> None:
        """Test that empty inputs return empty sets."""
        done_ids, orphan_ids = validate_resume_consistency([], [], "test-run")

        assert done_ids == set()
        assert orphan_ids == set()

    def test_extract_only_returns_all_orphans(
        self, make_llm_output: callable
    ) -> None:
        """Test that extract-only data returns all as orphans."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
        ]

        done_ids, orphan_ids = validate_resume_consistency(
            extract_outputs, [], "test-run"
        )

        assert done_ids == set()
        assert orphan_ids == {"SAMN001", "SAMN002"}
