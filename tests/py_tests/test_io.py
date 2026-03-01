"""Tests for io module (file I/O + resume)."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from bsllmner2.errors import ResumeDataError
from bsllmner2.io import (
    _replace_surrogates,
    dump_extract_result,
    dump_extract_resume_file,
    dump_select_result,
    dump_select_resume_file,
    list_run_names,
    load_bs_entries,
    load_extract_result,
    load_extract_resume_file,
    load_format_schema,
    load_mapping,
    load_prompt_file,
    load_run_metadata,
    load_select_config,
    load_select_resume_file,
    remove_resume_files,
    validate_extract_resume_file,
    validate_resume_consistency,
)
from bsllmner2.models import (
    Config,
    LlmOutput,
    Prompt,
    Result,
    RunMetadata,
    SelectResult,
    WfInput,
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
        chat_response: ChatResponse = {  # type: ignore[assignment]
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
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
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
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            loaded = load_extract_resume_file("nonexistent-run")
            assert loaded == []

    def test_dump_and_load_select_resume(
        self,
        temp_dir: Path,
        sample_select_result: SelectResult,
    ) -> None:
        """Test dumping and loading select resume files."""
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
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
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
            loaded = load_select_resume_file("nonexistent-run")
            assert loaded == []


class TestValidateResumeConsistency:
    """Test cases for validate_resume_consistency function."""

    @pytest.fixture
    def make_llm_output(self) -> Callable[..., LlmOutput]:
        """Create LlmOutput instances for testing."""
        from ollama import ChatResponse

        def _make(accession: str) -> LlmOutput:
            chat_response: ChatResponse = {  # type: ignore[assignment]
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
    def make_select_result(self) -> Callable[..., SelectResult]:
        """Create SelectResult instances for testing."""

        def _make(accession: str) -> SelectResult:
            return SelectResult(
                accession=accession,
                extract_output={"cell_line": "Test"},
            )

        return _make

    def test_consistent_data_returns_done_ids(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
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

        done_ids, orphan_ids = validate_resume_consistency(extract_outputs, select_results, "test-run")

        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}
        assert orphan_ids == set()

    def test_orphan_entries_detected(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
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

        done_ids, orphan_ids = validate_resume_consistency(extract_outputs, select_results, "test-run")

        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}
        assert orphan_ids == {"SAMN004"}

    def test_multiple_orphans_detected(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
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

        done_ids, orphan_ids = validate_resume_consistency(extract_outputs, select_results, "test-run")

        assert done_ids == {"SAMN001"}
        assert orphan_ids == {"SAMN002", "SAMN003"}

    def test_invalid_data_raises_error(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
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

    def test_extract_only_returns_all_orphans(self, make_llm_output: Callable[..., LlmOutput]) -> None:
        """Test that extract-only data returns all as orphans."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
        ]

        done_ids, orphan_ids = validate_resume_consistency(extract_outputs, [], "test-run")

        assert done_ids == set()
        assert orphan_ids == {"SAMN001", "SAMN002"}


class TestValidateExtractResumeFile:
    """Test cases for validate_extract_resume_file function."""

    @pytest.fixture
    def make_llm_output(self) -> Callable[..., LlmOutput]:
        """Create LlmOutput instances for testing."""
        from ollama import ChatResponse

        def _make(accession: str) -> LlmOutput:
            chat_response: ChatResponse = {  # type: ignore[assignment]
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

    def test_returns_all_ids(self, make_llm_output: Callable[..., LlmOutput]) -> None:
        """Test that all accession IDs are returned."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
            make_llm_output("SAMN003"),
        ]

        done_ids = validate_extract_resume_file(extract_outputs, "test-run")

        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}

    def test_empty_input_returns_empty_set(self) -> None:
        """Test that empty input returns empty set."""
        done_ids = validate_extract_resume_file([], "test-run")
        assert done_ids == set()

    def test_duplicate_entries_detected_and_logged(
        self,
        make_llm_output: Callable[..., LlmOutput],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that duplicate entries are detected and logged as warning."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN002"),
            make_llm_output("SAMN001"),  # Duplicate
            make_llm_output("SAMN003"),
        ]

        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            caplog.set_level(logging.WARNING, logger="bsllmner2")
            done_ids = validate_extract_resume_file(extract_outputs, "test-run")
        finally:
            logger.propagate = original_propagate

        # Should still return all unique IDs
        assert done_ids == {"SAMN001", "SAMN002", "SAMN003"}

        # Should log a warning about duplicates
        assert "duplicate" in caplog.text.lower()
        assert "SAMN001" in caplog.text

    def test_multiple_duplicates_detected(
        self,
        make_llm_output: Callable[..., LlmOutput],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that multiple duplicates are all detected."""
        extract_outputs = [
            make_llm_output("SAMN001"),
            make_llm_output("SAMN001"),  # Duplicate 1
            make_llm_output("SAMN002"),
            make_llm_output("SAMN002"),  # Duplicate 2
            make_llm_output("SAMN001"),  # Duplicate 3
        ]

        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            caplog.set_level(logging.WARNING, logger="bsllmner2")
            done_ids = validate_extract_resume_file(extract_outputs, "test-run")
        finally:
            logger.propagate = original_propagate

        assert done_ids == {"SAMN001", "SAMN002"}
        assert "3 duplicate" in caplog.text


# === Helpers for Phase 2 tests ===


def _make_chat_response() -> dict:
    """Minimal ChatResponse-like dict for building LlmOutput / Result."""
    return {
        "model": "test-model",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": '{"cell_line": "HeLa"}'},
        "done": True,
    }


def _make_result(run_name: str = "test-run") -> Result:
    return Result(
        input=WfInput(
            bs_entries=[{"accession": "SAMN001", "title": "s1"}],
            prompt=[Prompt(role="system", content="test")],
            model="test-model",
            config=Config(),
        ),
        output=[
            LlmOutput(
                accession="SAMN001",
                output={"cell_line": "HeLa"},
                chat_response=_make_chat_response(),
            ),
        ],
        run_metadata=RunMetadata(
            run_name=run_name,
            model="test-model",
            start_time="2024-01-01T00:00:00Z",
        ),
    )


# === TestDumpExtractResult ===


class TestDumpExtractResult:
    def test_dump_creates_file(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            dump_extract_result(_make_result(), "my-run")
        assert (temp_dir / "my-run.json").exists()

    def test_dump_content_is_valid_json(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            dump_extract_result(_make_result(), "my-run")
        data = json.loads((temp_dir / "my-run.json").read_text())
        assert "run_metadata" in data

    def test_dump_roundtrip_preserves_accession(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            dump_extract_result(_make_result(), "my-run")
        data = json.loads((temp_dir / "my-run.json").read_text())
        assert data["output"][0]["accession"] == "SAMN001"

    def test_dump_surrogate_characters_replaced(self, temp_dir: Path) -> None:
        """Surrogate chars in LLM output are replaced with U+FFFD."""
        result = _make_result()
        result.output[0].output = {"cell_line": "bad\ud800char"}
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            dump_extract_result(result, "surr-run")
        content = (temp_dir / "surr-run.json").read_text()
        assert "\ud800" not in content
        assert "bad\ufffdchar" in content

    def test_dump_returns_correct_path(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            path = dump_extract_result(_make_result(), "my-run")
        assert path == temp_dir / "my-run.json"

    def test_dump_creates_parent_dir(self, temp_dir: Path) -> None:
        nested = temp_dir / "sub" / "dir"
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", nested):
            dump_extract_result(_make_result(), "my-run")
        assert (nested / "my-run.json").exists()


# === TestDumpSelectResult ===


class TestDumpSelectResult:
    def test_dump_creates_file(self, temp_dir: Path) -> None:
        sr = SelectResult(accession="SAMN001", extract_output={"cell_line": "HeLa"})
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
            dump_select_result([sr], "my-run")
        assert (temp_dir / "select_my-run.json").exists()

    def test_dump_content_is_valid_json(self, temp_dir: Path) -> None:
        sr = SelectResult(accession="SAMN001", extract_output={"cell_line": "HeLa"})
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
            dump_select_result([sr], "my-run")
        data = json.loads((temp_dir / "select_my-run.json").read_text())
        assert isinstance(data, list)
        assert data[0]["accession"] == "SAMN001"

    def test_dump_empty_list(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
            dump_select_result([], "my-run")
        data = json.loads((temp_dir / "select_my-run.json").read_text())
        assert data == []

    def test_dump_returns_correct_path(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir):
            path = dump_select_result([], "my-run")
        assert path == temp_dir / "select_my-run.json"


# === TestLoadExtractResult ===


class TestLoadExtractResult:
    def test_load_roundtrip(self, temp_dir: Path) -> None:
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.PROGRESS_DIR", temp_dir),
        ):
            dump_extract_result(_make_result("rt-run"), "rt-run")
            loaded = load_extract_result(temp_dir / "rt-run.json")
        assert loaded.output[0].accession == "SAMN001"
        assert loaded.run_metadata.model == "test-model"

    def test_load_with_progress_file(self, temp_dir: Path) -> None:
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.PROGRESS_DIR", temp_dir),
        ):
            dump_extract_result(_make_result("prog-run"), "prog-run")
            progress = temp_dir / "prog-run.txt"
            progress.write_text("SAMN001\nSAMN002\nSAMN003\n")
            loaded = load_extract_result(temp_dir / "prog-run.json")
        assert loaded.run_metadata.completed_count == 3

    def test_load_without_progress_file(self, temp_dir: Path) -> None:
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.PROGRESS_DIR", temp_dir),
        ):
            dump_extract_result(_make_result("noprog-run"), "noprog-run")
            loaded = load_extract_result(temp_dir / "noprog-run.json")
        assert loaded.run_metadata.completed_count is None

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_extract_result(Path("/nonexistent/result.json"))


# === TestLoadRunMetadata ===


class TestLoadRunMetadata:
    def test_load_existing_metadata(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            dump_extract_result(_make_result("meta-run"), "meta-run")
        metadata = load_run_metadata(temp_dir / "meta-run.json")
        assert metadata.run_name == "meta-run"
        assert metadata.model == "test-model"

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_run_metadata(Path("/nonexistent/result.json"))

    def test_load_missing_run_metadata_key(self, temp_dir: Path) -> None:
        path = temp_dir / "no_metadata.json"
        path.write_text(json.dumps({"output": []}))
        with pytest.raises(ValueError, match="No run metadata"):
            load_run_metadata(path)

    def test_load_empty_json_object(self, temp_dir: Path) -> None:
        path = temp_dir / "empty.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="No run metadata"):
            load_run_metadata(path)


# === TestListRunNames ===


class TestListRunNames:
    def test_list_existing_runs(self, temp_dir: Path) -> None:
        for name in ["run1", "run2", "run3"]:
            (temp_dir / f"{name}.json").write_text("{}")
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            names = list_run_names()
        assert sorted(names) == ["run1", "run2", "run3"]

    def test_empty_directory(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            assert list_run_names() == []

    def test_directory_not_exists(self, temp_dir: Path) -> None:
        missing = temp_dir / "nonexistent"
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", missing):
            assert list_run_names() == []

    def test_non_json_files_excluded(self, temp_dir: Path) -> None:
        (temp_dir / "run1.json").write_text("{}")
        (temp_dir / "notes.txt").write_text("hello")
        (temp_dir / "data.csv").write_text("a,b")
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            names = list_run_names()
        assert names == ["run1"]


# === TestRemoveResumeFiles ===


class TestRemoveResumeFiles:
    def test_remove_both_existing(self, temp_dir: Path) -> None:
        extract_f = temp_dir / "my-run_resume.json"
        select_f = temp_dir / "select_my-run_resume.json"
        extract_f.write_text("[]")
        select_f.write_text("[]")
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir),
        ):
            remove_resume_files("my-run")
        assert not extract_f.exists()
        assert not select_f.exists()

    def test_remove_nonexistent_files(self, temp_dir: Path) -> None:
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir),
        ):
            remove_resume_files("ghost-run")  # should not raise

    def test_remove_only_extract_exists(self, temp_dir: Path) -> None:
        extract_f = temp_dir / "my-run_resume.json"
        extract_f.write_text("[]")
        with (
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir),
            patch("bsllmner2.io.SELECT_RESULT_DIR", temp_dir),
        ):
            remove_resume_files("my-run")
        assert not extract_f.exists()


# === TestLoadBsEntries additional cases ===


class TestLoadBsEntriesAdditional:
    def test_jsonl_non_dict_line(self, temp_dir: Path) -> None:
        """A JSONL file where one line is a list (not a dict) must raise."""
        path = temp_dir / "bad.jsonl"
        # Two lines: first is valid dict, second is a list.
        # This triggers the JSONL parser (invalid as a single JSON value)
        # and then the non-dict check on the second line.
        path.write_text('{"accession": "SAMN001"}\n[1, 2, 3]\n')
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_bs_entries(path)

    def test_json_dict_not_list_raises(self, temp_dir: Path) -> None:
        """JSON file containing a dict (not a list) raises ValueError."""
        path = temp_dir / "dict.json"
        path.write_text('{"accession": "SAMN001"}')
        with pytest.raises(ValueError, match="must contain a list"):
            load_bs_entries(path)

    def test_json_list_of_non_dicts_raises(self, temp_dir: Path) -> None:
        """JSON file with a list of non-dicts raises ValueError."""
        path = temp_dir / "bad_list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must contain a list"):
            load_bs_entries(path)

    def test_jsonl_blank_lines_skipped(self, temp_dir: Path) -> None:
        """Blank lines in JSONL are skipped without error."""
        path = temp_dir / "blanks.jsonl"
        path.write_text('\n{"accession": "SAMN001"}\n\n{"accession": "SAMN002"}\n\n')
        entries = load_bs_entries(path)
        assert len(entries) == 2


# === TestLoadMapping additional cases ===


class TestLoadMappingAdditional:
    def test_empty_file(self, temp_dir: Path) -> None:
        path = temp_dir / "empty.tsv"
        path.write_text("")
        assert load_mapping(path) == {}

    def test_header_only(self, temp_dir: Path) -> None:
        path = temp_dir / "header_only.tsv"
        path.write_text(
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
        )
        assert load_mapping(path) == {}


# === TestLoadExtractResumeFile additional cases ===


class TestLoadExtractResumeFileAdditional:
    def test_non_list_json(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            path = temp_dir / "bad_resume.json"
            path.write_text('{"key": "val"}')
            with pytest.raises(ValueError, match="must contain a list"):
                load_extract_resume_file("bad")

    def test_empty_whitespace_content(self, temp_dir: Path) -> None:
        with patch("bsllmner2.io.EXTRACT_RESULT_DIR", temp_dir):
            path = temp_dir / "ws_resume.json"
            path.write_text("   \n  \t  ")
            result = load_extract_resume_file("ws")
        assert result == []


# === _replace_surrogates ===


class TestReplaceSurrogates:
    """Tests for _replace_surrogates.

    Previously only tested indirectly via dump functions.
    Direct tests ensure the regex and replacement character survive mutations.
    """

    def test_lone_high_surrogate_replaced(self) -> None:
        """Lone high surrogate is replaced with U+FFFD."""
        assert _replace_surrogates("hello\ud800world") == "hello\ufffdworld"

    def test_lone_low_surrogate_replaced(self) -> None:
        """Lone low surrogate is replaced with U+FFFD."""
        assert _replace_surrogates("hello\udc00world") == "hello\ufffdworld"

    def test_no_surrogates_unchanged(self) -> None:
        """String without surrogates is unchanged."""
        assert _replace_surrogates("hello world") == "hello world"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _replace_surrogates("") == ""

    def test_multiple_surrogates_all_replaced(self) -> None:
        """Multiple surrogates are all replaced."""
        result = _replace_surrogates("\ud800\ud801\udfff")
        assert result == "\ufffd\ufffd\ufffd"

    def test_surrogate_at_boundary(self) -> None:
        """Surrogates at start and end of string are replaced."""
        result = _replace_surrogates("\ud800text\udfff")
        assert result == "\ufffdtext\ufffd"

    def test_replacement_char_is_ufffd(self) -> None:
        """Replacement character is specifically U+FFFD, not empty or other."""
        result = _replace_surrogates("\ud800")
        assert result == "\ufffd"
        assert len(result) == 1


# === validate_extract_resume_file: truncation boundary ===


class TestValidateExtractResumeFileTruncation:
    """Test the truncation boundary in validate_extract_resume_file warning.

    The code shows `duplicates[:5]` and `'...' if len(duplicates) > 5`.
    With exactly 5 duplicates, no '...' should appear.
    With 6, '...' should appear.
    """

    @pytest.fixture
    def make_llm_output(self) -> Callable[..., LlmOutput]:
        from ollama import ChatResponse

        def _make(accession: str) -> LlmOutput:
            chat_response: ChatResponse = {  # type: ignore[assignment]
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

    def test_exactly_5_duplicates_no_ellipsis(
        self,
        make_llm_output: Callable[..., LlmOutput],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With exactly 5 duplicates, no '...' in warning message."""
        outputs: list[LlmOutput] = []
        for i in range(5):
            outputs.append(make_llm_output(f"SAMN{i:03d}"))
            outputs.append(make_llm_output(f"SAMN{i:03d}"))  # duplicate

        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            caplog.set_level(logging.WARNING, logger="bsllmner2")
            validate_extract_resume_file(outputs, "test-run")
        finally:
            logger.propagate = original_propagate

        assert "5 duplicate" in caplog.text
        assert "..." not in caplog.text

    def test_6_duplicates_has_ellipsis(
        self,
        make_llm_output: Callable[..., LlmOutput],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With 6 duplicates, '...' appears in warning message."""
        outputs: list[LlmOutput] = []
        for i in range(6):
            outputs.append(make_llm_output(f"SAMN{i:03d}"))
            outputs.append(make_llm_output(f"SAMN{i:03d}"))  # duplicate

        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            caplog.set_level(logging.WARNING, logger="bsllmner2")
            validate_extract_resume_file(outputs, "test-run")
        finally:
            logger.propagate = original_propagate

        assert "6 duplicate" in caplog.text
        assert "..." in caplog.text


# === validate_resume_consistency: truncation boundary ===


class TestValidateResumeConsistencyTruncation:
    """Test the truncation boundary in validate_resume_consistency error message.

    The code uses `sorted(invalid_ids)[:5]` and `'...' if len(invalid_ids) > 5`.
    """

    @pytest.fixture
    def make_llm_output(self) -> Callable[..., LlmOutput]:
        from ollama import ChatResponse

        def _make(accession: str) -> LlmOutput:
            chat_response: ChatResponse = {  # type: ignore[assignment]
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
    def make_select_result(self) -> Callable[..., SelectResult]:
        def _make(accession: str) -> SelectResult:
            return SelectResult(
                accession=accession,
                extract_output={"cell_line": "Test"},
            )

        return _make

    def test_exactly_5_invalid_no_ellipsis(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
    ) -> None:
        """With exactly 5 invalid IDs, no '...' in error message."""
        extract_outputs = [make_llm_output("SAMN001")]
        select_results = [make_select_result("SAMN001")]
        # Add 5 invalid (select-only) entries
        select_results.extend(make_select_result(f"INVALID{i:03d}") for i in range(5))

        with pytest.raises(ResumeDataError, match="5 entries") as exc_info:
            validate_resume_consistency(extract_outputs, select_results, "test-run")
        assert "..." not in str(exc_info.value)

    def test_6_invalid_has_ellipsis(
        self,
        make_llm_output: Callable[..., LlmOutput],
        make_select_result: Callable[..., SelectResult],
    ) -> None:
        """With 6 invalid IDs, '...' appears in error message."""
        extract_outputs = [make_llm_output("SAMN001")]
        select_results = [make_select_result("SAMN001")]
        # Add 6 invalid (select-only) entries
        select_results.extend(make_select_result(f"INVALID{i:03d}") for i in range(6))

        with pytest.raises(ResumeDataError, match="6 entries") as exc_info:
            validate_resume_consistency(extract_outputs, select_results, "test-run")
        assert "..." in str(exc_info.value)


# === load_mapping: empty optional fields ===


class TestLoadMappingOptionalFields:
    """Test that empty optional fields in mapping are converted to None."""

    def test_empty_extraction_answer_becomes_none(self, temp_dir: Path) -> None:
        """Empty extraction_answer field becomes None."""
        path = temp_dir / "mapping.tsv"
        content = (
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
            "SAMN001\tRNA-seq\t\tCVCL_0030\tHeLa"
        )
        path.write_text(content)
        mapping = load_mapping(path)
        assert mapping["SAMN001"].extraction_answer is None

    def test_empty_mapping_answer_id_becomes_none(self, temp_dir: Path) -> None:
        """Empty mapping_answer_id field becomes None."""
        path = temp_dir / "mapping.tsv"
        content = (
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
            "SAMN001\tRNA-seq\tHeLa\t\tHeLa"
        )
        path.write_text(content)
        mapping = load_mapping(path)
        assert mapping["SAMN001"].mapping_answer_id is None

    def test_empty_mapping_answer_label_becomes_none(self, temp_dir: Path) -> None:
        """Empty mapping_answer_label field becomes None."""
        path = temp_dir / "mapping.tsv"
        content = (
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
            "SAMN001\tRNA-seq\tHeLa\tCVCL_0030\t"
        )
        path.write_text(content)
        mapping = load_mapping(path)
        assert mapping["SAMN001"].mapping_answer_label is None

    def test_all_optional_fields_empty(self, temp_dir: Path) -> None:
        """All optional fields empty become None; experiment_type is always present."""
        path = temp_dir / "mapping.tsv"
        content = (
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
            "SAMN001\tRNA-seq\t\t\t"
        )
        path.write_text(content)
        mapping = load_mapping(path)
        assert mapping["SAMN001"].experiment_type == "RNA-seq"
        assert mapping["SAMN001"].extraction_answer is None
        assert mapping["SAMN001"].mapping_answer_id is None
        assert mapping["SAMN001"].mapping_answer_label is None

    def test_field_count_mismatch_raises(self, temp_dir: Path) -> None:
        """Row with wrong field count raises ValueError with line number."""
        path = temp_dir / "mapping.tsv"
        content = (
            "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
            "SAMN001\tRNA-seq\tHeLa"
        )
        path.write_text(content)
        with pytest.raises(ValueError, match="line 2"):
            load_mapping(path)
