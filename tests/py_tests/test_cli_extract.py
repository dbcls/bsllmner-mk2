"""Tests for CLI extract mode argument parsing and async execution."""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bsllmner2.cli_extract import parse_args, run_cli_extract_async
from bsllmner2.config import RESUME_BATCH_SIZE
from bsllmner2.models import ExtractEntry
from tests.py_tests.conftest import FakeLlmBackend, make_chat_response


class TestParseArgsExtract:
    """Test cases for cli_extract.parse_args function."""

    def test_minimal_args(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test parsing with minimal required arguments."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
        ]
        _config, cli_args = parse_args(args)

        assert cli_args.bs_entries == bs_entries_json_file.resolve()
        assert cli_args.prompt == prompt_file.resolve()
        assert cli_args.format is None
        assert cli_args.model == "llama3.1:70b"
        assert cli_args.thinking is False
        assert cli_args.max_entries is None
        assert cli_args.run_name is None
        assert cli_args.resume is False
        assert cli_args.batch_size == RESUME_BATCH_SIZE

    def test_all_args(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        format_schema_file: Path,
    ) -> None:
        """Test parsing with all arguments specified."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--format",
            str(format_schema_file),
            "--model",
            "qwen2.5:72b",
            "--thinking",
            "true",
            "--max-entries",
            "100",
            "--ollama-host",
            "http://custom:11434",
            "--debug",
            "--run-name",
            "test-run",
            "--resume",
            "--batch-size",
            "512",
        ]
        config, cli_args = parse_args(args)  # config used below

        assert cli_args.bs_entries == bs_entries_json_file.resolve()
        assert cli_args.prompt == prompt_file.resolve()
        assert cli_args.format == format_schema_file.resolve()
        assert cli_args.model == "qwen2.5:72b"
        assert cli_args.max_entries == 100
        assert cli_args.run_name == "test-run"
        assert cli_args.resume is True
        assert cli_args.batch_size == 512
        assert config.ollama_host == "http://custom:11434"
        assert config.debug is True

    def test_thinking_flag_type(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test that --thinking flag produces correct type.

        Note: Although cli_extract.parse_args returns string ("true"/"false"),
        Pydantic's CliExtractArgs model automatically converts it to bool.
        This works, but is inconsistent with cli_select which explicitly
        uses str_to_bool converter in argparse.
        """
        args_true = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--thinking",
            "true",
        ]
        _, cli_args_true = parse_args(args_true)

        args_false = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--thinking",
            "false",
        ]
        _, cli_args_false = parse_args(args_false)

        # Pydantic auto-converts "true"/"false" strings to bool
        assert cli_args_true.thinking is True
        assert cli_args_false.thinking is False
        assert isinstance(cli_args_true.thinking, bool)

    def test_thinking_flag_invalid_value(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test that --thinking flag rejects invalid values."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--thinking",
            "invalid",
        ]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_max_entries_negative_becomes_none(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test that negative max_entries becomes None."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--max-entries",
            "-1",
        ]
        _, cli_args = parse_args(args)
        assert cli_args.max_entries is None

    def test_missing_bs_entries_file(self, prompt_file: Path) -> None:
        """Test that missing bs_entries file causes SystemExit via parser.error()."""
        args = [
            "--bs-entries",
            "/nonexistent/path/bs_entries.json",
            "--prompt",
            str(prompt_file),
        ]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_missing_prompt_file(self, bs_entries_json_file: Path, temp_dir: Path) -> None:
        """Test that missing prompt file causes SystemExit via parser.error()."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(temp_dir / "nonexistent_prompt.yml"),
        ]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_missing_format_file(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test that missing format file causes SystemExit via parser.error()."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--format",
            "/nonexistent/format.schema.json",
        ]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_missing_required_args(self) -> None:
        """Test that missing required arguments causes SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


# === CLI async integration test ===


@pytest.mark.asyncio(loop_scope="function")
class TestRunCliExtractAsync:
    async def test_basic_run(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        tmp_path: Path,
    ) -> None:
        """Smoke test: run_cli_extract_async completes with all externals mocked."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
        ]

        result_dir = tmp_path / "results"
        result_dir.mkdir()

        with (
            patch("bsllmner2.cli_extract.sys") as mock_sys,
            patch(
                "bsllmner2.cli_extract.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        '{"cell_line": "HEK293"}',
                    ]
                ),
            ),
            patch("bsllmner2.cli_extract.dump_extract_result", return_value=tmp_path / "result.json"),
            patch("bsllmner2.cli_extract.dump_extract_resume_file"),
            patch("bsllmner2.cli_extract.remove_resume_files"),
        ):
            mock_sys.argv = ["bsllmner2-extract", *cli_args]
            await run_cli_extract_async()

    async def test_resume_skips_processed_entries(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        tmp_path: Path,
    ) -> None:
        """Resume mode skips entries already present in the resume file."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--resume",
            "--run-name",
            "test-run",
        ]

        already_done = ExtractEntry(
            accession="SAMN00000001",
            extracted={"cell_line": "HeLa"},
        )

        with (
            patch("bsllmner2.cli_extract.sys") as mock_sys,
            patch(
                "bsllmner2.cli_extract.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        # Only one response needed: SAMN00000002
                        '{"cell_line": "HEK293"}',
                    ]
                ),
            ),
            patch(
                "bsllmner2.cli_extract.load_extract_resume_file",
                return_value=[already_done],
            ),
            patch(
                "bsllmner2.cli_extract.validate_extract_resume_file",
                return_value={"SAMN00000001"},
            ),
            patch("bsllmner2.cli_extract.dump_extract_result", return_value=tmp_path / "result.json"),
            patch("bsllmner2.cli_extract.dump_extract_resume_file"),
            patch("bsllmner2.cli_extract.remove_resume_files"),
            patch(
                "bsllmner2.cli_extract.ner",
                new_callable=AsyncMock,
                return_value=(
                    [ExtractEntry(accession="SAMN00000002", extracted={"cell_line": "HEK293"})],
                    [make_chat_response('{"cell_line": "HEK293"}')],
                ),
            ) as mock_ner,
        ):
            mock_sys.argv = ["bsllmner2-extract", *cli_args]
            await run_cli_extract_async()

            # ner should have been called with only the non-skipped entry
            assert mock_ner.call_count == 1
            ner_call_entries = mock_ner.call_args[0][1]  # second positional arg: bs_entries
            accessions = [e["accession"] for e in ner_call_entries]
            assert "SAMN00000001" not in accessions
            assert "SAMN00000002" in accessions

    async def test_failed_status_on_bsllmner2_error(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        tmp_path: Path,
    ) -> None:
        """OllamaConnectionError during ner sets status to 'failed' in the result."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
        ]

        with (
            patch("bsllmner2.cli_extract.sys") as mock_sys,
            patch(
                "bsllmner2.cli_extract.OllamaBackend",
                return_value=FakeLlmBackend(
                    # First chat call raises ConnectionError -> ner converts to OllamaConnectionError
                    [ConnectionError("refused")],
                ),
            ),
            patch(
                "bsllmner2.cli_extract.dump_extract_result",
                return_value=tmp_path / "result.json",
            ) as mock_dump,
            patch("bsllmner2.cli_extract.dump_extract_resume_file"),
            patch("bsllmner2.cli_extract.remove_resume_files") as mock_remove,
        ):
            mock_sys.argv = ["bsllmner2-extract", *cli_args]
            await run_cli_extract_async()

            # dump_extract_result should have been called with a result whose status is "failed"
            assert mock_dump.call_count == 1
            result_arg = mock_dump.call_args[0][0]
            assert result_arg.run_metadata.status == "failed"

            # resume files should NOT be removed on failure
            mock_remove.assert_not_called()


# === CLI integration tests with real file I/O ===


@pytest.mark.asyncio(loop_scope="function")
class TestCliExtractIntegration:
    """Integration tests with FakeLlmBackend and real file I/O."""

    async def test_real_file_io(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: real files are written, output structure is valid, status is 'completed'."""
        result_dir = tmp_path / "results" / "extract"
        progress_dir = tmp_path / "progress"

        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
        ]

        with (
            patch("bsllmner2.cli_extract.sys") as mock_sys,
            patch(
                "bsllmner2.cli_extract.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        '{"cell_line": "HEK293"}',
                    ]
                ),
            ),
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", result_dir),
            patch("bsllmner2.io.PROGRESS_DIR", progress_dir),
            patch("bsllmner2.cli_extract.remove_resume_files"),
        ):
            mock_sys.argv = ["bsllmner2-extract", *cli_args]
            await run_cli_extract_async()

        # Verify output files exist
        json_files = list(result_dir.glob("*.json"))
        # Filter out resume files
        result_files = [f for f in json_files if "_resume" not in f.name]
        assert len(result_files) >= 1

        # Verify file content is valid JSON with expected structure
        result_data = json.loads(result_files[0].read_text())
        assert "run_metadata" in result_data
        assert result_data["run_metadata"]["status"] == "completed"
        assert "entries" in result_data
        assert len(result_data["entries"]) == 2

    async def test_batch_loss_logged_at_error(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When ner() returns fewer outputs than entries, an ERROR is logged."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
        ]

        with (
            patch("bsllmner2.cli_extract.sys") as mock_sys,
            patch(
                "bsllmner2.cli_extract.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        RuntimeError("boom"),  # second entry fails
                    ]
                ),
            ),
            patch("bsllmner2.cli_extract.dump_extract_result", return_value=tmp_path / "result.json"),
            patch("bsllmner2.cli_extract.dump_extract_resume_file"),
            patch("bsllmner2.cli_extract.remove_resume_files"),
        ):
            mock_sys.argv = ["bsllmner2-extract", *cli_args]

            logger = logging.getLogger("bsllmner2")
            original_propagate = logger.propagate
            try:
                logger.propagate = True
                with caplog.at_level(logging.ERROR, logger="bsllmner2"):
                    await run_cli_extract_async()
            finally:
                logger.propagate = original_propagate

        batch_loss_records = [r for r in caplog.records if r.levelno == logging.ERROR and "lost" in r.message]
        assert len(batch_loss_records) >= 1
