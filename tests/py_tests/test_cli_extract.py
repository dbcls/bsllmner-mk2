"""Tests for CLI extract mode argument parsing."""

from pathlib import Path

import pytest

from bsllmner2.cli_extract import parse_args
from bsllmner2.config import RESUME_BATCH_SIZE


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
        assert cli_args.mapping is None
        assert cli_args.format is None
        assert cli_args.model == "llama3.1:70b"
        assert cli_args.thinking is None
        assert cli_args.max_entries is None
        assert cli_args.with_metrics is False
        assert cli_args.run_name is None
        assert cli_args.resume is False
        assert cli_args.batch_size == RESUME_BATCH_SIZE

    def test_all_args(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        format_schema_file: Path,
        mapping_file: Path,
    ) -> None:
        """Test parsing with all arguments specified."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--format",
            str(format_schema_file),
            "--mapping",
            str(mapping_file),
            "--model",
            "qwen2.5:72b",
            "--thinking",
            "true",
            "--max-entries",
            "100",
            "--ollama-host",
            "http://custom:11434",
            "--with-metrics",
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
        assert cli_args.mapping == mapping_file.resolve()
        assert cli_args.model == "qwen2.5:72b"
        assert cli_args.max_entries == 100
        assert cli_args.with_metrics is True
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
        """Test that missing bs_entries file raises FileNotFoundError."""
        args = [
            "--bs-entries",
            "/nonexistent/path/bs_entries.json",
            "--prompt",
            str(prompt_file),
        ]
        with pytest.raises(FileNotFoundError, match="BioSample entries file"):
            parse_args(args)

    def test_missing_prompt_file(self, bs_entries_json_file: Path, temp_dir: Path) -> None:
        """Test that missing prompt file raises FileNotFoundError."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(temp_dir / "nonexistent_prompt.yml"),
        ]
        with pytest.raises(FileNotFoundError, match="Prompt file"):
            parse_args(args)

    def test_missing_format_file(self, bs_entries_json_file: Path, prompt_file: Path) -> None:
        """Test that missing format file raises FileNotFoundError."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--format",
            "/nonexistent/format.schema.json",
        ]
        with pytest.raises(FileNotFoundError, match="Format schema file"):
            parse_args(args)

    def test_missing_required_args(self) -> None:
        """Test that missing required arguments causes SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])
