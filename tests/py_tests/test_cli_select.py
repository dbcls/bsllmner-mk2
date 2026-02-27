"""Tests for CLI select mode argument parsing."""

from pathlib import Path

import pytest

from bsllmner2.cli_select import parse_args
from bsllmner2.config import RESUME_BATCH_SIZE


class TestParseArgsSelect:
    """Test cases for cli_select.parse_args function."""

    def test_minimal_args(self, bs_entries_json_file: Path, select_config_file: Path) -> None:
        """Test parsing with minimal required arguments."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
        ]
        _config, cli_args = parse_args(args)

        assert cli_args.bs_entries == bs_entries_json_file.resolve()
        assert cli_args.select_config == select_config_file.resolve()
        assert cli_args.mapping is None
        assert cli_args.model == "llama3.1:70b"
        assert cli_args.thinking is None
        assert cli_args.max_entries is None
        assert cli_args.with_metrics is False
        assert cli_args.run_name is None
        assert cli_args.resume is False
        assert cli_args.batch_size == RESUME_BATCH_SIZE
        assert cli_args.include_reasoning is True

    def test_all_args(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        mapping_file: Path,
    ) -> None:
        """Test parsing with all arguments specified."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
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
            "--no-reasoning",
        ]
        config, cli_args = parse_args(args)  # config used below

        assert cli_args.bs_entries == bs_entries_json_file.resolve()
        assert cli_args.select_config == select_config_file.resolve()
        assert cli_args.mapping == mapping_file.resolve()
        assert cli_args.model == "qwen2.5:72b"
        assert cli_args.max_entries == 100
        assert cli_args.with_metrics is True
        assert cli_args.run_name == "test-run"
        assert cli_args.resume is True
        assert cli_args.batch_size == 512
        assert cli_args.include_reasoning is False
        assert config.ollama_host == "http://custom:11434"
        assert config.debug is True

    def test_thinking_flag_type(self, bs_entries_json_file: Path, select_config_file: Path) -> None:
        """Test that --thinking flag produces correct type (bool).

        cli_select uses str_to_bool to convert string to bool.
        This is the CORRECT behavior that cli_extract should also follow.
        """
        args_true = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--thinking",
            "true",
        ]
        _, cli_args_true = parse_args(args_true)

        args_false = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--thinking",
            "false",
        ]
        _, cli_args_false = parse_args(args_false)

        # cli_select correctly converts to bool
        assert cli_args_true.thinking is True
        assert cli_args_false.thinking is False
        assert isinstance(cli_args_true.thinking, bool)
        assert isinstance(cli_args_false.thinking, bool)

    def test_thinking_flag_case_insensitive(self, bs_entries_json_file: Path, select_config_file: Path) -> None:
        """Test that --thinking flag is case-insensitive."""
        args_upper = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--thinking",
            "TRUE",
        ]
        _, cli_args = parse_args(args_upper)
        assert cli_args.thinking is True

        args_mixed = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--thinking",
            "True",
        ]
        _, cli_args = parse_args(args_mixed)
        assert cli_args.thinking is True

    def test_max_entries_negative_becomes_none(self, bs_entries_json_file: Path, select_config_file: Path) -> None:
        """Test that negative max_entries becomes None."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--max-entries",
            "-1",
        ]
        _, cli_args = parse_args(args)
        assert cli_args.max_entries is None

    def test_missing_bs_entries_file(self, select_config_file: Path) -> None:
        """Test that missing bs_entries file raises FileNotFoundError."""
        args = [
            "--bs-entries",
            "/nonexistent/path/bs_entries.json",
            "--select-config",
            str(select_config_file),
        ]
        with pytest.raises(FileNotFoundError, match="BioSample entries file"):
            parse_args(args)

    def test_missing_select_config_file(self, bs_entries_json_file: Path) -> None:
        """Test that missing select_config file raises FileNotFoundError."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            "/nonexistent/select_config.json",
        ]
        with pytest.raises(FileNotFoundError, match="Select configuration file"):
            parse_args(args)

    def test_missing_required_args(self) -> None:
        """Test that missing required arguments causes SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


class TestThinkingTypeConsistency:
    """Test to verify type consistency between cli_extract and cli_select.

    Both cli_extract and cli_select now return bool for --thinking flag.
    cli_extract relies on Pydantic's auto-conversion, while cli_select
    uses explicit str_to_bool converter. The end result is the same.
    """

    def test_type_consistency_verified(
        self,
        bs_entries_json_file: Path,
        prompt_file: Path,
        select_config_file: Path,
    ) -> None:
        """Verify that both extract and select modes return bool for --thinking."""
        from bsllmner2.cli_extract import parse_args as parse_args_extract
        from bsllmner2.cli_select import parse_args as parse_args_select

        # Extract mode
        extract_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--prompt",
            str(prompt_file),
            "--thinking",
            "true",
        ]
        _, extract_cli_args = parse_args_extract(extract_args)

        # Select mode
        select_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--thinking",
            "true",
        ]
        _, select_cli_args = parse_args_select(select_args)

        # Both should return bool now
        extract_thinking_type = type(extract_cli_args.thinking)
        select_thinking_type = type(select_cli_args.thinking)

        assert extract_thinking_type is bool
        assert select_thinking_type is bool
        assert extract_thinking_type is select_thinking_type
