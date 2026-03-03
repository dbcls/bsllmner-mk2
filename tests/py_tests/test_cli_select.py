"""Tests for CLI select mode argument parsing and async execution."""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bsllmner2.benchmark import DiskIoTimings
from bsllmner2.cli_select import parse_args, run_cli_select_async
from bsllmner2.config import RESUME_BATCH_SIZE
from bsllmner2.models import LlmOutput, SelectResult
from tests.py_tests.conftest import FakeLlmBackend, make_chat_response

_EMPTY_INDEX_MAP_RESULT: tuple[dict[str, object], DiskIoTimings] = ({}, DiskIoTimings())
_EMPTY_SELECT_TIMINGS: dict[str, float] = {
    "ontology_search_sec": 0.0,
    "text2term_sec": 0.0,
    "llm_select_sec": 0.0,
}


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
        """Test that missing bs_entries file causes SystemExit via parser.error()."""
        args = [
            "--bs-entries",
            "/nonexistent/path/bs_entries.json",
            "--select-config",
            str(select_config_file),
        ]
        with pytest.raises(SystemExit):
            parse_args(args)

    def test_missing_select_config_file(self, bs_entries_json_file: Path) -> None:
        """Test that missing select_config file causes SystemExit via parser.error()."""
        args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            "/nonexistent/select_config.json",
        ]
        with pytest.raises(SystemExit):
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


# === CLI async integration test ===


@pytest.mark.asyncio(loop_scope="function")
class TestRunCliSelectAsync:
    async def test_basic_run(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """Smoke test: run_cli_select_async completes with all externals mocked."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
        ]

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        '{"cell_line": "HEK293"}',
                    ]
                ),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch("bsllmner2.cli_select.dump_extract_result", return_value=tmp_path / "extract.json"),
            patch("bsllmner2.cli_select.dump_select_result", return_value=tmp_path / "select.json"),
            patch("bsllmner2.cli_select.dump_extract_resume_file"),
            patch("bsllmner2.cli_select.dump_select_resume_file"),
            patch("bsllmner2.cli_select.remove_resume_files"),
            patch("bsllmner2.cli_select.dump_benchmark", return_value=tmp_path / "benchmark.json"),
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]
            await run_cli_select_async()

    async def test_resume_with_consistency_check(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """Resume mode skips entries that passed consistency validation."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--resume",
            "--run-name",
            "test-run",
        ]

        already_done_extract = LlmOutput(
            accession="SAMN00000001",
            output={"cell_line": "HeLa"},
            chat_response=make_chat_response('{"cell_line": "HeLa"}'),
        )
        already_done_select = SelectResult(
            accession="SAMN00000001",
            extract_output={"cell_line": "HeLa"},
        )

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend([]),
            ),
            patch(
                "bsllmner2.cli_select.load_extract_resume_file",
                return_value=[already_done_extract],
            ),
            patch(
                "bsllmner2.cli_select.load_select_resume_file",
                return_value=[already_done_select],
            ),
            patch(
                "bsllmner2.cli_select.validate_resume_consistency",
                return_value=({"SAMN00000001"}, set()),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch(
                "bsllmner2.cli_select.dump_extract_result",
                return_value=tmp_path / "extract.json",
            ),
            patch(
                "bsllmner2.cli_select.dump_select_result",
                return_value=tmp_path / "select.json",
            ),
            patch("bsllmner2.cli_select.dump_extract_resume_file"),
            patch("bsllmner2.cli_select.dump_select_resume_file"),
            patch("bsllmner2.cli_select.remove_resume_files"),
            patch("bsllmner2.cli_select.dump_benchmark", return_value=tmp_path / "benchmark.json"),
            patch(
                "bsllmner2.cli_select.ner",
                new_callable=AsyncMock,
                return_value=[
                    LlmOutput(
                        accession="SAMN00000002",
                        output={"cell_line": "HEK293"},
                        chat_response=make_chat_response('{"cell_line": "HEK293"}'),
                    ),
                ],
            ) as mock_ner,
            patch(
                "bsllmner2.cli_select.select",
                new_callable=AsyncMock,
                return_value=(
                    [
                        SelectResult(
                            accession="SAMN00000002",
                            extract_output={"cell_line": "HEK293"},
                        ),
                    ],
                    _EMPTY_SELECT_TIMINGS,
                ),
            ),
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]
            await run_cli_select_async()

            # ner should have been called with only SAMN00000002
            assert mock_ner.call_count == 1
            ner_call_entries = mock_ner.call_args[0][1]  # second positional arg
            accessions = [e["accession"] for e in ner_call_entries]
            assert "SAMN00000001" not in accessions
            assert "SAMN00000002" in accessions

    async def test_resume_orphan_entries_select_only(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """Orphan entries (extract done, select not done) run select only, not extract again."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
            "--resume",
            "--run-name",
            "test-run",
        ]

        # SAMN00000001 is done (both extract & select)
        done_extract = LlmOutput(
            accession="SAMN00000001",
            output={"cell_line": "HeLa"},
            chat_response=make_chat_response('{"cell_line": "HeLa"}'),
        )
        done_select = SelectResult(
            accession="SAMN00000001",
            extract_output={"cell_line": "HeLa"},
        )

        # SAMN00000002 is orphan (extract done, select not done)
        orphan_extract = LlmOutput(
            accession="SAMN00000002",
            output={"cell_line": "HEK293"},
            chat_response=make_chat_response('{"cell_line": "HEK293"}'),
        )

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend([]),
            ),
            patch(
                "bsllmner2.cli_select.load_extract_resume_file",
                return_value=[done_extract, orphan_extract],
            ),
            patch(
                "bsllmner2.cli_select.load_select_resume_file",
                return_value=[done_select],
            ),
            patch(
                "bsllmner2.cli_select.validate_resume_consistency",
                return_value=({"SAMN00000001"}, {"SAMN00000002"}),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch(
                "bsllmner2.cli_select.dump_extract_result",
                return_value=tmp_path / "extract.json",
            ),
            patch(
                "bsllmner2.cli_select.dump_select_result",
                return_value=tmp_path / "select.json",
            ),
            patch("bsllmner2.cli_select.dump_extract_resume_file"),
            patch("bsllmner2.cli_select.dump_select_resume_file"),
            patch("bsllmner2.cli_select.remove_resume_files"),
            patch("bsllmner2.cli_select.dump_benchmark", return_value=tmp_path / "benchmark.json"),
            patch(
                "bsllmner2.cli_select.ner",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_ner,
            patch(
                "bsllmner2.cli_select.select",
                new_callable=AsyncMock,
                return_value=(
                    [
                        SelectResult(
                            accession="SAMN00000002",
                            extract_output={"cell_line": "HEK293"},
                        ),
                    ],
                    _EMPTY_SELECT_TIMINGS,
                ),
            ) as mock_select,
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]
            await run_cli_select_async()

            # ner should NOT be called with the orphan entry
            # (no remaining entries after excluding done + orphan)
            if mock_ner.call_count > 0:
                for call in mock_ner.call_args_list:
                    ner_entries = call[0][1]
                    accessions = [e["accession"] for e in ner_entries]
                    assert "SAMN00000002" not in accessions

            # select should be called for orphan entries
            assert mock_select.call_count >= 1
            # Find the orphan select call
            orphan_select_found = False
            for call in mock_select.call_args_list:
                extract_outputs_arg = call[0][3]  # 4th positional arg
                acc_set = {o.accession for o in extract_outputs_arg}
                if "SAMN00000002" in acc_set:
                    orphan_select_found = True
            assert orphan_select_found

    async def test_failed_status_on_error(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """OllamaConnectionError during ner sets status to 'failed' in the result."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
        ]

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend(
                    # First chat call raises ConnectionError -> ner converts to OllamaConnectionError
                    [ConnectionError("refused")],
                ),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch(
                "bsllmner2.cli_select.dump_extract_result",
                return_value=tmp_path / "extract.json",
            ) as mock_dump_extract,
            patch(
                "bsllmner2.cli_select.dump_select_result",
                return_value=tmp_path / "select.json",
            ),
            patch("bsllmner2.cli_select.dump_extract_resume_file"),
            patch("bsllmner2.cli_select.dump_select_resume_file"),
            patch("bsllmner2.cli_select.remove_resume_files") as mock_remove,
            patch("bsllmner2.cli_select.dump_benchmark", return_value=tmp_path / "benchmark.json"),
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]
            await run_cli_select_async()

            # dump_extract_result should have been called with status "failed"
            assert mock_dump_extract.call_count == 1
            result_arg = mock_dump_extract.call_args[0][0]
            assert result_arg.run_metadata.status == "failed"

            # resume files should NOT be removed on failure
            mock_remove.assert_not_called()


# === CLI integration tests with real file I/O ===


@pytest.mark.asyncio(loop_scope="function")
class TestCliSelectIntegration:
    """Integration tests with FakeLlmBackend and real file I/O."""

    async def test_real_file_io(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: real files are written, output structure is valid, status is 'completed'."""
        extract_dir = tmp_path / "results" / "extract"
        select_dir = tmp_path / "results" / "select"
        progress_dir = tmp_path / "progress"

        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
        ]

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        '{"cell_line": "HEK293"}',
                    ]
                ),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch("bsllmner2.io.EXTRACT_RESULT_DIR", extract_dir),
            patch("bsllmner2.io.SELECT_RESULT_DIR", select_dir),
            patch("bsllmner2.io.PROGRESS_DIR", progress_dir),
            patch("bsllmner2.benchmark.BENCHMARK_DIR", tmp_path / "benchmarks"),
            patch("bsllmner2.cli_select.remove_resume_files"),
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]
            await run_cli_select_async()

        # Verify extract result files exist
        extract_files = [f for f in extract_dir.glob("*.json") if "_resume" not in f.name]
        assert len(extract_files) >= 1

        # Verify extract result structure
        extract_data = json.loads(extract_files[0].read_text())
        assert "run_metadata" in extract_data
        assert extract_data["run_metadata"]["status"] == "completed"

        # Verify select result files exist
        select_files = [f for f in select_dir.glob("*.json") if "_resume" not in f.name]
        assert len(select_files) >= 1

        # Verify select result is valid JSON array
        select_data = json.loads(select_files[0].read_text())
        assert isinstance(select_data, list)
        assert len(select_data) == 2

    async def test_batch_loss_logged_at_error(
        self,
        bs_entries_json_file: Path,
        select_config_file: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When ner() returns fewer outputs than entries, an ERROR is logged."""
        cli_args = [
            "--bs-entries",
            str(bs_entries_json_file),
            "--select-config",
            str(select_config_file),
        ]

        with (
            patch("bsllmner2.cli_select.sys") as mock_sys,
            patch(
                "bsllmner2.cli_select.OllamaBackend",
                return_value=FakeLlmBackend(
                    [
                        '{"cell_line": "HeLa"}',
                        RuntimeError("boom"),
                    ]
                ),
            ),
            patch("bsllmner2.cli_select.build_index_map", return_value=_EMPTY_INDEX_MAP_RESULT),
            patch("bsllmner2.cli_select.dump_extract_result", return_value=tmp_path / "extract.json"),
            patch("bsllmner2.cli_select.dump_select_result", return_value=tmp_path / "select.json"),
            patch("bsllmner2.cli_select.dump_extract_resume_file"),
            patch("bsllmner2.cli_select.dump_select_resume_file"),
            patch("bsllmner2.cli_select.remove_resume_files"),
            patch("bsllmner2.cli_select.dump_benchmark", return_value=tmp_path / "benchmark.json"),
        ):
            mock_sys.argv = ["bsllmner2-select", *cli_args]

            logger = logging.getLogger("bsllmner2")
            original_propagate = logger.propagate
            try:
                logger.propagate = True
                with caplog.at_level(logging.ERROR, logger="bsllmner2"):
                    await run_cli_select_async()
            finally:
                logger.propagate = original_propagate

        batch_loss_records = [r for r in caplog.records if r.levelno == logging.ERROR and "lost" in r.message]
        assert len(batch_loss_records) >= 1
