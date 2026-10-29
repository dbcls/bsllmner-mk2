"""Tests for the reselect CLI: argument parsing and the end-to-end run with a fake backend."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bsllmner2.cli_reselect import collect_targets, parse_args, run_cli_reselect_async
from bsllmner2.io import load_select_result
from bsllmner2.models import (
    DiskIoTimings,
    ExtractEntry,
    PerformanceSummary,
    ResolvedValue,
    RunMetadata,
    RunStatus,
    SearchResult,
    SelectEntry,
    SelectResult,
)
from tests.py_tests.conftest import FakeLlmBackend

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _source_result(path: Path, entries: list[SelectEntry], status: RunStatus = "completed") -> Path:
    result = SelectResult(
        entries=entries,
        run_metadata=RunMetadata(run_name="src_run", model="m", start_time="2026-10-01T00:00:00Z", status=status),
        performance=PerformanceSummary(total_input_entries=len(entries), completed_count=len(entries)),
    )
    path.write_text(json.dumps(result.model_dump(mode="json")), encoding="utf-8")
    return path


def _entry(
    accession: str, extracted: dict[str, Any] | None, results: dict[str, list[ResolvedValue]] | None = None
) -> SelectEntry:
    return SelectEntry(extract=ExtractEntry(accession=accession, extracted=extracted), results=results or {})


@pytest.fixture
def gene_select_config_file(temp_dir: Path) -> Path:
    owl = temp_dir / "genes.owl"
    owl.write_bytes(b"<rdf:RDF/>")
    config = {
        "fields": {
            "knockout_gene": {"ontology_file": str(owl), "value_type": "array"},
            "note": {"ontology_file": None, "value_type": "string"},
        }
    }
    path = temp_dir / "select-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def source_result_file(temp_dir: Path) -> Path:
    entries = [
        _entry(
            "SAMN00000001",
            {"knockout_gene": ["c-Myc", "TP53"]},
            {"knockout_gene": [ResolvedValue(value="TP53", term_id="NCBIGene:7157")]},
        ),
        _entry(
            "SAMN00000002",
            {"knockout_gene": ["PTEN"]},
            {"knockout_gene": [ResolvedValue(value="PTEN", term_id="NCBIGene:5728")]},
        ),
    ]
    return _source_result(temp_dir / "select_src_run.json", entries)


class TestParseArgsReselect:
    def test_minimal_args(
        self, bs_entries_json_file: Path, source_result_file: Path, gene_select_config_file: Path
    ) -> None:
        _config, args = parse_args(
            [
                "--bs-entries",
                str(bs_entries_json_file),
                "--from-result",
                str(source_result_file),
                "--select-config",
                str(gene_select_config_file),
            ]
        )
        assert args.from_result == source_result_file.resolve()
        assert args.select_config == gene_select_config_file.resolve()
        assert args.fields is None
        assert args.include_reasoning is True
        assert args.resume is False

    def test_fields_and_no_reasoning(
        self, bs_entries_json_file: Path, source_result_file: Path, gene_select_config_file: Path
    ) -> None:
        _config, args = parse_args(
            [
                "--bs-entries",
                str(bs_entries_json_file),
                "--from-result",
                str(source_result_file),
                "--select-config",
                str(gene_select_config_file),
                "--fields",
                "knockout_gene",
                "note",
                "--no-reasoning",
            ]
        )
        assert args.fields == ["knockout_gene", "note"]
        assert args.include_reasoning is False

    def test_resume_is_rejected(
        self, bs_entries_json_file: Path, source_result_file: Path, gene_select_config_file: Path
    ) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--bs-entries",
                    str(bs_entries_json_file),
                    "--from-result",
                    str(source_result_file),
                    "--select-config",
                    str(gene_select_config_file),
                    "--resume",
                ]
            )

    def test_missing_result_file(
        self, bs_entries_json_file: Path, gene_select_config_file: Path, temp_dir: Path
    ) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--bs-entries",
                    str(bs_entries_json_file),
                    "--from-result",
                    str(temp_dir / "nope.json"),
                    "--select-config",
                    str(gene_select_config_file),
                ]
            )

    def test_missing_required_args(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])


class TestCollectTargets:
    def test_only_entries_with_unresolved_values(self) -> None:
        entries = [
            _entry(
                "A", {"knockout_gene": ["c-Myc", "TP53"]}, {"knockout_gene": [ResolvedValue(value="TP53", term_id="x")]}
            ),
            _entry("B", {"knockout_gene": ["PTEN"]}, {"knockout_gene": [ResolvedValue(value="PTEN", term_id="y")]}),
            _entry("C", None),
        ]
        assert collect_targets(entries, ["knockout_gene"]) == {"A": {"knockout_gene": ["c-Myc"]}}


def _hit(term_id: str, value: str, score: float) -> SearchResult:
    return SearchResult(
        term_uri=f"http://example.org/{term_id}",
        term_id=term_id,
        prop_uri=RDFS_LABEL,
        value=value,
        label=value,
        exact_match=False,
        text2term_score=score,
    )


@pytest.mark.asyncio(loop_scope="function")
class TestRunCliReselectAsync:
    async def test_end_to_end_writes_only_reselected_entries(
        self,
        bs_entries_json_file: Path,
        source_result_file: Path,
        gene_select_config_file: Path,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        argv = [
            "bsllmner2_reselect",
            "--bs-entries",
            str(bs_entries_json_file),
            "--from-result",
            str(source_result_file),
            "--select-config",
            str(gene_select_config_file),
            "--fields",
            "knockout_gene",
            "--no-reasoning",
            "--run-name",
            "src_run_reselect",
            "--model",
            "test-model",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr("bsllmner2.io.SELECT_RESULT_DIR", temp_dir / "out")
        backend = FakeLlmBackend(['{"id": "NCBIGene:4609"}'])
        with (
            patch("bsllmner2.cli_reselect.OllamaBackend", return_value=backend),
            patch("bsllmner2.cli_reselect.build_index_map", return_value=({}, DiskIoTimings())),
            patch("bsllmner2.cli_reselect.build_text2term_cache", return_value=DiskIoTimings()),
            patch("bsllmner2.select.search_terms", return_value={}),
            patch(
                "bsllmner2.select.search_terms_with_text2term",
                return_value={"c-Myc": [_hit("NCBIGene:4609", "MYC", 0.9)]},
            ),
        ):
            await run_cli_reselect_async()

        written = load_select_result(temp_dir / "out" / "select_src_run_reselect.json")
        assert [e.extract.accession for e in written.entries] == ["SAMN00000001"]
        entry = written.entries[0]
        assert [(rv.value, rv.term_id) for rv in entry.results["knockout_gene"]] == [
            ("TP53", "NCBIGene:7157"),
            ("c-Myc", "NCBIGene:4609"),
        ]
        meta = written.run_metadata
        assert meta.status == "completed"
        assert meta.total_entries == 1
        assert meta.reselect is not None
        assert meta.reselect.source_result_file == "select_src_run.json"
        assert meta.reselect.source_run_name == "src_run"
        assert meta.reselect.fields == ["knockout_gene"]
        assert len(meta.reselect.source_sha256) == 64
        assert written.performance is not None
        assert written.performance.total_input_entries == 1
        assert written.performance.completed_count == 1
        assert written.performance.ner_llm_timing is None
        assert written.performance.stage_timings[0].ner_sec == 0.0

    async def test_target_entry_missing_from_bs_entries_aborts(
        self,
        source_result_file: Path,
        gene_select_config_file: Path,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        other = temp_dir / "other.json"
        other.write_text(json.dumps([{"accession": "SAMN99999999"}]), encoding="utf-8")
        argv = [
            "bsllmner2_reselect",
            "--bs-entries",
            str(other),
            "--from-result",
            str(source_result_file),
            "--select-config",
            str(gene_select_config_file),
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr("bsllmner2.io.SELECT_RESULT_DIR", temp_dir / "out")
        with (
            patch("bsllmner2.cli_reselect.OllamaBackend", return_value=FakeLlmBackend([])),
            pytest.raises(ValueError, match="not in"),
        ):
            await run_cli_reselect_async()
        assert not (temp_dir / "out").exists()
