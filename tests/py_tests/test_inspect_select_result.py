"""Tests for scripts/inspect_select_result.py."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_select_result.py"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mini_select_result.json"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_inspect_select_result_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script_module() -> ModuleType:
    return _load_script_module()


def _run(
    script_module: ModuleType,
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    code = script_module.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestSummaryPlainText:
    def test_run_header_renders_metadata(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH)], capsys)
        assert code == 0
        assert "Run" in out
        assert "mini-fixture" in out
        assert "mistral-small3.1:24b" in out
        assert "thinking: false" in out
        assert "status:   completed" in out
        assert "15m00s" in out
        assert "entries:  3/3" in out
        assert "errors: 0" in out

    def test_mapping_rate_rows_per_field(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH)], capsys)
        assert "Mapping rate" in out
        # cell_line: 2 extracted / 2 mapped
        assert "cell_line" in out
        # tissue: 1 extracted / 0 mapped
        assert "tissue" in out
        # gene: 0 extracted - rate should render as "-"
        assert "gene" in out

    def test_not_found_section_lists_unmapped_values(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH)], capsys)
        assert "NOT_FOUND top" in out
        assert "normal (1)" in out
        assert "PBMC (1)" in out
        # fields with no unmapped values are flagged as "(none)"
        assert "cell_line: (none)" in out

    def test_llm_timing_section(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH)], capsys)
        assert "LLM timing" in out
        assert "NER: 3 calls" in out
        assert "Select: 1 calls" in out

    def test_evaluation_section_rendered_when_present(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH)], capsys)
        assert "Evaluation" in out
        assert "accuracy" in out
        assert "0.75" in out
        assert "f1" in out

    def test_top_nf_limits_not_found_list(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["summary", str(FIXTURE_PATH), "--top-nf", "1"],
            capsys,
        )
        # fixture has 1 unmapped per field, so --top-nf 1 still shows it
        assert "normal (1)" in out


class TestSummaryJson:
    def test_json_structure(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(script_module, ["summary", str(FIXTURE_PATH), "--json"], capsys)
        payload: dict[str, Any] = json.loads(out)
        assert set(payload.keys()) == {
            "run",
            "mapping_rate",
            "not_found_top",
            "llm_timing",
            "evaluation",
        }
        assert payload["run"]["name"] == "mini-fixture"
        assert payload["mapping_rate"]["cell_line"]["mapped"] == 2
        assert payload["mapping_rate"]["tissue"]["mapped"] == 0
        assert payload["mapping_rate"]["gene"]["rate"] is None
        assert payload["not_found_top"]["tissue"] == [{"value": "PBMC", "count": 1}]
        assert payload["llm_timing"]["ner"]["call_count"] == 3
        assert payload["llm_timing"]["select"]["call_count"] == 1
        assert payload["evaluation"]["f1"] == pytest.approx(0.857)


class TestShowAccession:
    def test_exact_source_label(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--accession", "SAMX001"],
            capsys,
        )
        assert code == 0
        assert "Entry SAMX001" in out
        assert "HeLa -> HeLa (CVCL:0030) [exact]" in out
        assert "GSK1210151A -> I-BET151 (CHEBI:95083) [exact]" in out
        assert "normal -> NOT_FOUND" in out

    def test_llm_source_label(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--accession", "SAMX002"],
            capsys,
        )
        assert "B-lymphoblastoid -> B-lymphoblast (CL:0017006) [llm]" in out
        assert "PBMC -> NOT_FOUND" in out

    def test_text2term_source_label(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--accession", "SAMX003"],
            capsys,
        )
        assert "epithelial cell -> epithelial cell (CL:0000066) [text2term]" in out
        assert "H1299 -> NCI-H1299 (CVCL:0060) [exact]" in out

    def test_missing_accession_returns_non_zero_with_stderr(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, out, err = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--accession", "SAM_MISSING"],
            capsys,
        )
        assert code == 1
        assert "SAM_MISSING" in err
        assert out == ""

    def test_json_output(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--accession", "SAMX002", "--json"],
            capsys,
        )
        payload = json.loads(out)
        assert payload["accession"] == "SAMX002"
        cell_type_rows = payload["fields"]["cell_type"]
        assert len(cell_type_rows) == 1
        assert cell_type_rows[0]["source"] == "llm"
        assert cell_type_rows[0]["resolved"]["term_id"] == "CL:0017006"
        tissue_rows = payload["fields"]["tissue"]
        assert tissue_rows[0]["mapped"] is False
        assert tissue_rows[0]["resolved"] is None
        assert {"field": "tissue", "value": "PBMC"} in payload["unmapped"]


class TestShowUnmappedOnly:
    def test_lists_entries_with_unmapped(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--unmapped-only"],
            capsys,
        )
        # SAMX001 (disease=normal), SAMX002 (tissue=PBMC) → 2 entries
        # SAMX003 has no unmapped values
        assert "Entries with unmapped values: 2" in out
        assert "Entry SAMX001" in out
        assert "Entry SAMX002" in out
        assert "Entry SAMX003" not in out

    def test_limit_truncates_entry_count(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--unmapped-only", "--limit", "1"],
            capsys,
        )
        assert "Entries with unmapped values: 1" in out
        assert "Entry SAMX001" in out
        assert "Entry SAMX002" not in out

    def test_json_output(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["show", str(FIXTURE_PATH), "--unmapped-only", "--json"],
            capsys,
        )
        payload = json.loads(out)
        assert payload["count"] == 2
        accessions = [e["accession"] for e in payload["entries"]]
        assert accessions == ["SAMX001", "SAMX002"]


class TestFind:
    def test_match_with_mapped_entry(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["find", str(FIXTURE_PATH), "--field", "cell_line", "--value", "HeLa"],
            capsys,
        )
        assert "total:    1" in out
        assert "mapped:   1" in out
        assert "unmapped: 0" in out
        assert "SAMX001 -> HeLa (CVCL:0030) [exact]" in out

    def test_match_with_unmapped_entry(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["find", str(FIXTURE_PATH), "--field", "tissue", "--value", "PBMC"],
            capsys,
        )
        assert "total:    1" in out
        assert "unmapped: 1" in out
        assert "Unmapped (1):" in out
        assert "SAMX002" in out

    def test_no_matches(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["find", str(FIXTURE_PATH), "--field", "disease", "--value", "no_such_disease"],
            capsys,
        )
        assert "total:    0" in out
        assert "Mapped" not in out
        assert "Unmapped" not in out

    def test_json_structure(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _, out, _ = _run(
            script_module,
            ["find", str(FIXTURE_PATH), "--field", "cell_line", "--value", "HeLa", "--json"],
            capsys,
        )
        payload = json.loads(out)
        assert payload["field"] == "cell_line"
        assert payload["value"] == "HeLa"
        assert payload["total"] == 1
        assert payload["mapped"] == 1
        assert payload["entries"][0]["accession"] == "SAMX001"
        assert payload["entries"][0]["source"] == "exact"

    def test_case_sensitive_match(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Find matches on exact string equality, not case-insensitively."""
        _, out, _ = _run(
            script_module,
            ["find", str(FIXTURE_PATH), "--field", "cell_line", "--value", "hela"],
            capsys,
        )
        assert "total:    0" in out


class TestLoader:
    def test_coerces_legacy_thinking_null(
        self,
        script_module: ModuleType,
        tmp_path: Path,
    ) -> None:
        """run_metadata.thinking=null from older runs must still parse."""
        raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        raw["run_metadata"]["thinking"] = None
        p = tmp_path / "legacy_thinking.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        sr = script_module.load_select_result(p)
        assert sr.run_metadata.thinking is False


class TestCli:
    def test_show_requires_one_of_accession_or_unmapped_only(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit):
            script_module.main(["show", str(FIXTURE_PATH)])
        captured = capsys.readouterr()
        assert "required" in captured.err.lower() or "one of" in captured.err.lower()

    def test_unknown_subcommand_exits(
        self,
        script_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit):
            script_module.main(["bogus"])
