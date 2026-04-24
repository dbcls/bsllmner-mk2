"""Tests for scripts/preprocess_cellosaurus.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "preprocess_cellosaurus.py"


MINI_OBO = """format-version: 1.2
default-namespace: cellosaurus
subsetdef: Cancer_cell_line "Cancer cell line"
subsetdef: Transformed_cell_line "Transformed cell line"
subsetdef: Hybridoma "Hybridoma"
subsetdef: Female "Female"
subsetdef: Male "Male"
subsetdef: Mixed_sex "Mixed sex"
ontology: Cellosaurus

[Term]
id: CVCL_HUMA
name:HumanCancerCell
synonym: "HCC" RELATED []
subset: Cancer_cell_line
subset: Female
xref: NCIt:C12345 ! cervical cancer
xref: NCBI_TaxID:9606 ! Homo sapiens (Human)
comment: "Legacy comment that should be stripped"
relationship: derived_from CVCL_PARENT ! ParentHumanCell
creation_date: 2020-01-01T00:00:00Z

[Term]
id: CVCL_NODIS
name:NoDiseaseHuman
subset: Transformed_cell_line
subset: Mixed_sex
xref: NCBI_TaxID:9606 ! Homo sapiens (Human)

[Term]
id: CVCL_MOUS
name:MouseHybrid
subset: Hybridoma
subset: Male
xref: NCIt:C23456 ! mouse tumor
xref: NCBI_TaxID:10090 ! Mus musculus (Mouse)
relationship: derived_from CVCL_MPARENT ! ParentMouseCell
creation_date: 2021-01-01T00:00:00Z

[Term]
id: CVCL_BARE
name:BareMouseCell
xref: NCBI_TaxID:10090 ! Mus musculus (Mouse)
"""


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_preprocess_cellosaurus_under_test",
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


@pytest.fixture
def mini_obo_file(tmp_path: Path) -> Path:
    p = tmp_path / "cellosaurus.obo"
    p.write_text(MINI_OBO, encoding="utf-8")
    return p


def _run_main_on_fixture(
    script_module: ModuleType,
    mini_obo_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    taxid: str,
) -> Path:
    """Invoke main() with INPUT_FILE and ONTOLOGY_DIR patched onto tmp_path."""
    monkeypatch.setattr(script_module, "INPUT_FILE", mini_obo_file)
    monkeypatch.setattr(script_module, "ONTOLOGY_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["preprocess_cellosaurus.py", "--taxid", taxid])
    script_module.main()
    return cast("Path", script_module.output_path_for(taxid))


def _read_output(path: Path) -> str:
    assert path.exists()
    return path.read_text(encoding="utf-8")


class TestOutputPath:
    def test_human_suffix(self, script_module: ModuleType) -> None:
        p = script_module.output_path_for("9606")
        assert p.name == "cellosaurus_human.mod.obo"

    def test_mouse_suffix(self, script_module: ModuleType) -> None:
        p = script_module.output_path_for("10090")
        assert p.name == "cellosaurus_mouse.mod.obo"

    def test_unknown_taxid_uses_literal(self, script_module: ModuleType) -> None:
        p = script_module.output_path_for("7227")
        assert p.name == "cellosaurus_7227.mod.obo"


class TestTaxidFiltering:
    def test_human_run_keeps_only_human_terms(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        assert out_path == tmp_path / "cellosaurus_human.mod.obo"
        text = _read_output(out_path)
        assert "id: CVCL_HUMA" in text
        assert "id: CVCL_NODIS" in text
        assert "id: CVCL_MOUS" not in text
        assert "id: CVCL_BARE" not in text

    def test_mouse_run_keeps_only_mouse_terms(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "10090")
        assert out_path == tmp_path / "cellosaurus_mouse.mod.obo"
        text = _read_output(out_path)
        assert "id: CVCL_MOUS" in text
        assert "id: CVCL_BARE" in text
        assert "id: CVCL_HUMA" not in text
        assert "id: CVCL_NODIS" not in text


class TestDefinitionSynthesis:
    def test_full_attributes_term(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        # HumanCancerCell: Female + cervical cancer + derived_from ParentHumanCell +
        # Cancer_cell_line + Homo sapiens
        assert (
            'def: "Female cervical cancer cell line derived from ParentHumanCell; '
            'Category: Cancer cell line; Species of origin: Homo sapiens" []'
        ) in text

    def test_missing_disease_skipped(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        # NoDiseaseHuman: no Disease, no derived_from, Mixed_sex → main has no sex prefix
        assert (
            'def: "cell line; Category: Transformed cell line; Sex: Mixed sex; Species of origin: Homo sapiens" []'
        ) in text

    def test_bare_term_falls_back_to_species_only(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "10090")
        text = _read_output(out_path)
        # BareMouseCell has only NCBI_TaxID; definition is "cell line; Species of origin: Mus musculus"
        assert 'def: "cell line; Species of origin: Mus musculus" []' in text

    def test_male_sex_included_inline(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "10090")
        text = _read_output(out_path)
        # MouseHybrid: Male + mouse tumor + derived_from ParentMouseCell + Hybridoma + Mus musculus
        assert (
            'def: "Male mouse tumor cell line derived from ParentMouseCell; '
            'Category: Hybridoma; Species of origin: Mus musculus" []'
        ) in text


class TestCommentAnnotations:
    def test_disease_comment_preserved(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "comment: Disease: cervical cancer" in text

    def test_derived_from_comment_preserved(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "comment: derived_from: ParentHumanCell" in text


class TestLineFiltering:
    def test_legacy_comment_lines_removed(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "Legacy comment that should be stripped" not in text

    def test_creation_date_lines_removed(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "creation_date:" not in text

    def test_raw_relationship_lines_removed(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """relationship: derived_from is consumed into a comment, the raw line is dropped."""
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "relationship: derived_from" not in text

    def test_ncbi_taxid_xref_preserved(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "xref: NCBI_TaxID:9606 ! Homo sapiens (Human)" in text

    def test_non_taxid_xref_removed(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert "xref: NCIt:C12345" not in text


class TestHeaderPassthrough:
    def test_subsetdef_lines_preserved_in_header(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_path = _run_main_on_fixture(script_module, mini_obo_file, tmp_path, monkeypatch, "9606")
        text = _read_output(out_path)
        assert 'subsetdef: Cancer_cell_line "Cancer cell line"' in text
        assert 'subsetdef: Female "Female"' in text


class TestComposeDefinition:
    """compose_definition is exercised through main; cover pure-logic edge cases here."""

    def test_all_present(self, script_module: ModuleType) -> None:
        out = script_module.compose_definition(
            disease="cervical cancer",
            category="Cancer cell line",
            sex="Female",
            species="Homo sapiens",
            derived_from="Parent",
        )
        assert out == (
            "Female cervical cancer cell line derived from Parent; "
            "Category: Cancer cell line; Species of origin: Homo sapiens"
        )

    def test_non_binary_sex_goes_to_tail(self, script_module: ModuleType) -> None:
        out = script_module.compose_definition(
            disease=None,
            category=None,
            sex="Sex ambiguous",
            species=None,
            derived_from=None,
        )
        assert out == "cell line; Sex: Sex ambiguous"

    def test_all_none_falls_back_to_cell_line(self, script_module: ModuleType) -> None:
        out = script_module.compose_definition(
            disease=None,
            category=None,
            sex=None,
            species=None,
            derived_from=None,
        )
        assert out == "cell line"


class TestSubsetDescriptionMap:
    def test_parses_header_subsetdefs(
        self,
        script_module: ModuleType,
        mini_obo_file: Path,
    ) -> None:
        mapping = script_module.build_subset_description_map(mini_obo_file)
        assert mapping["Cancer_cell_line"] == "Cancer cell line"
        assert mapping["Mixed_sex"] == "Mixed sex"
        # [Term]-level subset lines must not leak into the header map
        assert "CVCL_HUMA" not in mapping
