"""Tests for scripts/ncbi_gene_to_owl.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from rdflib import RDF, Graph
from rdflib.namespace import OWL

from bsllmner2.ontology_search import build_index_from_owl

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ncbi_gene_to_owl.py"

# Columns of ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz (16 columns).
GENE_INFO_HEADER = "\t".join([
    "#tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs",
    "chromosome", "map_location", "description", "type_of_gene",
    "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority",
    "Nomenclature_status", "Other_designations", "Modification_date", "Feature_type",
])
_TAIL = "\t".join(["-"] * 6)  # Columns 11-16 unused by the script.
ROW_YAP1 = f"9606\t10413\tYAP1\t-\tCOB1|YAP|YAP-1|YAP2|YAP65|YKI\t-\tchr11\t-\tYes1 associated transcriptional regulator\tprotein-coding\t{_TAIL}"
ROW_CTNNB1 = f"9606\t1499\tCTNNB1\t-\tCTNNB\t-\tchr3\t-\tcatenin beta 1\tprotein-coding\t{_TAIL}"
ROW_MOUSE_YAP1 = f"10090\t22060\tYap1\t-\tSyn1|Syn2\t-\tchr9\t-\tmouse YAP1\tprotein-coding\t{_TAIL}"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_ncbi_gene_to_owl_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mini_gene_info(tmp_path: Path) -> Path:
    p = tmp_path / "gene_info"
    p.write_text(
        "\n".join([GENE_INFO_HEADER, ROW_YAP1, ROW_CTNNB1, ROW_MOUSE_YAP1]) + "\n",
        encoding="utf-8",
    )
    return p


def _run_script(
    tmp_path: Path,
    mini_gene_info: Path,
    monkeypatch: pytest.MonkeyPatch,
    taxid: str,
) -> ModuleType:
    mod = _load_script_module()
    monkeypatch.setattr(mod, "ONTOLOGY_DIR", tmp_path)
    monkeypatch.setattr(mod, "NCBI_GENE_FILE", mini_gene_info)
    monkeypatch.setattr(sys, "argv", ["ncbi_gene_to_owl.py", "--taxid", taxid])
    mod.main()
    return mod


class TestAnnotationPropertyDeclarations:
    """Regression: hasExactSynonym / IAO_0000115 must be declared as owl:AnnotationProperty.

    Without explicit declarations, owlready2 silently drops the corresponding values
    when loading the OWL, so synonym / definition based ontology lookups return zero
    candidates. See bsllmner2.ontology_search.iter_term_annotations.
    """

    def test_synonym_is_recognized_by_owlready2(
        self, tmp_path: Path, mini_gene_info: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _run_script(tmp_path, mini_gene_info, monkeypatch, taxid="9606")
        owl_path = tmp_path / "ncbi_gene_human.owl"
        idx = build_index_from_owl(owl_path)

        anns = idx.value_to_annotations.get("yap", [])
        assert any(a.term_id == "NCBIGene:10413" for a in anns), (
            "YAP synonym of NCBIGene:10413 (YAP1) must be indexed from hasExactSynonym."
        )

    def test_definition_is_recognized_by_owlready2(
        self, tmp_path: Path, mini_gene_info: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _run_script(tmp_path, mini_gene_info, monkeypatch, taxid="9606")
        owl_path = tmp_path / "ncbi_gene_human.owl"
        idx = build_index_from_owl(owl_path)

        defs = idx.term_id_to_definitions.get("NCBIGene:10413")
        assert defs is not None
        assert any("Yes1 associated" in d for d in defs), (
            "IAO_0000115 description of NCBIGene:10413 must be indexed."
        )

    def test_annotation_properties_are_declared_in_owl(
        self, tmp_path: Path, mini_gene_info: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod = _run_script(tmp_path, mini_gene_info, monkeypatch, taxid="9606")
        owl_path = tmp_path / "ncbi_gene_human.owl"

        g = Graph()
        g.parse(owl_path)
        assert (mod.OBOINOWL.hasExactSynonym, RDF.type, OWL.AnnotationProperty) in g
        assert (mod.IAO_DEFINITION, RDF.type, OWL.AnnotationProperty) in g


class TestTaxidFiltering:
    """--taxid filter must keep only the selected species."""

    def test_human_run_excludes_mouse(
        self, tmp_path: Path, mini_gene_info: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _run_script(tmp_path, mini_gene_info, monkeypatch, taxid="9606")
        owl_path = tmp_path / "ncbi_gene_human.owl"
        idx = build_index_from_owl(owl_path)

        assert "NCBIGene:10413" in idx.term_id_to_labels
        assert "NCBIGene:1499" in idx.term_id_to_labels
        assert "NCBIGene:22060" not in idx.term_id_to_labels

    def test_mouse_run_writes_mouse_owl_and_excludes_human(
        self, tmp_path: Path, mini_gene_info: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _run_script(tmp_path, mini_gene_info, monkeypatch, taxid="10090")
        human_owl = tmp_path / "ncbi_gene_human.owl"
        mouse_owl = tmp_path / "ncbi_gene_mouse.owl"
        assert not human_owl.exists()
        assert mouse_owl.exists()

        idx = build_index_from_owl(mouse_owl)
        assert "NCBIGene:22060" in idx.term_id_to_labels
        assert "NCBIGene:10413" not in idx.term_id_to_labels
