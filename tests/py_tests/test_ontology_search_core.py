"""Integration tests for the core of ontology_search using a small OWL fixture."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from bsllmner2.models import OntologyIndex
from bsllmner2.ontology_search import (
    build_index_from_file,
    build_index_from_owl,
    build_index_from_table,
    build_text2term_cache_for_owl,
    search_terms,
    search_terms_with_text2term,
    text2term_cache_exists,
)

FIXTURE_OWL = Path(__file__).parent.parent / "data" / "ontology_fixture.owl"


@pytest.fixture(scope="module")
def fixture_index() -> OntologyIndex:
    return build_index_from_owl(FIXTURE_OWL)


class TestBuildIndexFromOwl:
    def test_index_contains_expected_term_ids(self, fixture_index: OntologyIndex) -> None:
        assert "CL:0000000" in fixture_index.term_id_to_labels
        assert "CL:0000057" in fixture_index.term_id_to_labels
        assert "CVCL:0030" in fixture_index.term_id_to_labels
        assert "CVCL:0063" in fixture_index.term_id_to_labels

    def test_labels_are_primary_rdfs_label(self, fixture_index: OntologyIndex) -> None:
        assert fixture_index.term_id_to_labels["CVCL:0030"][0] == "HeLa"
        assert fixture_index.term_id_to_labels["CL:0000057"][0] == "fibroblast"

    def test_comments_are_collected(self, fixture_index: OntologyIndex) -> None:
        comments = fixture_index.term_id_to_comments.get("CL:0000000")
        assert comments is not None
        assert any("structural unit" in c for c in comments)

    def test_definitions_are_collected(self, fixture_index: OntologyIndex) -> None:
        defs = fixture_index.term_id_to_definitions.get("CL:0000000")
        assert defs is not None
        assert any("anatomical entity" in d for d in defs)

    def test_synonyms_reach_value_index(self, fixture_index: OntologyIndex) -> None:
        # oboInOwl:hasExactSynonym "HeLa cell" should be searchable.
        anns = fixture_index.value_to_annotations.get("hela cell")
        assert anns is not None
        assert any(a.term_id == "CVCL:0030" for a in anns)


class TestSearchTermsExactMatch:
    def test_exact_label_match(self, fixture_index: OntologyIndex) -> None:
        results = search_terms(fixture_index, ["HeLa"])
        hits = results["HeLa"]
        assert any(r.term_id == "CVCL:0030" and r.exact_match for r in hits)

    def test_exact_synonym_match(self, fixture_index: OntologyIndex) -> None:
        results = search_terms(fixture_index, ["fibroblast cell"])
        hits = results["fibroblast cell"]
        assert any(r.term_id == "CL:0000057" and r.exact_match for r in hits)

    def test_case_insensitive_match(self, fixture_index: OntologyIndex) -> None:
        results = search_terms(fixture_index, ["HELA"])
        hits = results["HELA"]
        assert any(r.term_id == "CVCL:0030" and r.exact_match for r in hits)

    def test_no_match_returns_no_entry(self, fixture_index: OntologyIndex) -> None:
        results = search_terms(fixture_index, ["definitely-not-in-ontology"])
        assert "definitely-not-in-ontology" not in results

    def test_empty_queries_returns_empty(self, fixture_index: OntologyIndex) -> None:
        assert search_terms(fixture_index, []) == {}

    def test_exact_match_reasoning_cites_property(self, fixture_index: OntologyIndex) -> None:
        results = search_terms(fixture_index, ["HeLa"])
        exact = next(r for r in results["HeLa"] if r.exact_match)
        assert exact.reasoning is not None
        assert "Exact match" in exact.reasoning


class TestSearchTermsCombinations:
    def test_synonym_matches_label_with_dash(self, fixture_index: OntologyIndex) -> None:
        # Label "K-562" with synonym "K562"; the synonym matches via normalized key.
        results = search_terms(fixture_index, ["K562"])
        hits = results.get("K562", [])
        assert any(r.term_id == "CVCL:0063" for r in hits)

    def test_exact_synonym_with_space_matches(self, fixture_index: OntologyIndex) -> None:
        # oboInOwl:hasExactSynonym "HeLa cell" is indexed via _normalize_key("HeLa cell").
        results = search_terms(fixture_index, ["hela cell"])
        hits = results.get("hela cell", [])
        assert any(r.term_id == "CVCL:0030" and r.exact_match for r in hits)


class TestBuildIndexFromFile:
    def test_owl_dispatch(self, fixture_index: OntologyIndex) -> None:
        built = build_index_from_file(FIXTURE_OWL)
        assert set(built.term_id_to_labels.keys()) == set(fixture_index.term_id_to_labels.keys())

    def test_tsv_dispatch(self, tmp_path: Path) -> None:
        tsv = tmp_path / "mini.tsv"
        tsv.write_text(
            "CVCL_0030\trdfs:label\tHeLa\nCVCL_0063\trdfs:label\tK-562\n",
            encoding="utf-8",
        )
        index = build_index_from_file(tsv)
        assert "CVCL:0030" in index.term_id_to_labels
        assert index.term_id_to_labels["CVCL:0030"][0] == "HeLa"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "ontology.xyz"
        bogus.write_text("anything", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported ontology file format"):
            build_index_from_file(bogus)


class TestBuildIndexFromTable:
    def test_csv_parses(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "mini.csv"
        csv_file.write_text(
            "CVCL_0030,rdfs:label,HeLa\n",
            encoding="utf-8",
        )
        index = build_index_from_table(csv_file)
        assert "CVCL:0030" in index.term_id_to_labels

    def test_short_rows_are_skipped(self, tmp_path: Path) -> None:
        tsv = tmp_path / "short.tsv"
        tsv.write_text("CVCL_0030\trdfs:label\nCVCL_0063\trdfs:label\tK-562\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "CVCL:0063" in index.term_id_to_labels
        assert "CVCL:0030" not in index.term_id_to_labels


class TestSearchTermsWithText2Term:
    """text2term itself is the external boundary -- mock it, not the wrapper."""

    def _make_df(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            rows, columns=["Source Term", "Mapped Term Label", "Mapped Term IRI", "Mapped Term CURIE", "Mapping Score"]
        )

    def test_returns_decorated_search_result(self, fixture_index: OntologyIndex) -> None:
        df = self._make_df(
            [
                {
                    "Source Term": "Henrietta Lacks",
                    "Mapped Term Label": "HeLa",
                    "Mapped Term IRI": "http://purl.obolibrary.org/obo/CVCL_0030",
                    "Mapped Term CURIE": "CVCL:0030",
                    "Mapping Score": 0.85,
                },
            ],
        )
        with patch("bsllmner2.ontology_search.text2term.map_terms", return_value=df):
            results = search_terms_with_text2term(
                ["Henrietta Lacks"],
                FIXTURE_OWL,
                index=fixture_index,
            )

        assert results["Henrietta Lacks"]
        hit = results["Henrietta Lacks"][0]
        assert hit.term_id == "CVCL:0030"
        assert hit.label == "HeLa"
        assert hit.text2term_score == pytest.approx(0.85)
        assert hit.exact_match is False

    def test_drops_results_without_index_evidence(self, fixture_index: OntologyIndex) -> None:
        df = self._make_df(
            [
                {
                    "Source Term": "phantom",
                    "Mapped Term Label": "not-in-fixture",
                    "Mapped Term IRI": "http://purl.obolibrary.org/obo/PHANTOM_1",
                    "Mapped Term CURIE": "PHANTOM:1",
                    "Mapping Score": 0.99,
                },
            ],
        )
        with patch("bsllmner2.ontology_search.text2term.map_terms", return_value=df):
            results = search_terms_with_text2term(
                ["phantom"],
                FIXTURE_OWL,
                index=fixture_index,
            )

        assert results["phantom"] == []

    def test_missing_required_columns_raises(self, fixture_index: OntologyIndex) -> None:
        df = pd.DataFrame([{"Source Term": "x", "Mapped Term Label": "y"}])
        with (
            patch("bsllmner2.ontology_search.text2term.map_terms", return_value=df),
            pytest.raises(ValueError, match="Expected columns missing"),
        ):
            search_terms_with_text2term(["x"], FIXTURE_OWL, index=fixture_index)


class TestText2TermCacheHelpers:
    def test_cache_exists_delegates_to_text2term(self, tmp_path: Path) -> None:
        with patch("bsllmner2.ontology_search.text2term.cache_exists", return_value=True) as mock_fn:
            assert text2term_cache_exists("ACRONYM", tmp_path) is True
            mock_fn.assert_called_once_with("ACRONYM", cache_folder=str(tmp_path))

    def test_build_cache_delegates_to_text2term(self, tmp_path: Path) -> None:
        with patch("bsllmner2.ontology_search.text2term.cache_ontology") as mock_fn:
            build_text2term_cache_for_owl(FIXTURE_OWL, "ACRONYM", tmp_path)
            mock_fn.assert_called_once_with(
                ontology_url=str(FIXTURE_OWL),
                ontology_acronym="ACRONYM",
                cache_folder=str(tmp_path),
            )
