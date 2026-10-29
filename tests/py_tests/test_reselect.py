"""Tests for bsllmner2.reselect: target values, restricted inputs, merging and the batch pass."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.models import (
    ExtractEntry,
    LlmTimingFields,
    OntologyIndex,
    ResolvedValue,
    SearchResult,
    SelectConfig,
    SelectConfigField,
    SelectEntry,
)
from bsllmner2.reselect import (
    extract_entry_for_reselect,
    fields_with_ontology,
    merge_reselected,
    reselect_entries,
    restrict_select_config,
    sha256_of_file,
    unresolved_values,
)
from tests.py_tests.conftest import FakeLlmBackend

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _config(**fields: Path | None) -> SelectConfig:
    return SelectConfig(
        fields={name: SelectConfigField(ontology_file=owl, value_type="array") for name, owl in fields.items()},
    )


def _hit(term_id: str, value: str, exact: bool, score: float | None = None) -> SearchResult:
    return SearchResult(
        term_uri=f"http://example.org/{term_id}",
        term_id=term_id,
        prop_uri=RDFS_LABEL,
        value=value,
        label=value,
        exact_match=exact,
        text2term_score=score,
    )


def _resolved(value: str, term_id: str | None) -> ResolvedValue:
    return ResolvedValue(value=value, term_id=term_id, label=value)


def _entry(
    accession: str,
    extracted: dict[str, Any] | None,
    results: dict[str, list[ResolvedValue]] | None = None,
) -> SelectEntry:
    return SelectEntry(extract=ExtractEntry(accession=accession, extracted=extracted), results=results or {})


# === fields_with_ontology / restrict_select_config ===


class TestFieldsWithOntology:
    def test_default_is_every_field_with_an_ontology_in_config_order(self, tmp_path: Path) -> None:
        config = _config(tissue=tmp_path / "u.owl", note=None, knockout_gene=tmp_path / "g.owl")
        assert fields_with_ontology(config, None) == ["tissue", "knockout_gene"]

    def test_requested_subset_keeps_config_order_and_drops_duplicates(self, tmp_path: Path) -> None:
        config = _config(tissue=tmp_path / "u.owl", knockout_gene=tmp_path / "g.owl", drug=tmp_path / "c.owl")
        assert fields_with_ontology(config, ["drug", "tissue", "drug"]) == ["tissue", "drug"]

    def test_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        config = _config(tissue=tmp_path / "u.owl")
        with pytest.raises(ValueError, match="not in the select config"):
            fields_with_ontology(config, ["tissue", "organ"])

    def test_field_without_ontology_is_rejected(self, tmp_path: Path) -> None:
        config = _config(tissue=tmp_path / "u.owl", note=None)
        with pytest.raises(ValueError, match="without an ontology_file"):
            fields_with_ontology(config, ["note"])

    def test_restrict_keeps_only_named_fields_in_order(self, tmp_path: Path) -> None:
        config = _config(tissue=tmp_path / "u.owl", knockout_gene=tmp_path / "g.owl", drug=tmp_path / "c.owl")
        restricted = restrict_select_config(config, ["drug", "tissue"])
        assert list(restricted.fields) == ["tissue", "drug"]
        assert restricted.fields["drug"] == config.fields["drug"]


# === unresolved_values ===


class TestUnresolvedValues:
    def test_values_without_a_term_are_targets_in_extracted_order(self) -> None:
        entry = _entry(
            "S1",
            {"knockout_gene": ["c-Myc", "TP53", "Dicer"]},
            {"knockout_gene": [_resolved("TP53", "NCBIGene:7157")]},
        )
        assert unresolved_values(entry, ["knockout_gene"]) == {"knockout_gene": ["c-Myc", "Dicer"]}

    def test_string_field_is_a_single_target(self) -> None:
        entry = _entry("S1", {"tissue": "limb muscle"})
        assert unresolved_values(entry, ["tissue"]) == {"tissue": ["limb muscle"]}

    def test_resolved_value_without_term_id_still_counts_as_resolved(self) -> None:
        # Fields without an ontology record the value as-is; the value is in results, so it is not a target.
        entry = _entry("S1", {"note": ["x"]}, {"note": [_resolved("x", None)]})
        assert unresolved_values(entry, ["note"]) == {}

    def test_fields_missing_none_or_non_string_values_yield_nothing(self) -> None:
        entry = _entry("S1", {"tissue": None, "drug": [1, None], "knockout_gene": []})
        assert unresolved_values(entry, ["tissue", "drug", "knockout_gene", "cell_line"]) == {}

    def test_entry_without_extracted_yields_nothing(self) -> None:
        assert unresolved_values(_entry("S1", None), ["tissue"]) == {}

    def test_duplicate_values_appear_once(self) -> None:
        entry = _entry("S1", {"knockout_gene": ["Myc", "Myc", "Rb"]})
        assert unresolved_values(entry, ["knockout_gene"]) == {"knockout_gene": ["Myc", "Rb"]}

    def test_only_requested_fields_are_considered(self) -> None:
        entry = _entry("S1", {"knockout_gene": ["Myc"], "tissue": ["liver"]})
        assert unresolved_values(entry, ["tissue"]) == {"tissue": ["liver"]}


# === extract_entry_for_reselect ===


class TestExtractEntryForReselect:
    def test_keeps_only_targets_and_the_source_shape(self) -> None:
        entry = _entry("S1", {"knockout_gene": ["c-Myc", "TP53"], "tissue": "limb muscle", "drug": ["x"]})
        restricted = extract_entry_for_reselect(entry, {"knockout_gene": ["c-Myc"], "tissue": ["limb muscle"]})
        assert restricted.accession == "S1"
        assert restricted.extracted == {"knockout_gene": ["c-Myc"], "tissue": "limb muscle"}

    def test_no_targets_gives_empty_extracted(self) -> None:
        entry = _entry("S1", {"tissue": ["liver"]})
        assert extract_entry_for_reselect(entry, {}).extracted == {}


# === merge_reselected ===


def _timing(n: int) -> LlmTimingFields:
    return LlmTimingFields(total_duration=n, eval_count=n)


class TestMergeReselected:
    def _original(self) -> SelectEntry:
        return SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc", "TP53"], "tissue": ["liver"]}),
            search_results={
                "knockout_gene": {"c-Myc": [], "TP53": [_hit("NCBIGene:7157", "TP53", True)]},
                "tissue": {"liver": [_hit("UBERON:0002107", "liver", True)]},
            },
            text2term_results={"knockout_gene": {"c-Myc": []}, "tissue": {}},
            select_timings={"knockout_gene": {"c-Myc": _timing(1)}, "tissue": {}},
            results={
                "knockout_gene": [_resolved("TP53", "NCBIGene:7157")],
                "tissue": [_resolved("liver", "UBERON:0002107")],
            },
            ambiguous_fields={"knockout_gene": {"c-Myc": ["NCBIGene:1", "NCBIGene:2"]}, "tissue": {}},
        )

    def test_target_value_takes_new_candidates_results_and_records(self) -> None:
        original = self._original()
        new = SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc"]}),
            search_results={"knockout_gene": {"c-Myc": []}},
            text2term_results={"knockout_gene": {"c-Myc": [_hit("NCBIGene:4609", "MYC", False, 0.9)]}},
            select_timings={"knockout_gene": {"c-Myc": _timing(7)}},
            results={"knockout_gene": [_resolved("c-Myc", "NCBIGene:4609")]},
            ambiguous_fields={"knockout_gene": {}},
        )
        merged = merge_reselected(original, new, {"knockout_gene": ["c-Myc"]})

        assert [rv.term_id for rv in merged.results["knockout_gene"]] == ["NCBIGene:7157", "NCBIGene:4609"]
        assert merged.text2term_results["knockout_gene"]["c-Myc"][0].term_id == "NCBIGene:4609"
        assert merged.select_timings["knockout_gene"]["c-Myc"] == _timing(7)
        assert "c-Myc" not in merged.ambiguous_fields["knockout_gene"]
        # untouched: the other value of the field and the other field
        assert merged.search_results["knockout_gene"]["TP53"][0].term_id == "NCBIGene:7157"
        assert merged.results["tissue"] == original.results["tissue"]
        assert merged.extract == original.extract

    def test_original_is_not_mutated(self) -> None:
        original = self._original()
        snapshot = original.model_dump()
        new = SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc"]}),
            results={"knockout_gene": [_resolved("c-Myc", "NCBIGene:4609")]},
        )
        merge_reselected(original, new, {"knockout_gene": ["c-Myc"]})
        assert original.model_dump() == snapshot

    def test_unpicked_target_stays_unresolved_with_fresh_empty_records(self) -> None:
        original = self._original()
        new = SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc"]}),
            search_results={"knockout_gene": {"c-Myc": []}},
            text2term_results={"knockout_gene": {"c-Myc": []}},
        )
        merged = merge_reselected(original, new, {"knockout_gene": ["c-Myc"]})
        assert [rv.value for rv in merged.results["knockout_gene"]] == ["TP53"]
        assert merged.text2term_results["knockout_gene"]["c-Myc"] == []
        assert "c-Myc" not in merged.select_timings["knockout_gene"]
        assert "c-Myc" not in merged.ambiguous_fields["knockout_gene"]

    def test_new_results_for_non_target_values_are_ignored(self) -> None:
        original = self._original()
        new = SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc"]}),
            results={"knockout_gene": [_resolved("TP53", "NCBIGene:9999"), _resolved("c-Myc", "NCBIGene:4609")]},
        )
        merged = merge_reselected(original, new, {"knockout_gene": ["c-Myc"]})
        assert [rv.term_id for rv in merged.results["knockout_gene"]] == ["NCBIGene:7157", "NCBIGene:4609"]

    def test_field_absent_from_original_records_is_created(self) -> None:
        original = _entry("S1", {"knockout_gene": ["c-Myc"]})
        new = SelectEntry(
            extract=ExtractEntry(accession="S1", extracted={"knockout_gene": ["c-Myc"]}),
            text2term_results={"knockout_gene": {"c-Myc": [_hit("NCBIGene:4609", "MYC", False, 0.9)]}},
            results={"knockout_gene": [_resolved("c-Myc", "NCBIGene:4609")]},
        )
        merged = merge_reselected(original, new, {"knockout_gene": ["c-Myc"]})
        assert merged.results["knockout_gene"][0].term_id == "NCBIGene:4609"
        assert merged.search_results["knockout_gene"]["c-Myc"] == []


_value = st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=6)


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(_value, min_size=1, max_size=6, unique=True),
    resolved_mask=st.lists(st.booleans(), min_size=6, max_size=6),
    picked_mask=st.lists(st.booleans(), min_size=6, max_size=6),
)
def test_merge_keeps_resolved_values_and_adds_only_targets(
    values: list[str], resolved_mask: list[bool], picked_mask: list[bool]
) -> None:
    """Whatever the new pass returns, the merged entry resolves the original terms plus the picked targets."""
    resolved = [_resolved(v, f"T:{v}") for v, r in zip(values, resolved_mask, strict=False) if r]
    original = SelectEntry(
        extract=ExtractEntry(accession="S1", extracted={"g": values}),
        results={"g": resolved},
    )
    targets = unresolved_values(original, ["g"])
    target_values = targets.get("g", [])
    picked = [_resolved(v, f"N:{v}") for v, pk in zip(values, picked_mask, strict=False) if pk]
    new = SelectEntry(extract=ExtractEntry(accession="S1", extracted={"g": target_values}), results={"g": picked})

    merged = merge_reselected(original, new, targets)

    got = [(rv.value, rv.term_id) for rv in merged.results.get("g", [])]
    expected = [(rv.value, rv.term_id) for rv in resolved] + [
        (rv.value, rv.term_id) for rv in picked if rv.value in target_values
    ]
    assert got == expected
    # every value is resolved at most once per pass, and never both by the original and the new pass
    assert len({v for v, _ in got}) == len(got)
    assert unresolved_values(merged, ["g"]).get("g", []) == [
        v for v in target_values if v not in {rv.value for rv in picked}
    ]


# === reselect_entries (end to end with a fake backend) ===


@pytest.mark.asyncio(loop_scope="function")
class TestReselectEntries:
    async def test_targets_get_text2term_candidates_and_an_llm_pick(self, tmp_path: Path) -> None:
        owl = tmp_path / "genes.owl"
        owl.write_bytes(b"<rdf:RDF/>")
        config = _config(knockout_gene=owl)
        bs_entries: list[dict[str, Any]] = [
            {"accession": "S1", "title": "c-Myc knockout fibroblasts"},
            {"accession": "S2", "title": "unrelated"},
        ]
        original = _entry(
            "S1",
            {"knockout_gene": ["c-Myc", "TP53"], "tissue": ["skin"]},
            {"knockout_gene": [_resolved("TP53", "NCBIGene:7157")], "tissue": [_resolved("skin", "UBERON:0002097")]},
        )
        untouched = _entry("S2", {"knockout_gene": ["PTEN"]}, {"knockout_gene": [_resolved("PTEN", "NCBIGene:5728")]})
        targets = {"S1": {"knockout_gene": ["c-Myc"]}}
        text2term_hits = {"c-Myc": [_hit("NCBIGene:4609", "MYC", False, 0.91)]}
        backend = FakeLlmBackend(['{"id": "NCBIGene:4609", "reasoning": null}'])

        with (
            patch("bsllmner2.select.search_terms", return_value={}) as search_mock,
            patch("bsllmner2.select.search_terms_with_text2term", return_value=text2term_hits),
        ):
            merged, responses, timings, errors = await reselect_entries(
                backend,
                bs_entries,
                "test-model",
                [original, untouched],
                targets,
                config,
                include_reasoning=False,
                index_map={owl: OntologyIndex()},
            )

        assert errors == []
        assert len(responses) == 1
        assert set(timings) == {"ontology_search_sec", "text2term_sec", "llm_select_sec"}
        # only the target value was searched
        assert search_mock.call_args.args[1] == {"c-Myc"}
        s1 = next(e for e in merged if e.extract.accession == "S1")
        assert [(rv.value, rv.term_id) for rv in s1.results["knockout_gene"]] == [
            ("TP53", "NCBIGene:7157"),
            ("c-Myc", "NCBIGene:4609"),
        ]
        assert s1.text2term_results["knockout_gene"]["c-Myc"][0].text2term_score == 0.91
        assert s1.results["tissue"] == original.results["tissue"]
        s2 = next(e for e in merged if e.extract.accession == "S2")
        assert s2 == untouched

    async def test_entry_missing_from_bs_entries_is_left_unresolved(self, tmp_path: Path) -> None:
        owl = tmp_path / "genes.owl"
        owl.write_bytes(b"<rdf:RDF/>")
        config = _config(knockout_gene=owl)
        original = _entry("S1", {"knockout_gene": ["c-Myc"]})
        backend = FakeLlmBackend([])
        with (
            patch("bsllmner2.select.search_terms", return_value={}),
            patch(
                "bsllmner2.select.search_terms_with_text2term",
                return_value={"c-Myc": [_hit("NCBIGene:4609", "MYC", False, 0.9)]},
            ),
        ):
            merged, responses, _timings, errors = await reselect_entries(
                backend,
                [],
                "test-model",
                [original],
                {"S1": {"knockout_gene": ["c-Myc"]}},
                config,
                index_map={owl: OntologyIndex()},
            )
        assert responses == []
        assert errors == []
        assert merged[0].results.get("knockout_gene", []) == []
        assert merged[0].text2term_results["knockout_gene"]["c-Myc"][0].term_id == "NCBIGene:4609"


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    f = tmp_path / "r.json"
    f.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert sha256_of_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()
