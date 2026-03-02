"""Tests for bsllmner2.select module (ontology search, text2term, LLM-based selection)."""

import logging
import pickle
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from bsllmner2.models import (
    LlmOutput,
    OntologyIndex,
    SearchResult,
    SelectConfig,
    SelectConfigField,
    SelectResult,
)
from bsllmner2.select import (
    _build_select_schema,
    _build_select_system_message,
    _collect_candidates_for_field,
    _collect_queries,
    _compute_filter_hash,
    _distribute_results,
    _parse_output_object,
    _pick_exact_match_search_result,
    _pick_search_result_by_id,
    _serialize_candidates_for_llm,
    build_index_map,
    select,
)
from tests.py_tests.conftest import FakeLlmBackend, make_chat_response

# === helpers ===

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_PREFLABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
HAS_EXACT_SYN = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"


def _make_search_result(
    term_id: str,
    prop_uri: str | None,
    exact_match: bool,
    value: str = "dummy",
    text2term_score: float | None = None,
) -> SearchResult:
    return SearchResult(
        term_uri=f"http://example.org/{term_id}",
        term_id=term_id,
        prop_uri=prop_uri,
        value=value,
        exact_match=exact_match,
        text2term_score=text2term_score,
    )


# === TestPickExactMatchSearchResult ===


class TestPickExactMatchSearchResult:
    def test_empty_list(self) -> None:
        assert _pick_exact_match_search_result([]) is None

    def test_no_exact_matches(self) -> None:
        results = [
            _make_search_result("ID:001", RDFS_LABEL, exact_match=False),
            _make_search_result("ID:002", RDFS_LABEL, exact_match=False),
        ]
        assert _pick_exact_match_search_result(results) is None

    def test_single_exact_match(self) -> None:
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        result = _pick_exact_match_search_result([sr])
        assert result is sr

    def test_single_exact_among_non_exact(self) -> None:
        exact = _make_search_result("ID:002", RDFS_LABEL, exact_match=True)
        results = [
            _make_search_result("ID:001", RDFS_LABEL, exact_match=False),
            exact,
            _make_search_result("ID:003", RDFS_LABEL, exact_match=False),
        ]
        assert _pick_exact_match_search_result(results) is exact

    def test_multiple_different_term_ids(self) -> None:
        results = [
            _make_search_result("ID:001", RDFS_LABEL, exact_match=True),
            _make_search_result("ID:002", RDFS_LABEL, exact_match=True),
        ]
        assert _pick_exact_match_search_result(results) is None

    def test_same_term_id_prefers_label(self) -> None:
        non_label = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True)
        label = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        result = _pick_exact_match_search_result([non_label, label])
        assert result is label

    def test_same_term_id_prefers_preflabel(self) -> None:
        non_label = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True)
        preflabel = _make_search_result("ID:001", SKOS_PREFLABEL, exact_match=True)
        result = _pick_exact_match_search_result([non_label, preflabel])
        assert result is preflabel

    def test_same_term_id_no_label_fallback_first(self) -> None:
        first = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn1")
        second = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn2")
        result = _pick_exact_match_search_result([first, second])
        assert result is first

    def test_three_exact_same_id_label_at_end(self) -> None:
        """Label property is preferred regardless of position."""
        a = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="v1")
        b = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="v2")
        label = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="v3")
        result = _pick_exact_match_search_result([a, b, label])
        assert result is label


# === TestComputeFilterHash ===


class TestComputeFilterHash:
    def test_none_returns_nofilter(self) -> None:
        assert _compute_filter_hash(None) == "nofilter"

    def test_same_dict_same_hash(self) -> None:
        d = {"a": "1", "b": "2"}
        assert _compute_filter_hash(d) == _compute_filter_hash(d)

    def test_key_order_independent(self) -> None:
        d1 = {"a": "1", "b": "2"}
        d2 = {"b": "2", "a": "1"}
        assert _compute_filter_hash(d1) == _compute_filter_hash(d2)

    def test_empty_dict_not_nofilter(self) -> None:
        h = _compute_filter_hash({})
        assert h != "nofilter"
        assert re.fullmatch(r"[0-9a-f]{16}", h)

    def test_different_dicts_different_hash(self) -> None:
        assert _compute_filter_hash({"a": "1"}) != _compute_filter_hash({"a": "2"})

    def test_hash_is_16_char_hex(self) -> None:
        h = _compute_filter_hash({"key": "value"})
        assert re.fullmatch(r"[0-9a-f]{16}", h)


# === TestParseOutputObject ===


class TestParseOutputObject:
    """Tests for _parse_output_object (select mode output parsing)."""

    def test_valid_dict(self) -> None:
        resp = make_chat_response('{"id": "CL:0000001", "reasoning": "test"}')
        obj = _parse_output_object(resp)
        assert obj == {"id": "CL:0000001", "reasoning": "test"}

    def test_null_string_replaced(self) -> None:
        resp = make_chat_response('{"id": "null", "reasoning": "None"}')
        obj = _parse_output_object(resp)
        assert obj == {"id": None, "reasoning": None}

    def test_array_json_returns_none(self) -> None:
        resp = make_chat_response('[{"id": "CL:0000001"}]')
        assert _parse_output_object(resp) is None

    def test_no_json(self) -> None:
        resp = make_chat_response("just text, no json")
        assert _parse_output_object(resp) is None

    def test_integer_value_not_replaced(self) -> None:
        resp = make_chat_response('{"count": 42}')
        obj = _parse_output_object(resp)
        assert obj is not None
        assert obj["count"] == 42


# === TestBuildSelectSchema ===


class TestBuildSelectSchema:
    def test_with_reasoning_includes_reasoning_field(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=True)
        assert "reasoning" in schema["properties"]
        assert "reasoning" in schema["required"]

    def test_without_reasoning_omits_reasoning_field(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert "reasoning" not in schema["properties"]
        assert "reasoning" not in schema["required"]

    def test_enum_matches_candidate_term_ids(self) -> None:
        candidates = [
            _make_search_result("ID:001", RDFS_LABEL, exact_match=True),
            _make_search_result("ID:002", RDFS_LABEL, exact_match=True),
            _make_search_result("ID:003", HAS_EXACT_SYN, exact_match=False),
        ]
        schema = _build_select_schema(candidates, reasoning=False)
        id_schema = schema["properties"]["id"]
        enum_values = None
        for option in id_schema["anyOf"]:
            if "enum" in option:
                enum_values = option["enum"]
        assert enum_values == ["ID:001", "ID:002", "ID:003"]

    def test_id_allows_null(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        id_schema = schema["properties"]["id"]
        null_options = [opt for opt in id_schema["anyOf"] if opt.get("type") == "null"]
        assert len(null_options) == 1

    def test_additional_properties_false(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert schema["additionalProperties"] is False

    def test_id_always_required(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        for reasoning in [True, False]:
            schema = _build_select_schema(candidates, reasoning=reasoning)
            assert "id" in schema["required"]

    def test_schema_type_is_object(self) -> None:
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert schema["type"] == "object"

    def test_empty_candidates(self) -> None:
        schema = _build_select_schema([], reasoning=False)
        id_schema = schema["properties"]["id"]
        enum_values = None
        for option in id_schema["anyOf"]:
            if "enum" in option:
                enum_values = option["enum"]
        assert enum_values == []


# === TestCollectCandidatesForField ===


class TestCollectCandidatesForField:
    def test_merges_search_and_text2term(self) -> None:
        sr1 = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="label1")
        sr2 = _make_search_result("ID:002", RDFS_LABEL, exact_match=False, value="label2")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr1]}},
            text2term_results={"field": {"val": [sr2]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        term_ids = {c.term_id for c in candidates}
        assert term_ids == {"ID:001", "ID:002"}

    def test_deduplicates_same_term_id(self) -> None:
        sr1 = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn")
        sr2 = _make_search_result("ID:001", RDFS_LABEL, exact_match=False, value="label")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr1]}},
            text2term_results={"field": {"val": [sr2]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert len(candidates) == 1

    def test_prefers_label_prop_in_dedup(self) -> None:
        sr_syn = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn")
        sr_label = _make_search_result("ID:001", RDFS_LABEL, exact_match=False, value="label")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr_syn]}},
            text2term_results={"field": {"val": [sr_label]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert len(candidates) == 1
        assert candidates[0].prop_uri == RDFS_LABEL

    def test_label_not_replaced_by_non_label(self) -> None:
        sr_label = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="label")
        sr_syn = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=False, value="syn")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr_label]}},
            text2term_results={"field": {"val": [sr_syn]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert len(candidates) == 1
        assert candidates[0].prop_uri == RDFS_LABEL

    def test_empty_results_returns_empty(self) -> None:
        select_result = SelectResult(accession="SAMN001")
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert candidates == []

    def test_missing_field_returns_empty(self) -> None:
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"other_field": {"val": [sr]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert candidates == []


# === TestPickSearchResultById ===


class TestPickSearchResultById:
    def test_prefers_label_prop(self) -> None:
        sr_label = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="label")
        sr_syn = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr_syn, sr_label]}},
        )
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:001")
        assert result is not None
        assert result.prop_uri == RDFS_LABEL

    def test_fallback_to_non_label(self) -> None:
        sr_syn = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr_syn]}},
        )
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:001")
        assert result is not None
        assert result.prop_uri == HAS_EXACT_SYN

    def test_not_found_returns_none(self) -> None:
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr]}},
        )
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:999")
        assert result is None

    def test_empty_candidates_returns_none(self) -> None:
        select_result = SelectResult(accession="SAMN001")
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:001")
        assert result is None


# === TestCollectQueries (NEW) ===


class TestCollectQueries:
    """Tests for _collect_queries: gather unique query strings from SelectResults."""

    def test_collects_string_value(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            results={},
        )
        queries = _collect_queries([sr], "cell_line")
        assert queries == {"HeLa"}

    def test_collects_list_values(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"diseases": ["cancer", "diabetes"]},
            results={},
        )
        queries = _collect_queries([sr], "diseases")
        assert queries == {"cancer", "diabetes"}

    def test_skips_entries_with_existing_results(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            results={"cell_line": {"HeLa": _make_search_result("ID:001", RDFS_LABEL, exact_match=True)}},
        )
        queries = _collect_queries([sr], "cell_line")
        assert queries == set()

    def test_skips_entries_without_field(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"organism": "human"},
            results={},
        )
        queries = _collect_queries([sr], "cell_line")
        assert queries == set()

    def test_skips_non_dict_extract_output(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output=["not", "a", "dict"],
            results={},
        )
        queries = _collect_queries([sr], "cell_line")
        assert queries == set()

    def test_skips_none_extract_output(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output=None,
            results={},
        )
        queries = _collect_queries([sr], "cell_line")
        assert queries == set()

    def test_multiple_results_deduplicates(self) -> None:
        sr1 = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            results={},
        )
        sr2 = SelectResult(
            accession="SAMN002",
            extract_output={"cell_line": "HeLa"},
            results={},
        )
        queries = _collect_queries([sr1, sr2], "cell_line")
        assert queries == {"HeLa"}

    def test_list_with_non_string_elements_skipped(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"diseases": ["cancer", 42, None, "diabetes"]},
            results={},
        )
        queries = _collect_queries([sr], "diseases")
        assert queries == {"cancer", "diabetes"}

    def test_non_dict_non_none_extract_output_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-dict, non-None extract_output logs a WARNING."""
        sr = SelectResult(
            accession="SAMN001",
            extract_output=["not", "a", "dict"],
            results={},
        )
        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            with caplog.at_level(logging.WARNING, logger="bsllmner2"):
                _collect_queries([sr], "cell_line")
        finally:
            logger.propagate = original_propagate

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "SAMN001" in r.message]
        assert len(warning_records) >= 1


# === TestDistributeResults (NEW) ===


class TestDistributeResults:
    """Tests for _distribute_results: distribute search results back into SelectResult objects."""

    def test_distributes_candidates_to_search_results(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            search_results={"cell_line": {}},
            results={},
        )
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=False, value="HeLa")]
        all_results = {"HeLa": candidates}
        _distribute_results([sr], "cell_line", all_results, "search_results")
        assert sr.search_results["cell_line"]["HeLa"] == candidates

    def test_exact_match_sets_result_automatically(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            search_results={"cell_line": {}},
            results={},
        )
        exact = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="HeLa")
        all_results = {"HeLa": [exact]}
        _distribute_results([sr], "cell_line", all_results, "search_results")
        cell_line_results = sr.results["cell_line"]
        assert isinstance(cell_line_results, dict)
        assert cell_line_results["HeLa"] is exact

    def test_skips_entries_with_existing_results(self) -> None:
        existing = _make_search_result("ID:999", RDFS_LABEL, exact_match=True, value="existing")
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            search_results={"cell_line": {}},
            results={"cell_line": {"HeLa": existing}},
        )
        new_result = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="HeLa")
        _distribute_results([sr], "cell_line", {"HeLa": [new_result]}, "search_results")
        cell_line_results = sr.results["cell_line"]
        assert isinstance(cell_line_results, dict)
        assert cell_line_results["HeLa"] is existing

    def test_handles_list_values(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"diseases": ["cancer", "diabetes"]},
            search_results={"diseases": {}},
            results={},
        )
        cancer_result = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="cancer")
        all_results: dict[str, list[SearchResult]] = {
            "cancer": [cancer_result],
            "diabetes": [],
        }
        _distribute_results([sr], "diseases", all_results, "search_results")
        assert sr.search_results["diseases"]["cancer"] == [cancer_result]
        assert sr.search_results["diseases"]["diabetes"] == []

    def test_skips_non_dict_extract_output(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output=["not", "a", "dict"],
            search_results={},
            results={},
        )
        _distribute_results([sr], "cell_line", {}, "search_results")

    def test_non_dict_non_none_extract_output_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-dict, non-None extract_output logs a WARNING in _distribute_results."""
        sr = SelectResult(
            accession="SAMN001",
            extract_output=["not", "a", "dict"],
            search_results={},
            results={},
        )
        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            with caplog.at_level(logging.WARNING, logger="bsllmner2"):
                _distribute_results([sr], "cell_line", {}, "search_results")
        finally:
            logger.propagate = original_propagate

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "SAMN001" in r.message]
        assert len(warning_records) >= 1

    def test_no_exact_match_does_not_set_result(self) -> None:
        sr = SelectResult(
            accession="SAMN001",
            extract_output={"cell_line": "HeLa"},
            search_results={"cell_line": {}},
            results={},
        )
        non_exact = _make_search_result("ID:001", RDFS_LABEL, exact_match=False, value="HeLa")
        _distribute_results([sr], "cell_line", {"HeLa": [non_exact]}, "search_results")
        field_results = sr.results.get("cell_line", {})
        assert isinstance(field_results, dict)
        assert "HeLa" not in field_results


# === TestBuildSelectSystemMessage (NEW) ===


class TestBuildSelectSystemMessage:
    """Tests for _build_select_system_message."""

    def test_with_reasoning_includes_reasoning_instructions(self) -> None:
        msg = _build_select_system_message(reasoning=True)
        assert msg.role == "system"
        assert msg.content is not None
        assert "reasoning" in msg.content

    def test_without_reasoning_omits_reasoning_instructions(self) -> None:
        msg = _build_select_system_message(reasoning=False)
        assert msg.role == "system"
        assert msg.content is not None
        assert "reasoning" not in msg.content.lower()

    def test_always_contains_curator_role(self) -> None:
        for reasoning in [True, False]:
            msg = _build_select_system_message(reasoning=reasoning)
            assert msg.content is not None
            assert "curator" in msg.content.lower()

    def test_always_mentions_null(self) -> None:
        for reasoning in [True, False]:
            msg = _build_select_system_message(reasoning=reasoning)
            assert msg.content is not None
            assert "null" in msg.content


# === TestSerializeCandidatesForLlm (NEW) ===


class TestSerializeCandidatesForLlm:
    """Tests for _serialize_candidates_for_llm."""

    def test_excludes_internal_fields(self) -> None:
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="HeLa")
        sr.text2term_score = 0.95
        sr.reasoning = "test reasoning"
        serialized = _serialize_candidates_for_llm([sr])
        assert len(serialized) == 1
        assert "exact_match" not in serialized[0]
        assert "text2term_score" not in serialized[0]
        assert "reasoning" not in serialized[0]

    def test_preserves_public_fields(self) -> None:
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True, value="HeLa")
        serialized = _serialize_candidates_for_llm([sr])
        assert serialized[0]["term_id"] == "ID:001"
        assert serialized[0]["value"] == "HeLa"

    def test_empty_list(self) -> None:
        assert _serialize_candidates_for_llm([]) == []


# === TestSelect ===


def _make_select_config_no_ontology() -> SelectConfig:
    return SelectConfig(
        fields={
            "cell_line": SelectConfigField(
                ontology_file=None,
                prompt_description="Cell line name",
                value_type="string",
            ),
        },
    )


@pytest.mark.asyncio(loop_scope="function")
class TestSelect:
    """Tests for the select() async function."""

    async def test_no_select_fields_passthrough(self) -> None:
        """When ontology_file=None, extract output is passed through directly."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        extract_outputs = [
            LlmOutput(
                accession="SAMN001",
                output={"cell_line": "HeLa"},
                chat_response=make_chat_response('{"cell_line": "HeLa"}'),
            ),
        ]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        results = await select(backend, entries, "test-model", extract_outputs, config)
        assert len(results) == 1
        assert results[0].results["cell_line"] == "HeLa"

    async def test_extract_output_none_handled(self) -> None:
        """Entries with output=None are handled gracefully."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        extract_outputs = [
            LlmOutput(
                accession="SAMN001",
                output=None,
                chat_response=make_chat_response("no json"),
            ),
        ]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        results = await select(backend, entries, "test-model", extract_outputs, config)
        assert len(results) == 1
        assert "cell_line" not in results[0].results

    async def test_alignment_with_missing_entries(self) -> None:
        """When ner() skips entries, select aligns correctly via accession."""
        bs_entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
            {"accession": "SAMN003", "title": "Sample 3"},
        ]
        extract_outputs = [
            LlmOutput(
                accession="SAMN001",
                output={"cell_line": "HeLa"},
                chat_response=make_chat_response('{"cell_line": "HeLa"}'),
            ),
            LlmOutput(
                accession="SAMN003",
                output={"cell_line": "K562"},
                chat_response=make_chat_response('{"cell_line": "K562"}'),
            ),
        ]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        results = await select(backend, bs_entries, "test-model", extract_outputs, config)

        result_map = {r.accession: r for r in results}
        assert result_map["SAMN003"].results["cell_line"] == "K562"
        assert result_map["SAMN001"].results["cell_line"] == "HeLa"

    async def test_empty_extract_outputs(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        results = await select(backend, entries, "test-model", [], config)
        assert results == []

    async def test_non_dict_output_warns_in_no_select_fields(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-dict output for no_select_fields logs a WARNING."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        extract_outputs = [
            LlmOutput(
                accession="SAMN001",
                output=["not", "a", "dict"],
                chat_response=make_chat_response('["not", "a", "dict"]'),
            ),
        ]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            with caplog.at_level(logging.WARNING, logger="bsllmner2"):
                results = await select(backend, entries, "test-model", extract_outputs, config)
        finally:
            logger.propagate = original_propagate

        assert len(results) == 1
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "SAMN001" in r.message]
        assert len(warning_records) >= 1


# === TestBuildIndexMap ===

RDFS_LABEL_URI = "http://www.w3.org/2000/01/rdf-schema#label"


class TestBuildIndexMap:
    """Tests for build_index_map (ontology file loading + caching)."""

    @staticmethod
    def _write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for term_id, prop, value in rows:
                f.write(f"{term_id}\t{prop}\t{value}\n")

    def test_no_ontology_files_returns_empty(self, tmp_path: Path) -> None:
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=None,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        with patch("bsllmner2.select.INDEX_CACHE_DIR", tmp_path):
            result = build_index_map(config)
        assert result == {}

    def test_tsv_file_builds_index(self, tmp_path: Path) -> None:
        tsv = tmp_path / "cells.tsv"
        self._write_tsv(tsv, [("CL:0000001", RDFS_LABEL_URI, "HeLa")])
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            result = build_index_map(config)
        assert tsv in result
        assert isinstance(result[tsv], OntologyIndex)

    def test_cache_file_created(self, tmp_path: Path) -> None:
        tsv = tmp_path / "cells.tsv"
        self._write_tsv(tsv, [("CL:0000001", RDFS_LABEL_URI, "HeLa")])
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            build_index_map(config)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_cache_reused_on_second_call(self, tmp_path: Path) -> None:
        tsv = tmp_path / "cells.tsv"
        self._write_tsv(tsv, [("CL:0000001", RDFS_LABEL_URI, "HeLa")])
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            result1 = build_index_map(config)
        tsv.unlink()
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            result2 = build_index_map(config)
        assert tsv in result2
        pkl_files = list(cache_dir.glob("*.pkl"))
        with pkl_files[0].open("rb") as f:
            cached = pickle.load(f)
        assert isinstance(cached, OntologyIndex)
        assert result1[tsv].term_id_to_labels == result2[tsv].term_id_to_labels

    def test_duplicate_path_built_once(self, tmp_path: Path) -> None:
        tsv = tmp_path / "cells.tsv"
        self._write_tsv(tsv, [("CL:0000001", RDFS_LABEL_URI, "HeLa")])
        config = SelectConfig(
            fields={
                "field_a": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Field A",
                    value_type="string",
                ),
                "field_b": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Field B",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            result = build_index_map(config)
        assert len(result) == 1
        assert tsv in result

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "data.xml"
        bad_file.write_text("<xml/>")
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=bad_file,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir), pytest.raises(ValueError, match="Unsupported"):
            build_index_map(config)

    def test_corrupted_cache_falls_back_to_rebuild(self, tmp_path: Path) -> None:
        """When cache file is corrupted, it rebuilds from source."""
        tsv = tmp_path / "cells.tsv"
        self._write_tsv(tsv, [("CL:0000001", RDFS_LABEL_URI, "HeLa")])
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    ontology_file=tsv,
                    prompt_description="Cell line name",
                    value_type="string",
                ),
            },
        )
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        corrupted_pkl = cache_dir / "cells.tsv_nofilter.pkl"
        corrupted_pkl.write_bytes(b"not a valid pickle")

        with patch("bsllmner2.select.INDEX_CACHE_DIR", cache_dir):
            result = build_index_map(config)
        assert tsv in result
        assert isinstance(result[tsv], OntologyIndex)
