"""Tests for private helper functions in bsllmner2.llm."""

import json
import re
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.errors import OllamaConnectionError
from bsllmner2.llm import (
    _build_select_schema,
    _collect_candidates_for_field,
    _compute_filter_hash,
    _construct_output,
    _extract_last_json,
    _parse_output_object,
    _pick_exact_match_search_result,
    _pick_search_result_by_id,
    ner,
    select,
)
from bsllmner2.models import LlmOutput, Prompt, SelectConfig, SelectConfigField, SelectResult
from bsllmner2.ontology_search import SearchResult
from tests.py_tests.conftest import make_chat_response

# === helpers ===


def _make_search_result(
    term_id: str,
    prop_uri: str | None,
    exact_match: bool,
    value: str = "dummy",
) -> SearchResult:
    return SearchResult(
        term_uri=f"http://example.org/{term_id}",
        term_id=term_id,
        prop_uri=prop_uri,
        value=value,
        exact_match=exact_match,
    )


RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_PREFLABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
HAS_EXACT_SYN = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"


# === TestExtractLastJson ===


class TestExtractLastJson:
    def test_simple_json_object(self) -> None:
        result = _extract_last_json('text {"key": "val"} text')
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"key": "val"}

    def test_simple_json_array(self) -> None:
        result = _extract_last_json("prefix [1, 2, 3] suffix")
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == [1, 2, 3]

    def test_multiple_json_returns_last(self) -> None:
        result = _extract_last_json('{"a": 1} noise {"b": 2}')
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"b": 2}

    def test_no_json_returns_none(self) -> None:
        assert _extract_last_json("no json here") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_last_json("") is None

    def test_invalid_json_braces(self) -> None:
        assert _extract_last_json("{not valid}") is None

    def test_json_with_newlines(self) -> None:
        result = _extract_last_json('{"k":\n"v"}')
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"k": "v"}

    def test_nested_json_extracts_outer(self) -> None:
        """raw_decode correctly handles nested JSON objects."""
        result = _extract_last_json('{"a": {"b": 1}}')
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"a": {"b": 1}}

    def test_unicode_preserved(self) -> None:
        result = _extract_last_json('{"name": "日本語"}')
        assert result is not None
        assert "日本語" in result

    def test_mixed_valid_invalid(self) -> None:
        result = _extract_last_json('{bad} {"ok": true}')
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"ok": True}

    def test_empty_json_object(self) -> None:
        result = _extract_last_json("{}")
        assert result is not None
        assert __import__("json").loads(result) == {}

    def test_empty_json_array(self) -> None:
        result = _extract_last_json("[]")
        assert result is not None
        assert __import__("json").loads(result) == []

    def test_only_opening_brace(self) -> None:
        assert _extract_last_json("{") is None

    def test_llm_thinking_then_json(self) -> None:
        text = '<think>reasoning goes here...</think>\n{"cell_line": "HeLa"}'
        result = _extract_last_json(text)
        assert result is not None
        obj = __import__("json").loads(result)
        assert obj == {"cell_line": "HeLa"}


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
        assert re.fullmatch(r"[0-9a-f]{12}", h)

    def test_different_dicts_different_hash(self) -> None:
        assert _compute_filter_hash({"a": "1"}) != _compute_filter_hash({"a": "2"})

    def test_hash_is_12_char_hex(self) -> None:
        h = _compute_filter_hash({"key": "value"})
        assert re.fullmatch(r"[0-9a-f]{12}", h)


# === TestConstructOutput ===


class TestConstructOutput:
    """Tests for _construct_output (extract mode output parsing)."""

    _BS_ENTRY: ClassVar[dict[str, Any]] = {"accession": "SAMN00000001", "title": "Test"}

    def test_valid_dict_json(self) -> None:
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.output == {"cell_line": "HeLa"}
        assert out.accession == "SAMN00000001"

    def test_null_string_replaced_with_none(self) -> None:
        resp = make_chat_response('{"cell_line": "null"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.output == {"cell_line": None}

    def test_none_string_replaced_with_none(self) -> None:
        resp = make_chat_response('{"cell_line": "None"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.output == {"cell_line": None}

    def test_array_json_preserved(self) -> None:
        """Fixed BUG 1: array JSON is now preserved as-is (not discarded)."""
        resp = make_chat_response('[{"cell_line": "HeLa"}]')
        out = _construct_output(self._BS_ENTRY, resp)
        # After fix: array JSON is preserved in output
        assert out.output == [{"cell_line": "HeLa"}]
        assert out.output_full is not None

    def test_no_json_in_response(self) -> None:
        resp = make_chat_response("no json here")
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.output is None
        assert out.output_full is None

    def test_ebi_format_adds_characteristics(self) -> None:
        ebi_entry = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(ebi_entry, resp)
        assert out.characteristics == {"cell_line": {"text": "HeLa"}}

    def test_ebi_format_list_value(self) -> None:
        ebi_entry = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"tags": ["A", "B"]}')
        out = _construct_output(ebi_entry, resp)
        assert out.characteristics == {"tags": [{"text": "A"}, {"text": "B"}]}


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
        """BUG 1: array JSON triggers AttributeError on .items(), caught and returns None."""
        resp = make_chat_response('[{"id": "CL:0000001"}]')
        # Before fix: returns None because .items() fails on list
        assert _parse_output_object(resp) is None

    def test_no_json(self) -> None:
        resp = make_chat_response("just text, no json")
        assert _parse_output_object(resp) is None

    def test_integer_value_not_replaced(self) -> None:
        resp = make_chat_response('{"count": 42}')
        obj = _parse_output_object(resp)
        assert obj is not None
        assert obj["count"] == 42


# === FakeLlmBackend ===


class FakeLlmBackend:
    """In-memory LlmBackend for testing. Returns pre-configured responses."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.host = "http://fake:11434"

    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool | None = None,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse:
        idx = self._call_index
        self._call_index += 1
        if idx >= len(self._responses):
            raise RuntimeError(f"FakeLlmBackend: no response configured for call {idx}")
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return make_chat_response(item)

    async def ensure_model(self, model: str) -> None:
        pass

    def list_models(self) -> list[str]:
        return ["test-model"]


_SIMPLE_PROMPT = [Prompt(role="system", content="test"), Prompt(role="user", content="Extract:")]


# === TestNer ===


@pytest.mark.asyncio(loop_scope="function")
class TestNer:
    """Tests for the ner() async function using FakeLlmBackend."""

    async def test_successful_extraction(self) -> None:
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
        ]
        backend = FakeLlmBackend([
            '{"cell_line": "HeLa"}',
            '{"cell_line": "HEK293"}',
        ])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 2
        accessions = {o.accession for o in outputs}
        assert accessions == {"SAMN001", "SAMN002"}

    async def test_empty_entries(self) -> None:
        backend = FakeLlmBackend([])
        outputs = await ner(backend, [], _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_entry_without_accession(self) -> None:
        entries = [{"title": "No accession"}]
        backend = FakeLlmBackend([])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_connection_error_first_entry(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([ConnectionError("refused")])
        with pytest.raises(OllamaConnectionError):
            await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")

    async def test_connection_error_after_success(self) -> None:
        """After a successful call, ConnectionError is logged but not raised."""
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
        ]
        backend = FakeLlmBackend([
            '{"cell_line": "HeLa"}',
            ConnectionError("connection lost"),
        ])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        # Only the first entry succeeds
        assert len(outputs) == 1
        assert outputs[0].accession == "SAMN001"

    async def test_general_exception_not_connection_error(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([RuntimeError("unexpected")])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_progress_file_tracking(self, tmp_path: Path) -> None:
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
        ]
        backend = FakeLlmBackend([
            '{"cell_line": "HeLa"}',
            '{"cell_line": "HEK293"}',
        ])
        progress_file = tmp_path / "progress.txt"
        await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model", progress_file_path=progress_file)
        content = progress_file.read_text()
        lines = content.strip().split("\n")
        assert set(lines) == {"SAMN001", "SAMN002"}

    async def test_progress_file_closed_on_error(self, tmp_path: Path) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([ConnectionError("fail")])
        progress_file = tmp_path / "progress.txt"
        with pytest.raises(OllamaConnectionError):
            await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model", progress_file_path=progress_file)
        # File should exist (was opened then closed in finally block)
        assert progress_file.exists()

    async def test_array_json_output_preserved(self) -> None:
        """Fixed BUG 1: JSON array from LLM is now preserved in output."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend(['[{"cell_line": "HeLa"}]'])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        # After fix: array JSON is preserved
        assert outputs[0].output == [{"cell_line": "HeLa"}]
        assert outputs[0].output_full is not None


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

    @patch("bsllmner2.llm._ontology_search_wrapper", side_effect=lambda r, *a, **kw: r)
    @patch("bsllmner2.llm._text2term_wrapper", side_effect=lambda r, *a, **kw: r)
    async def test_no_select_fields_passthrough(self, mock_t2t: Any, mock_onto: Any) -> None:
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
        # With ontology_file=None, cell_line is passed through directly
        assert results[0].results["cell_line"] == "HeLa"

    @patch("bsllmner2.llm._ontology_search_wrapper", side_effect=lambda r, *a, **kw: r)
    @patch("bsllmner2.llm._text2term_wrapper", side_effect=lambda r, *a, **kw: r)
    async def test_extract_output_none_handled(self, mock_t2t: Any, mock_onto: Any) -> None:
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
        # cell_line not in results because output was None
        assert "cell_line" not in results[0].results

    @patch("bsllmner2.llm._ontology_search_wrapper", side_effect=lambda r, *a, **kw: r)
    @patch("bsllmner2.llm._text2term_wrapper", side_effect=lambda r, *a, **kw: r)
    async def test_alignment_with_missing_entries(self, mock_t2t: Any, mock_onto: Any) -> None:
        """BUG 2: When ner() skips entries, positional zip misaligns data.

        Scenario: 3 bs_entries but ner() only succeeds for entries 1 and 3.
        extract_outputs has 2 items (for accessions SAMN001 and SAMN003).
        The positional zip pairs SAMN001's data with SAMN001 (correct)
        but SAMN003's data with SAMN002 (wrong!).
        """
        bs_entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
            {"accession": "SAMN003", "title": "Sample 3"},
        ]
        # ner() returned 2 outputs: for SAMN001 and SAMN003 (SAMN002 was skipped)
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

        # Before fix (BUG 2): positional zip pairs:
        #   SAMN001 (bs_entry) with SAMN001 extract_output → correct
        #   SAMN002 (bs_entry) with SAMN003 extract_output → WRONG!
        # SAMN003's data ("K562") is paired with SAMN002's metadata.
        # After fix: accession-based lookup ensures correct pairing.

        # SAMN003 should have cell_line = "K562"
        assert "SAMN003" in result_map
        assert result_map["SAMN003"].results["cell_line"] == "K562"

        # SAMN001 should have cell_line = "HeLa"
        assert "SAMN001" in result_map
        assert result_map["SAMN001"].results["cell_line"] == "HeLa"

    @patch("bsllmner2.llm._ontology_search_wrapper", side_effect=lambda r, *a, **kw: r)
    @patch("bsllmner2.llm._text2term_wrapper", side_effect=lambda r, *a, **kw: r)
    async def test_empty_extract_outputs(self, mock_t2t: Any, mock_onto: Any) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([])
        config = _make_select_config_no_ontology()
        results = await select(backend, entries, "test-model", [], config)
        assert results == []


# === TestBuildSelectSchema ===


class TestBuildSelectSchema:
    """Tests for _build_select_schema.

    This function was completely untested. Mutations to the schema structure,
    required fields, or reasoning conditional would go undetected.
    """

    def test_with_reasoning_includes_reasoning_field(self) -> None:
        """With reasoning=True, schema includes 'reasoning' property and required."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=True)
        assert "reasoning" in schema["properties"]
        assert "reasoning" in schema["required"]

    def test_without_reasoning_omits_reasoning_field(self) -> None:
        """With reasoning=False, schema omits 'reasoning'."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert "reasoning" not in schema["properties"]
        assert "reasoning" not in schema["required"]

    def test_enum_matches_candidate_term_ids(self) -> None:
        """Enum values in the id field match candidate term_ids in order."""
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
        """ID field allows null (for uncertain cases)."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        id_schema = schema["properties"]["id"]
        null_options = [opt for opt in id_schema["anyOf"] if opt.get("type") == "null"]
        assert len(null_options) == 1

    def test_additional_properties_false(self) -> None:
        """Schema has additionalProperties=False."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert schema["additionalProperties"] is False

    def test_id_always_required(self) -> None:
        """'id' is always in required regardless of reasoning flag."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        for reasoning in [True, False]:
            schema = _build_select_schema(candidates, reasoning=reasoning)
            assert "id" in schema["required"]

    def test_schema_type_is_object(self) -> None:
        """Top-level type is 'object'."""
        candidates = [_make_search_result("ID:001", RDFS_LABEL, exact_match=True)]
        schema = _build_select_schema(candidates, reasoning=False)
        assert schema["type"] == "object"

    def test_empty_candidates(self) -> None:
        """Empty candidates produce empty enum list."""
        schema = _build_select_schema([], reasoning=False)
        id_schema = schema["properties"]["id"]
        enum_values = None
        for option in id_schema["anyOf"]:
            if "enum" in option:
                enum_values = option["enum"]
        assert enum_values == []


# === TestCollectCandidatesForField ===


class TestCollectCandidatesForField:
    """Tests for _collect_candidates_for_field deduplication logic.

    This function was completely untested. The deduplication that prefers
    label properties would survive any mutation without these tests.
    """

    def test_merges_search_and_text2term(self) -> None:
        """Candidates from both search_results and text2term_results are merged."""
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
        """Same term_id from search and text2term is deduplicated to one entry."""
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
        """When deduplicating, label prop replaces non-label prop.

        Kills mutation on `_is_label_prop(result.prop_uri) and not _is_label_prop(prev.prop_uri)`.
        """
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
        """Non-label prop does NOT replace existing label prop."""
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
        """Empty search and text2term results return empty list."""
        select_result = SelectResult(accession="SAMN001")
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert candidates == []

    def test_missing_field_returns_empty(self) -> None:
        """Non-existent field returns empty list."""
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"other_field": {"val": [sr]}},
        )
        candidates = _collect_candidates_for_field("field", "val", select_result)
        assert candidates == []


# === TestPickSearchResultById ===


class TestPickSearchResultById:
    """Tests for _pick_search_result_by_id.

    This function was completely untested. The two-pass logic (prefer label prop,
    then fallback to any match) would survive mutations without these tests.
    """

    def test_prefers_label_prop(self) -> None:
        """When both label and non-label match term_id, picks label."""
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
        """When only non-label matches, falls back to it."""
        sr_syn = _make_search_result("ID:001", HAS_EXACT_SYN, exact_match=True, value="syn")
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr_syn]}},
        )
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:001")
        assert result is not None
        assert result.prop_uri == HAS_EXACT_SYN

    def test_not_found_returns_none(self) -> None:
        """Returns None when term_id is not in any candidates."""
        sr = _make_search_result("ID:001", RDFS_LABEL, exact_match=True)
        select_result = SelectResult(
            accession="SAMN001",
            search_results={"field": {"val": [sr]}},
        )
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:999")
        assert result is None

    def test_empty_candidates_returns_none(self) -> None:
        """Empty candidates returns None."""
        select_result = SelectResult(accession="SAMN001")
        result = _pick_search_result_by_id(select_result, "field", "val", "ID:001")
        assert result is None


# === TestConstructOutput: mutation-killing additions ===


class TestConstructOutputMutations:
    """Mutation-killing tests for _construct_output."""

    _NON_EBI_ENTRY: ClassVar[dict[str, Any]] = {"accession": "SAMN00000001", "title": "Test"}

    def test_ebi_format_copies_taxid(self) -> None:
        """EBI entry with taxId copies it to output."""
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
            "taxId": 9606,
        }
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(ebi_entry, resp)
        assert out.taxId == 9606

    def test_ebi_format_no_taxid_remains_none(self) -> None:
        """EBI entry without taxId leaves it as None."""
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(ebi_entry, resp)
        assert out.taxId is None

    def test_non_ebi_no_characteristics(self) -> None:
        """Non-EBI entry does not get characteristics even if output is dict."""
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(self._NON_EBI_ENTRY, resp)
        assert out.characteristics is None

    def test_ebi_null_value_in_characteristics(self) -> None:
        """EBI entry with null-replaced value creates {"text": None} in characteristics."""
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"cell_line": "null"}')
        out = _construct_output(ebi_entry, resp)
        assert out.characteristics == {"cell_line": {"text": None}}

    def test_output_full_is_json_string(self) -> None:
        """output_full contains the extracted JSON as a string."""
        resp = make_chat_response('prefix {"cell_line": "HeLa"} suffix')
        out = _construct_output(self._NON_EBI_ENTRY, resp)
        assert out.output_full is not None
        parsed = json.loads(out.output_full)
        assert parsed == {"cell_line": "HeLa"}
