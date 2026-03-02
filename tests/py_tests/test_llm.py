"""Tests for private helper functions in bsllmner2.llm."""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.errors import OllamaConnectionError
from bsllmner2.llm import _construct_output, _extract_last_json, ner
from bsllmner2.models import Prompt
from tests.py_tests.conftest import FakeLlmBackend, make_chat_response

# === TestExtractLastJson ===


class TestExtractLastJson:
    def test_simple_json_object(self) -> None:
        result = _extract_last_json('text {"key": "val"} text')
        assert result is not None
        obj = json.loads(result)
        assert obj == {"key": "val"}

    def test_simple_json_array(self) -> None:
        result = _extract_last_json("prefix [1, 2, 3] suffix")
        assert result is not None
        obj = json.loads(result)
        assert obj == [1, 2, 3]

    def test_multiple_json_returns_last(self) -> None:
        result = _extract_last_json('{"a": 1} noise {"b": 2}')
        assert result is not None
        obj = json.loads(result)
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
        obj = json.loads(result)
        assert obj == {"k": "v"}

    def test_nested_json_extracts_outer(self) -> None:
        """raw_decode correctly handles nested JSON objects."""
        result = _extract_last_json('{"a": {"b": 1}}')
        assert result is not None
        obj = json.loads(result)
        assert obj == {"a": {"b": 1}}

    def test_unicode_preserved(self) -> None:
        result = _extract_last_json('{"name": "日本語"}')
        assert result is not None
        assert "日本語" in result

    def test_mixed_valid_invalid(self) -> None:
        result = _extract_last_json('{bad} {"ok": true}')
        assert result is not None
        obj = json.loads(result)
        assert obj == {"ok": True}

    def test_empty_json_object(self) -> None:
        result = _extract_last_json("{}")
        assert result is not None
        assert json.loads(result) == {}

    def test_empty_json_array(self) -> None:
        result = _extract_last_json("[]")
        assert result is not None
        assert json.loads(result) == []

    def test_only_opening_brace(self) -> None:
        assert _extract_last_json("{") is None

    def test_llm_thinking_then_json(self) -> None:
        text = '<think>reasoning goes here...</think>\n{"cell_line": "HeLa"}'
        result = _extract_last_json(text)
        assert result is not None
        obj = json.loads(result)
        assert obj == {"cell_line": "HeLa"}


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
        resp = make_chat_response('[{"cell_line": "HeLa"}]')
        out = _construct_output(self._BS_ENTRY, resp)
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
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                '{"cell_line": "HEK293"}',
            ]
        )
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
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                ConnectionError("connection lost"),
            ]
        )
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                '{"cell_line": "HEK293"}',
            ]
        )
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
        assert progress_file.exists()

    async def test_array_json_output_preserved(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend(['[{"cell_line": "HeLa"}]'])
        outputs = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        assert outputs[0].output == [{"cell_line": "HeLa"}]
        assert outputs[0].output_full is not None


# === TestConstructOutput: mutation-killing additions ===


class TestConstructOutputMutations:
    """Mutation-killing tests for _construct_output."""

    _NON_EBI_ENTRY: ClassVar[dict[str, Any]] = {"accession": "SAMN00000001", "title": "Test"}

    def test_ebi_format_copies_taxid(self) -> None:
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
            "taxId": 9606,
        }
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(ebi_entry, resp)
        assert out.taxId == 9606

    def test_ebi_format_no_taxid_remains_none(self) -> None:
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(ebi_entry, resp)
        assert out.taxId is None

    def test_non_ebi_no_characteristics(self) -> None:
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(self._NON_EBI_ENTRY, resp)
        assert out.characteristics is None

    def test_ebi_null_value_in_characteristics(self) -> None:
        ebi_entry: dict[str, Any] = {
            "accession": "SAMEA00000001",
            "characteristics": {"organism": [{"text": "Homo sapiens"}]},
        }
        resp = make_chat_response('{"cell_line": "null"}')
        out = _construct_output(ebi_entry, resp)
        assert out.characteristics == {"cell_line": {"text": None}}

    def test_output_full_is_json_string(self) -> None:
        resp = make_chat_response('prefix {"cell_line": "HeLa"} suffix')
        out = _construct_output(self._NON_EBI_ENTRY, resp)
        assert out.output_full is not None
        parsed = json.loads(out.output_full)
        assert parsed == {"cell_line": "HeLa"}


# === Property-based tests ===


class TestExtractLastJsonPBT:
    """Property-based tests for _extract_last_json."""

    @given(
        prefix=st.text(alphabet=st.characters(blacklist_characters="{}[]"), min_size=0, max_size=50),
        d=st.fixed_dictionaries({"key": st.text(min_size=0, max_size=30)}),
        suffix=st.text(alphabet=st.characters(blacklist_characters="{}[]"), min_size=0, max_size=50),
    )
    @settings(max_examples=200)
    def test_valid_json_always_extracted(self, prefix: str, d: dict[str, str], suffix: str) -> None:
        """Any valid JSON dict embedded in arbitrary text is always extracted."""
        json_str = json.dumps(d)
        text = prefix + json_str + suffix
        result = _extract_last_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == d

    @given(text=st.text(alphabet=st.characters(blacklist_characters="{}[]")))
    @settings(max_examples=200)
    def test_no_braces_returns_none(self, text: str) -> None:
        """Text without '{', '}', '[', or ']' always returns None."""
        assert _extract_last_json(text) is None


class TestConstructOutputPBT:
    """Property-based tests for _construct_output."""

    @given(accession=st.text(min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_accession_always_preserved(self, accession: str) -> None:
        """The accession from bs_entry is always preserved in the output."""
        bs_entry: dict[str, Any] = {"accession": accession, "title": "Test"}
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(bs_entry, resp)
        assert out.accession == accession

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda s: s.isprintable()),
            values=st.sampled_from(["null", "None"]),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=200)
    def test_null_none_strings_always_replaced(self, data: dict[str, str]) -> None:
        """Dict values that are 'null' or 'None' are replaced with None."""
        bs_entry: dict[str, Any] = {"accession": "SAMN00000001", "title": "Test"}
        resp = make_chat_response(json.dumps(data))
        out = _construct_output(bs_entry, resp)
        assert out.output is not None
        assert isinstance(out.output, dict)
        for v in out.output.values():
            assert v is None
