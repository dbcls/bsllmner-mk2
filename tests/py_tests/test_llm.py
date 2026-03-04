"""Tests for private helper functions in bsllmner2.llm."""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.errors import OllamaConnectionError
from bsllmner2.llm import (
    _construct_messages,
    _construct_output,
    _extract_last_json,
    _extract_last_json_str,
    _normalize_null_strings,
    ner,
)
from bsllmner2.models import Prompt
from tests.py_tests.conftest import FakeLlmBackend, make_chat_response

# === TestExtractLastJson ===


class TestExtractLastJson:
    def test_simple_json_object(self) -> None:
        result = _extract_last_json('text {"key": "val"} text')
        assert result == {"key": "val"}

    def test_simple_json_array(self) -> None:
        result = _extract_last_json("prefix [1, 2, 3] suffix")
        assert result == [1, 2, 3]

    def test_multiple_json_returns_last(self) -> None:
        result = _extract_last_json('{"a": 1} noise {"b": 2}')
        assert result == {"b": 2}

    def test_no_json_returns_none(self) -> None:
        assert _extract_last_json("no json here") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_last_json("") is None

    def test_invalid_json_braces(self) -> None:
        assert _extract_last_json("{not valid}") is None

    def test_json_with_newlines(self) -> None:
        result = _extract_last_json('{"k":\n"v"}')
        assert result == {"k": "v"}

    def test_nested_json_extracts_outer(self) -> None:
        """raw_decode correctly handles nested JSON objects."""
        result = _extract_last_json('{"a": {"b": 1}}')
        assert result == {"a": {"b": 1}}

    def test_unicode_preserved(self) -> None:
        result = _extract_last_json('{"name": "日本語"}')
        assert isinstance(result, dict)
        assert result["name"] == "日本語"

    def test_mixed_valid_invalid(self) -> None:
        result = _extract_last_json('{bad} {"ok": true}')
        assert result == {"ok": True}

    def test_empty_json_object(self) -> None:
        assert _extract_last_json("{}") == {}

    def test_empty_json_array(self) -> None:
        assert _extract_last_json("[]") == []

    def test_only_opening_brace(self) -> None:
        assert _extract_last_json("{") is None

    def test_llm_thinking_then_json(self) -> None:
        text = '<think>reasoning goes here...</think>\n{"cell_line": "HeLa"}'
        result = _extract_last_json(text)
        assert result == {"cell_line": "HeLa"}


# === TestExtractLastJsonStr ===


class TestExtractLastJsonStr:
    """Tests for _extract_last_json_str (returns raw substring, no re-serialization)."""

    def test_simple_object(self) -> None:
        result = _extract_last_json_str('text {"key": "val"} text')
        assert result == '{"key": "val"}'

    def test_multiple_json_returns_last(self) -> None:
        result = _extract_last_json_str('{"a": 1} noise {"b": 2}')
        assert result == '{"b": 2}'

    def test_no_json_returns_none(self) -> None:
        assert _extract_last_json_str("no json here") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_last_json_str("") is None

    def test_preserves_original_formatting(self) -> None:
        text = 'prefix {"k":  "v"} suffix'
        result = _extract_last_json_str(text)
        assert result == '{"k":  "v"}'


# === TestNormalizeNullStrings ===


class TestNormalizeNullStrings:
    """Tests for _normalize_null_strings (recursive null normalization)."""

    def test_top_level_dict(self) -> None:
        result = _normalize_null_strings({"a": "null", "b": "None", "c": "ok"})
        assert result == {"a": None, "b": None, "c": "ok"}

    def test_nested_dict(self) -> None:
        result = _normalize_null_strings({"outer": {"inner": "null"}})
        assert result == {"outer": {"inner": None}}

    def test_nested_list(self) -> None:
        result = _normalize_null_strings(["null", "None", "ok"])
        assert result == [None, None, "ok"]

    def test_deeply_nested(self) -> None:
        result = _normalize_null_strings({"a": [{"b": "null"}, "None"]})
        assert result == {"a": [{"b": None}, None]}

    def test_non_string_values_unchanged(self) -> None:
        result = _normalize_null_strings({"a": 42, "b": True, "c": None})
        assert result == {"a": 42, "b": True, "c": None}

    def test_empty_containers(self) -> None:
        assert _normalize_null_strings({}) == {}
        assert _normalize_null_strings([]) == []


# === TestConstructOutput ===


class TestConstructOutput:
    """Tests for _construct_output (extract mode output parsing)."""

    _BS_ENTRY: ClassVar[dict[str, Any]] = {"accession": "SAMN00000001", "title": "Test"}

    def test_valid_dict_json(self) -> None:
        resp = make_chat_response('{"cell_line": "HeLa"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted == {"cell_line": "HeLa"}
        assert out.accession == "SAMN00000001"

    def test_null_string_replaced_with_none(self) -> None:
        resp = make_chat_response('{"cell_line": "null"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted == {"cell_line": None}

    def test_none_string_replaced_with_none(self) -> None:
        resp = make_chat_response('{"cell_line": "None"}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted == {"cell_line": None}

    def test_nested_null_strings_replaced(self) -> None:
        resp = make_chat_response('{"outer": {"inner": "null"}, "list": ["None", "ok"]}')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted == {"outer": {"inner": None}, "list": [None, "ok"]}

    def test_array_json_preserved(self) -> None:
        resp = make_chat_response('[{"cell_line": "HeLa"}]')
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted == [{"cell_line": "HeLa"}]
        assert out.raw_output is not None

    def test_no_json_in_response(self) -> None:
        resp = make_chat_response("no json here")
        out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted is None
        assert out.raw_output is None


_SIMPLE_PROMPT = [Prompt(role="system", content="test"), Prompt(role="user", content="Extract:")]


# === TestNer: Message mutation ===


@pytest.mark.asyncio(loop_scope="function")
class TestNerMessageMutation:
    """Verify that ner() does not mutate the original prompt messages."""

    async def test_prompt_messages_not_mutated(self) -> None:
        """The original prompt list and its Message objects must not be modified."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        prompt = [Prompt(role="system", content="test"), Prompt(role="user", content="Extract:")]
        messages_before = _construct_messages(prompt)
        original_contents = [m.content for m in messages_before]

        backend = FakeLlmBackend(['{"cell_line": "HeLa"}'])
        await ner(backend, entries, prompt, None, "test-model")

        messages_after = _construct_messages(prompt)
        for orig, after in zip(original_contents, messages_after, strict=True):
            assert orig == after.content

    async def test_empty_content_prompt_still_works(self) -> None:
        """A prompt with empty content in the last message still produces output."""
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        prompt = [Prompt(role="system", content="test"), Prompt(role="user", content="")]

        backend = FakeLlmBackend(['{"cell_line": "HeLa"}'])
        outputs, _ = await ner(backend, entries, prompt, None, "test-model")
        assert len(outputs) == 1


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
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 2
        accessions = {o.accession for o in outputs}
        assert accessions == {"SAMN001", "SAMN002"}

    async def test_empty_entries(self) -> None:
        backend = FakeLlmBackend([])
        outputs, _ = await ner(backend, [], _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_entry_without_accession(self) -> None:
        entries = [{"title": "No accession"}]
        backend = FakeLlmBackend([])
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        assert outputs[0].accession == "SAMN001"

    async def test_general_exception_not_connection_error(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([RuntimeError("unexpected")])
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        assert outputs[0].extracted == [{"cell_line": "HeLa"}]
        assert outputs[0].raw_output is not None

    async def test_error_summary_logged_at_error_level(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When some entries fail, the summary log is at ERROR level."""
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
        ]
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                RuntimeError("boom"),
            ]
        )
        logger = logging.getLogger("bsllmner2")
        original_propagate = logger.propagate
        try:
            logger.propagate = True
            with caplog.at_level(logging.ERROR, logger="bsllmner2"):
                outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        finally:
            logger.propagate = original_propagate

        assert len(outputs) == 1
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR and "Completed with" in r.message]
        assert len(error_records) == 1
        assert "1 errors" in error_records[0].message


# === TestNerErrorPaths ===


@pytest.mark.asyncio(loop_scope="function")
class TestNerErrorPaths:
    """Error path tests for ner() function."""

    async def test_partial_failure_returns_successful_entries(self) -> None:
        """When some entries fail, successful entries are still returned."""
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
            {"accession": "SAMN003", "title": "Sample 3"},
        ]
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                RuntimeError("fail on SAMN002"),
                '{"cell_line": "K562"}',
            ]
        )
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        accessions = {o.accession for o in outputs}
        assert "SAMN001" in accessions
        assert "SAMN003" in accessions
        assert "SAMN002" not in accessions
        assert len(outputs) == 2

    async def test_all_entries_fail_returns_empty(self) -> None:
        """When all entries fail, an empty list is returned."""
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
        ]
        # First entry succeeds (to establish connection), then second fails
        # Actually, RuntimeError on first entry should work since it's not ConnectionError
        backend = FakeLlmBackend(
            [
                RuntimeError("fail 1"),
                RuntimeError("fail 2"),
            ]
        )
        outputs, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []


# === TestConstructOutput: mutation-killing additions ===


class TestConstructOutputMutations:
    """Mutation-killing tests for _construct_output."""

    _NON_EBI_ENTRY: ClassVar[dict[str, Any]] = {"accession": "SAMN00000001", "title": "Test"}

    def test_raw_output_is_original_substring(self) -> None:
        resp = make_chat_response('prefix {"cell_line": "HeLa"} suffix')
        out = _construct_output(self._NON_EBI_ENTRY, resp)
        assert out.raw_output == '{"cell_line": "HeLa"}'


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
        assert result == d

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
        assert out.extracted is not None
        assert isinstance(out.extracted, dict)
        for v in out.extracted.values():
            assert v is None
