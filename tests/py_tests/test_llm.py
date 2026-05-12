"""Tests for private helper functions in bsllmner2.llm."""

import json
import logging
from typing import Any, ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.errors import OllamaConnectionError
from bsllmner2.llm import (
    OllamaBackend,
    _construct_messages,
    _construct_output,
    _extract_last_json,
    _extract_last_json_match,
    _extract_last_json_str,
    _normalize_null_strings,
    build_ollama_options,
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

    def test_array_json_discarded_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # ExtractEntry.extracted is narrowed to ``dict | None`` so that downstream
        # code in select.py does not need defensive type-checking. List-shaped LLM
        # output is therefore discarded at the construction boundary with a WARN.
        resp = make_chat_response('[{"cell_line": "HeLa"}]')
        with caplog.at_level(logging.WARNING, logger="bsllmner2"):
            out = _construct_output(self._BS_ENTRY, resp)
        assert out.extracted is None
        # raw_output preserves the substring so the operator can audit what was dropped.
        assert out.raw_output == '[{"cell_line": "HeLa"}]'
        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Discarding list-shaped LLM output" in m and "SAMN00000001" in m for m in warning_messages)

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
        outputs, _, _, _ = await ner(backend, entries, prompt, None, "test-model")
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
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 2
        accessions = {o.accession for o in outputs}
        assert accessions == {"SAMN001", "SAMN002"}

    async def test_empty_entries(self) -> None:
        backend = FakeLlmBackend([])
        outputs, _, _, _ = await ner(backend, [], _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_entry_without_accession(self) -> None:
        entries = [{"title": "No accession"}]
        backend = FakeLlmBackend([])
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        assert outputs[0].accession == "SAMN001"

    async def test_general_exception_not_connection_error(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend([RuntimeError("unexpected")])
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert outputs == []

    async def test_array_json_output_normalised_to_none(self) -> None:
        # ``extracted`` is narrowed to ``dict | None``. A list response is logged
        # and dropped; the raw substring is preserved for post-hoc inspection.
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend(['[{"cell_line": "HeLa"}]'])
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
        assert len(outputs) == 1
        assert outputs[0].extracted is None
        assert outputs[0].raw_output == '[{"cell_line": "HeLa"}]'

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
                outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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
        outputs, _, _, _ = await ner(backend, entries, _SIMPLE_PROMPT, None, "test-model")
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


# === TestBuildOllamaOptions (Critical-5 / num_predict fix) ===


class TestBuildOllamaOptions:
    """Verify the post-Critical-5 contract for ``build_ollama_options``.

    The old implementation forced ``num_predict = num_ctx`` which made Ollama
    pre-allocate buffers for the full context window even for short responses,
    hurting throughput. The new contract leaves ``num_predict`` unset by default
    and exposes it (plus ``seed``/``temperature``) as opt-in kwargs.
    """

    def test_seed_and_temperature_defaults(self) -> None:
        opts = build_ollama_options()
        assert opts["seed"] == 0
        assert opts["temperature"] == 0.0

    def test_num_predict_not_set_by_default(self) -> None:
        opts = build_ollama_options(num_ctx=8192)
        # Absent => Ollama's own default (-1 = unlimited) applies.
        assert "num_predict" not in opts
        assert opts["num_ctx"] == 8192

    def test_num_predict_can_be_overridden(self) -> None:
        opts = build_ollama_options(num_ctx=4096, num_predict=512)
        assert opts["num_predict"] == 512

    def test_seed_temperature_overridable(self) -> None:
        opts = build_ollama_options(seed=42, temperature=0.7)
        assert opts["seed"] == 42
        assert opts["temperature"] == 0.7

    def test_num_ctx_omitted_when_none(self) -> None:
        opts = build_ollama_options(num_ctx=None)
        assert "num_ctx" not in opts


# === TestExtractLastJsonMatch (Critical-6 / dedup of the two extractors) ===


class TestExtractLastJsonMatch:
    """Shared engine used by both ``_extract_last_json`` and ``_extract_last_json_str``.

    The two public helpers used to walk the string twice; ``_extract_last_json_match``
    collapses them into a single pass so the regression we want to lock in is
    "both helpers agree on the same span".
    """

    @pytest.mark.parametrize(
        ("text", "expected_obj"),
        [
            ('{"a": 1}', {"a": 1}),
            ('garbage {"a": 1} more', {"a": 1}),
            ('{"first": 1} {"second": 2}', {"second": 2}),
            ("[1, 2, 3]", [1, 2, 3]),
        ],
    )
    def test_match_returns_parsed_obj(self, text: str, expected_obj: Any) -> None:
        match = _extract_last_json_match(text)
        assert match is not None
        obj, _start, _end = match
        assert obj == expected_obj

    def test_no_match_returns_none(self) -> None:
        assert _extract_last_json_match("no json at all") is None
        assert _extract_last_json_match("") is None

    def test_both_extractors_consistent(self) -> None:
        text = 'prefix {"a": 1} junk {"b": [2, 3]} suffix'
        as_obj = _extract_last_json(text)
        as_str = _extract_last_json_str(text)
        assert as_obj == {"b": [2, 3]}
        assert as_str == '{"b": [2, 3]}'
        # Round-trip: the substring parsed by json.loads must equal the obj.
        assert as_str is not None
        assert json.loads(as_str) == as_obj


# === TestNerCollectsErrorLogs (Critical-1 / per-entry errors_log wiring) ===


@pytest.mark.asyncio(loop_scope="function")
class TestNerCollectsErrorLogs:
    """``ner()`` now returns a 4-tuple whose last element is the per-entry :class:`ErrorLog` list.

    The wiring used to be a bare ``error_count: int`` which made the failure
    invisible in the final result JSON (Critical-1). These tests lock in the
    new contract.
    """

    async def test_partial_failures_appended_to_errors_log(self) -> None:
        entries = [
            {"accession": "SAMN001", "title": "Sample 1"},
            {"accession": "SAMN002", "title": "Sample 2"},
            {"accession": "SAMN003", "title": "Sample 3"},
        ]
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',  # SAMN001 succeeds
                RuntimeError("LLM blew up"),  # SAMN002 fails
                '{"cell_line": "HEK293"}',  # SAMN003 succeeds
            ],
        )
        outputs, _, error_count, errors_log = await ner(
            backend,
            entries,
            _SIMPLE_PROMPT,
            None,
            "test-model",
        )
        assert len(outputs) == 2
        assert error_count == 1
        assert len(errors_log) == 1
        log = errors_log[0]
        # OllamaProcessingError wraps the original exception with the accession.
        assert log.error.type == "OllamaProcessingError"
        assert "SAMN002" in log.error.message
        assert "LLM blew up" in log.error.message
        assert log.timestamp.tzinfo is not None

    async def test_all_success_yields_empty_errors_log(self) -> None:
        entries = [{"accession": "SAMN001", "title": "Sample 1"}]
        backend = FakeLlmBackend(['{"cell_line": "HeLa"}'])
        outputs, _, error_count, errors_log = await ner(
            backend,
            entries,
            _SIMPLE_PROMPT,
            None,
            "test-model",
        )
        assert len(outputs) == 1
        assert error_count == 0
        assert errors_log == []

    async def test_error_count_matches_errors_log_length(self) -> None:
        entries = [{"accession": f"SAMN00{i}", "title": f"Sample {i}"} for i in range(1, 4)]
        # First success, then two failures.
        backend = FakeLlmBackend(
            [
                '{"cell_line": "HeLa"}',
                RuntimeError("boom1"),
                RuntimeError("boom2"),
            ],
        )
        _outputs, _resp, error_count, errors_log = await ner(
            backend,
            entries,
            _SIMPLE_PROMPT,
            None,
            "test-model",
        )
        # error_count is now derived from len(errors_log); the two stay in sync.
        assert error_count == len(errors_log) == 2


# === TestOllamaBackendSemaphore (Critical-4 / env-driven concurrency) ===


class TestOllamaBackendSemaphore:
    """``OllamaBackend`` now reads ``BSLLMNER2_OLLAMA_CONCURRENCY`` lazily.

    The previous hard-coded ``256`` was undocumented and required a code edit
    to tune. The default is still ``256`` but the env var lets operators
    throttle without redeploying.
    """

    def test_default_concurrency_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BSLLMNER2_OLLAMA_CONCURRENCY", raising=False)
        backend = OllamaBackend("http://example:11434")
        assert backend.semaphore_limit == 256

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BSLLMNER2_OLLAMA_CONCURRENCY", "8")
        backend = OllamaBackend("http://example:11434")
        assert backend.semaphore_limit == 8

    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BSLLMNER2_OLLAMA_CONCURRENCY", "8")
        backend = OllamaBackend("http://example:11434", semaphore_limit=2)
        assert backend.semaphore_limit == 2

    def test_invalid_env_falls_back_to_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("BSLLMNER2_OLLAMA_CONCURRENCY", "not_a_number")
        with caplog.at_level(logging.WARNING, logger="bsllmner2"):
            backend = OllamaBackend("http://example:11434")
        assert backend.semaphore_limit == 256
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Invalid BSLLMNER2_OLLAMA_CONCURRENCY" in m for m in warnings)

    def test_negative_env_falls_back_to_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("BSLLMNER2_OLLAMA_CONCURRENCY", "0")
        with caplog.at_level(logging.WARNING, logger="bsllmner2"):
            backend = OllamaBackend("http://example:11434")
        assert backend.semaphore_limit == 256
