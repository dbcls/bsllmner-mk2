"""Tests for Pydantic model validation in bsllmner2.models."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ollama import ChatResponse, Message
from pydantic import ValidationError

from bsllmner2.config import Config
from bsllmner2.models import (
    CliExtractArgs,
    CliSelectArgs,
    ErrorInfo,
    ErrorLog,
    Evaluation,
    LlmOutput,
    MappingValue,
    Prompt,
    Result,
    RunMetadata,
    SelectConfig,
    SelectConfigField,
    SelectResult,
    WfInput,
)

# === Helpers ===


def _make_chat_response() -> ChatResponse:
    return ChatResponse(
        model="test",
        created_at="2024-01-01T00:00:00Z",
        message=Message(role="assistant", content=""),
        done=True,
    )


def _make_minimal_extract_args(**overrides: object) -> CliExtractArgs:
    defaults: dict[str, object] = {
        "bs_entries": Path("data.json"),
        "prompt": Path("prompt.yml"),
        "batch_size": 10,
    }
    defaults.update(overrides)

    return CliExtractArgs(**defaults)


def _make_minimal_select_args(**overrides: object) -> CliSelectArgs:
    defaults: dict[str, object] = {
        "bs_entries": Path("data.json"),
        "select_config": Path("config.json"),
        "batch_size": 10,
    }
    defaults.update(overrides)

    return CliSelectArgs(**defaults)


def _make_minimal_run_metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = {
        "run_name": "test-run",
        "model": "llama3.1:70b",
        "start_time": "20240101_120000",
    }
    defaults.update(overrides)

    return RunMetadata(**defaults)


def _make_minimal_wf_input(**overrides: object) -> WfInput:
    defaults: dict[str, object] = {
        "bs_entries": [{"accession": "SAMN001"}],
        "prompt": [Prompt(role="system", content="hello")],
        "model": "llama3.1:70b",
        "config": Config(),
    }
    defaults.update(overrides)

    return WfInput(**defaults)


# === TestPrompt ===


class TestPrompt:
    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_valid_roles(self, role: str) -> None:
        p = Prompt(role=role, content="hello")
        assert p.role == role

    @pytest.mark.parametrize("role", ["moderator", "System", "SYSTEM", ""])
    def test_invalid_roles(self, role: str) -> None:
        with pytest.raises(ValidationError):
            Prompt(role=role, content="hello")

    def test_content_required(self) -> None:
        with pytest.raises(ValidationError):
            Prompt(role="system")  # type: ignore[call-arg]

    def test_content_empty_string_allowed(self) -> None:
        p = Prompt(role="user", content="")
        assert p.content == ""

    @given(role=st.text().filter(lambda s: s not in {"system", "user", "assistant"}))
    @settings(max_examples=50)
    def test_pbt_invalid_roles_rejected(self, role: str) -> None:
        with pytest.raises(ValidationError):
            Prompt(role=role, content="x")


# === TestSelectConfigField ===


class TestSelectConfigField:
    def test_default_value_type_is_string(self) -> None:
        f = SelectConfigField()
        assert f.value_type == "string"

    @pytest.mark.parametrize("vt", ["string", "array"])
    def test_valid_value_types(self, vt: str) -> None:
        f = SelectConfigField(value_type=vt)
        assert f.value_type == vt

    @pytest.mark.parametrize("vt", ["String", "ARRAY", "list", ""])
    def test_invalid_value_types(self, vt: str) -> None:
        with pytest.raises(ValidationError):
            SelectConfigField(value_type=vt)

    def test_all_optional_fields_accept_none(self) -> None:
        f = SelectConfigField(
            ontology_file=None,
            prompt_description=None,
            ontology_filter=None,
        )
        assert f.ontology_file is None
        assert f.prompt_description is None
        assert f.ontology_filter is None

    @given(vt=st.text().filter(lambda s: s not in {"string", "array"}))
    @settings(max_examples=50)
    def test_pbt_invalid_value_types_rejected(self, vt: str) -> None:
        with pytest.raises(ValidationError):
            SelectConfigField(value_type=vt)


# === TestRunMetadata ===


class TestRunMetadata:
    @pytest.mark.parametrize("status", ["running", "completed", "failed"])
    def test_valid_statuses(self, status: str) -> None:
        m = _make_minimal_run_metadata(status=status)
        assert m.status == status

    def test_default_status_is_running(self) -> None:
        m = _make_minimal_run_metadata()
        assert m.status == "running"

    @pytest.mark.parametrize("status", ["pending", "error", "Running", ""])
    def test_invalid_statuses(self, status: str) -> None:
        with pytest.raises(ValidationError):
            _make_minimal_run_metadata(status=status)

    def test_required_fields_missing(self) -> None:
        with pytest.raises(ValidationError):
            RunMetadata()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            RunMetadata(run_name="r", model="m")  # type: ignore[call-arg]

    def test_all_optional_fields_accept_none(self) -> None:
        m = _make_minimal_run_metadata(
            end_time=None,
            processing_time=None,
            matched_entries=None,
            total_entries=None,
            accuracy=None,
            completed_count=None,
            thinking=None,
            username=None,
        )
        assert m.end_time is None
        assert m.processing_time is None

    @given(status=st.text().filter(lambda s: s not in {"running", "completed", "failed"}))
    @settings(max_examples=50)
    def test_pbt_invalid_statuses_rejected(self, status: str) -> None:
        with pytest.raises(ValidationError):
            _make_minimal_run_metadata(status=status)


# === TestMappingValue ===


class TestMappingValue:
    def test_all_fields_specified(self) -> None:
        mv = MappingValue(
            experiment_type="RNA-seq",
            extraction_answer="HeLa",
            mapping_answer_id="CVCL:0030",
            mapping_answer_label="HeLa",
        )
        assert mv.experiment_type == "RNA-seq"

    def test_omitting_nullable_field_raises_validation_error(self) -> None:
        """B4: nullable fields are required — omitting them raises ValidationError."""
        with pytest.raises(ValidationError):
            MappingValue(experiment_type="RNA-seq")  # type: ignore[call-arg]

    def test_explicit_none_accepted_for_nullable_fields(self) -> None:
        mv = MappingValue(
            experiment_type="RNA-seq",
            extraction_answer=None,
            mapping_answer_id=None,
            mapping_answer_label=None,
        )
        assert mv.extraction_answer is None
        assert mv.mapping_answer_id is None
        assert mv.mapping_answer_label is None

    def test_experiment_type_required_and_not_none(self) -> None:
        with pytest.raises(ValidationError):
            MappingValue(  # type: ignore[call-arg]
                extraction_answer=None,
                mapping_answer_id=None,
                mapping_answer_label=None,
            )

    def test_experiment_type_rejects_none(self) -> None:
        with pytest.raises(ValidationError):
            MappingValue(
                experiment_type=None,
                extraction_answer=None,
                mapping_answer_id=None,
                mapping_answer_label=None,
            )

    def test_experiment_type_empty_string_allowed(self) -> None:
        mv = MappingValue(
            experiment_type="",
            extraction_answer=None,
            mapping_answer_id=None,
            mapping_answer_label=None,
        )
        assert mv.experiment_type == ""


# === TestCliExtractArgs ===


class TestCliExtractArgs:
    def test_minimal_valid_construction(self) -> None:
        args = _make_minimal_extract_args()
        assert args.bs_entries == Path("data.json")
        assert args.prompt == Path("prompt.yml")
        assert args.batch_size == 10

    def test_format_default_is_none(self) -> None:
        """B1: After fix, format defaults to None when omitted."""
        args = _make_minimal_extract_args()
        assert args.format is None

    def test_format_explicit_none_accepted(self) -> None:
        args = _make_minimal_extract_args(format=None)
        assert args.format is None

    def test_format_explicit_path_accepted(self) -> None:
        args = _make_minimal_extract_args(format=Path("schema.json"))
        assert args.format == Path("schema.json")

    def test_batch_size_zero_raises(self) -> None:
        """B2: batch_size=0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_extract_args(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        """B2: batch_size=-1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_extract_args(batch_size=-1)

    def test_batch_size_one_accepted(self) -> None:
        args = _make_minimal_extract_args(batch_size=1)
        assert args.batch_size == 1

    def test_batch_size_none_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_minimal_extract_args(batch_size=None)

    def test_default_values(self) -> None:
        args = _make_minimal_extract_args()
        assert args.model == "llama3.1:70b"
        assert args.with_metrics is False
        assert args.resume is False
        assert args.mapping is None
        assert args.thinking is None
        assert args.max_entries is None
        assert args.run_name is None

    @given(batch_size=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_pbt_non_positive_batch_size_raises(self, batch_size: int) -> None:
        """B2: Any non-positive batch_size should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_extract_args(batch_size=batch_size)


# === TestCliSelectArgs ===


class TestCliSelectArgs:
    def test_minimal_valid_construction(self) -> None:
        args = _make_minimal_select_args()
        assert args.bs_entries == Path("data.json")
        assert args.select_config == Path("config.json")

    def test_batch_size_zero_raises(self) -> None:
        """B3: batch_size=0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_select_args(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        """B3: batch_size=-1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_select_args(batch_size=-1)

    def test_default_include_reasoning_is_true(self) -> None:
        args = _make_minimal_select_args()
        assert args.include_reasoning is True

    def test_select_config_required(self) -> None:
        with pytest.raises(ValidationError):
            CliSelectArgs(bs_entries=Path("data.json"), batch_size=10)  # type: ignore[call-arg]

    @given(batch_size=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_pbt_non_positive_batch_size_raises(self, batch_size: int) -> None:
        """B3: Any non-positive batch_size should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_select_args(batch_size=batch_size)


# === TestLlmOutput ===


class TestLlmOutput:
    def test_minimal_construction(self) -> None:
        o = LlmOutput(accession="SAMN001", chat_response=_make_chat_response())
        assert o.accession == "SAMN001"
        assert o.output is None

    def test_chat_response_required(self) -> None:
        with pytest.raises(ValidationError):
            LlmOutput(accession="SAMN001")  # type: ignore[call-arg]

    def test_output_default_is_none(self) -> None:
        o = LlmOutput(accession="SAMN001", chat_response=_make_chat_response())
        assert o.output is None

    def test_output_accepts_dict_list_none(self) -> None:
        cr = _make_chat_response()
        assert LlmOutput(accession="a", output={"k": "v"}, chat_response=cr).output == {"k": "v"}
        assert LlmOutput(accession="a", output=[1, 2], chat_response=cr).output == [1, 2]
        assert LlmOutput(accession="a", output=None, chat_response=cr).output is None

    def test_output_rejects_scalar(self) -> None:
        cr = _make_chat_response()
        with pytest.raises(ValidationError):
            LlmOutput(accession="a", output="str", chat_response=cr)
        with pytest.raises(ValidationError):
            LlmOutput(accession="a", output=42, chat_response=cr)

    def test_accession_required(self) -> None:
        with pytest.raises(ValidationError):
            LlmOutput(chat_response=_make_chat_response())  # type: ignore[call-arg]


# === TestSelectResult ===


class TestSelectResult:
    def test_dict_fields_default_empty(self) -> None:
        sr = SelectResult(accession="SAMN001")
        assert sr.search_results == {}
        assert sr.text2term_results == {}
        assert sr.llm_chat_response == {}
        assert sr.results == {}

    def test_two_instances_dicts_independent(self) -> None:
        sr1 = SelectResult(accession="A")
        sr2 = SelectResult(accession="B")
        sr1.search_results["field"] = {}
        assert sr2.search_results == {}

    def test_accession_required(self) -> None:
        with pytest.raises(ValidationError):
            SelectResult()  # type: ignore[call-arg]

    def test_extract_output_default_is_none(self) -> None:
        sr = SelectResult(accession="SAMN001")
        assert sr.extract_output is None

    def test_extract_output_accepts_dict_list_none(self) -> None:
        sr_dict = SelectResult(accession="SAMN001", extract_output={"cell_line": "HeLa"})
        assert sr_dict.extract_output == {"cell_line": "HeLa"}
        sr_list = SelectResult(accession="SAMN002", extract_output=["a", "b"])
        assert sr_list.extract_output == ["a", "b"]
        sr_none = SelectResult(accession="SAMN003", extract_output=None)
        assert sr_none.extract_output is None


# === TestEvaluation ===


class TestEvaluation:
    def test_default_match_is_false(self) -> None:
        e = Evaluation(accession="SAMN001")
        assert e.match is False

    def test_match_true_with_mismatched_values_accepted(self) -> None:
        """B5: No cross-field validation — match=True is accepted.

        even when expected != actual. This is by design: the Evaluation
        model is a data container; consistency is the caller's responsibility.
        """
        e = Evaluation(accession="SAMN001", expected="HeLa", actual="HEK293", match=True)
        assert e.match is True
        assert e.expected != e.actual

    def test_defaults_for_optional_fields(self) -> None:
        e = Evaluation(accession="SAMN001")
        assert e.expected is None
        assert e.actual is None

    @given(expected=st.text(), actual=st.text())
    @settings(max_examples=50)
    def test_pbt_match_true_always_accepted(self, expected: str, actual: str) -> None:
        """B5 (record): match=True is always accepted regardless of expected/actual."""
        e = Evaluation(accession="X", expected=expected, actual=actual, match=True)
        assert e.match is True


# === TestWfInput ===


class TestWfInput:
    def test_minimal_construction(self) -> None:
        wf = _make_minimal_wf_input()
        assert wf.model == "llama3.1:70b"
        assert len(wf.prompt) == 1

    def test_empty_prompt_raises(self) -> None:
        """B6: After fix, empty prompt list should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_wf_input(prompt=[])

    def test_empty_model_raises(self) -> None:
        """B7: After fix, empty model string should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal_wf_input(model="")

    def test_cli_args_none_accepted(self) -> None:
        wf = _make_minimal_wf_input(cli_args=None)
        assert wf.cli_args is None

    def test_mapping_none_accepted(self) -> None:
        wf = _make_minimal_wf_input(mapping=None)
        assert wf.mapping is None

    def test_format_none_accepted(self) -> None:
        wf = _make_minimal_wf_input(format=None)
        assert wf.format is None


# === TestSelectConfig ===


class TestSelectConfig:
    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            SelectConfig()  # type: ignore[call-arg]

    def test_empty_fields_dict_accepted(self) -> None:
        sc = SelectConfig(fields={})
        assert sc.fields == {}

    def test_valid_fields(self) -> None:
        sc = SelectConfig(
            fields={
                "cell_line": SelectConfigField(value_type="string"),
                "diseases": SelectConfigField(value_type="array"),
            },
        )
        assert "cell_line" in sc.fields
        assert sc.fields["diseases"].value_type == "array"


# === TestErrorInfo, TestErrorLog, TestResult ===


class TestErrorInfo:
    def test_all_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            ErrorInfo()  # type: ignore[call-arg]

    def test_valid_construction(self) -> None:
        ei = ErrorInfo(type="ValueError", message="bad value", traceback="Traceback ...")
        assert ei.type == "ValueError"
        assert ei.message == "bad value"


class TestErrorLog:
    def test_valid_construction(self) -> None:
        el = ErrorLog(
            timestamp="2024-01-01T00:00:00Z",
            error=ErrorInfo(type="ValueError", message="bad value", traceback="tb"),
        )
        assert el.timestamp == "2024-01-01T00:00:00Z"

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ErrorLog()  # type: ignore[call-arg]


class TestResult:
    def test_output_evaluation_default_empty_list(self) -> None:
        r = Result(
            input=_make_minimal_wf_input(),
            run_metadata=_make_minimal_run_metadata(),
        )
        assert r.output == []
        assert r.evaluation == []

    def test_two_instances_lists_independent(self) -> None:
        r1 = Result(input=_make_minimal_wf_input(), run_metadata=_make_minimal_run_metadata())
        r2 = Result(input=_make_minimal_wf_input(), run_metadata=_make_minimal_run_metadata())
        r1.output.append(LlmOutput(accession="A", chat_response=_make_chat_response()))
        assert r2.output == []

    def test_metrics_default_is_none(self) -> None:
        r = Result(input=_make_minimal_wf_input(), run_metadata=_make_minimal_run_metadata())
        assert r.metrics is None
