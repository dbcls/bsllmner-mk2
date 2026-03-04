"""Tests for Pydantic model validation in bsllmner2.models."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bsllmner2.models import (
    CliExtractArgs,
    CliSelectArgs,
    ErrorInfo,
    ErrorLog,
    EvaluationMetrics,
    ExtractEntry,
    MappingValue,
    Prompt,
    ResolvedValue,
    RunMetadata,
    SelectConfig,
    SelectConfigField,
    SelectEntry,
)

# === Helpers ===


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
        "start_time": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return RunMetadata(**defaults)


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
            processing_time_sec=None,
            total_entries=None,
            thinking=None,
        )
        assert m.end_time is None
        assert m.processing_time_sec is None

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
        assert args.resume is False
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


# === TestExtractEntry ===


class TestExtractEntry:
    def test_minimal_construction(self) -> None:
        e = ExtractEntry(accession="SAMN001")
        assert e.accession == "SAMN001"
        assert e.extracted is None
        assert e.raw_output is None

    def test_extracted_accepts_dict_list_none(self) -> None:
        assert ExtractEntry(accession="a", extracted={"k": "v"}).extracted == {"k": "v"}
        assert ExtractEntry(accession="a", extracted=[1, 2]).extracted == [1, 2]
        assert ExtractEntry(accession="a", extracted=None).extracted is None

    def test_accession_required(self) -> None:
        with pytest.raises(ValidationError):
            ExtractEntry()  # type: ignore[call-arg]

    def test_llm_timing_defaults(self) -> None:
        e = ExtractEntry(accession="SAMN001")
        assert e.llm_timing.total_duration == 0
        assert e.llm_timing.eval_count == 0


# === TestSelectEntry ===


class TestSelectEntry:
    def test_dict_fields_default_empty(self) -> None:
        extract = ExtractEntry(accession="SAMN001")
        se = SelectEntry(extract=extract)
        assert se.search_results == {}
        assert se.text2term_results == {}
        assert se.select_timings == {}
        assert se.results == {}

    def test_two_instances_dicts_independent(self) -> None:
        e1 = ExtractEntry(accession="A")
        e2 = ExtractEntry(accession="B")
        se1 = SelectEntry(extract=e1)
        se2 = SelectEntry(extract=e2)
        se1.search_results["field"] = {}
        assert se2.search_results == {}

    def test_extract_required(self) -> None:
        with pytest.raises(ValidationError):
            SelectEntry()  # type: ignore[call-arg]


# === TestResolvedValue ===


class TestResolvedValue:
    def test_minimal_construction(self) -> None:
        rv = ResolvedValue(value="HeLa")
        assert rv.value == "HeLa"
        assert rv.term_id is None

    def test_all_fields(self) -> None:
        rv = ResolvedValue(
            value="HeLa",
            term_id="CVCL:0030",
            term_uri="http://example.org/CVCL:0030",
            label="HeLa",
            exact_match=True,
            reasoning="test",
        )
        assert rv.term_id == "CVCL:0030"
        assert rv.exact_match is True

    def test_value_required(self) -> None:
        with pytest.raises(ValidationError):
            ResolvedValue()  # type: ignore[call-arg]


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


# === TestErrorInfo, TestErrorLog ===


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
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        el = ErrorLog(
            timestamp=ts,
            error=ErrorInfo(type="ValueError", message="bad value", traceback="tb"),
        )
        assert el.timestamp == ts

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ErrorLog()  # type: ignore[call-arg]


# === TestEvaluationMetrics ===


class TestEvaluationMetrics:
    def test_defaults(self) -> None:
        m = EvaluationMetrics()
        assert m.tp == 0
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 0
        assert m.correct == 0
        assert m.total == 0
        assert m.accuracy is None
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None

    def test_all_fields_specified(self) -> None:
        m = EvaluationMetrics(
            tp=10,
            fp=2,
            fn=3,
            tn=5,
            total=20,
            accuracy=0.6,
            precision=0.833,
            recall=0.769,
            f1=0.8,
        )
        assert m.tp == 10
        assert m.fp == 2
        assert m.fn == 3
        assert m.tn == 5
        assert m.correct == m.tp + m.tn
        assert m.total == 20
        assert m.accuracy == pytest.approx(0.6)
        assert m.precision == pytest.approx(0.833)
        assert m.recall == pytest.approx(0.769)
        assert m.f1 == pytest.approx(0.8)

    def test_correct_is_computed_field(self) -> None:
        """Correct is always tp + tn, computed automatically."""
        m = EvaluationMetrics(tp=7, tn=3)
        assert m.correct == 10
        dumped = m.model_dump()
        assert "correct" in dumped
        assert dumped["correct"] == 10
