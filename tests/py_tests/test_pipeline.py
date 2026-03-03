"""Tests for pipeline module (evaluation + builders + time utils)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bsllmner2.config import Config
from bsllmner2.io import load_select_config
from bsllmner2.models import (
    EvaluationMetrics,
    MappingValue,
    Prompt,
    Result,
    RunMetadata,
    SearchResult,
    SelectConfig,
    SelectConfigField,
    SelectResult,
)
from bsllmner2.pipeline import (
    build_error_log,
    build_extract_prompt_for_select,
    build_extract_schema_for_select,
    compute_classification_metrics,
    compute_processing_time,
    evaluate_select_output,
    extract_predicted_term_id,
    get_now_str,
    populate_run_metadata,
    to_result,
)
from tests.py_tests.conftest import make_llm_output


class TestGetNowStr:
    """Test cases for get_now_str function."""

    def test_format(self) -> None:
        """Test that the output format is correct."""
        result = get_now_str()
        # Should match YYYYMMDD_HHMMSS format
        datetime.strptime(result, "%Y%m%d_%H%M%S")

    def test_returns_current_utc_time(self) -> None:
        """Test that get_now_str returns UTC time close to now."""
        before = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        result = get_now_str()
        after = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Result should be between before and after (or equal)
        assert before <= result <= after

    def test_uses_utc_time(self) -> None:
        """Verify that get_now_str uses UTC time for consistency."""
        utc_now = datetime.now(timezone.utc)
        result = get_now_str()
        result_dt = datetime.strptime(result, "%Y%m%d_%H%M%S")

        # The result should be within 1 second of UTC time
        utc_diff = abs((result_dt - utc_now.replace(microsecond=0, tzinfo=None)).total_seconds())
        assert utc_diff <= 1  # Should be within 1 second of UTC time


class TestBuildExtractSchemaForSelect:
    """Test cases for build_extract_schema_for_select function."""

    def test_string_field(self, select_config_file: Path) -> None:
        """Test schema generation for string field."""
        config = load_select_config(select_config_file)
        schema = build_extract_schema_for_select(config)

        assert schema["type"] == "object"
        assert "cell_line" in schema["properties"]
        assert schema["properties"]["cell_line"]["type"] == ["string", "null"]

    def test_array_field(self, temp_dir: Path) -> None:
        """Test schema generation for array field."""
        config_data = {
            "fields": {
                "diseases": {
                    "value_type": "array",
                    "prompt_description": "List of diseases",
                },
            },
        }
        config_file = temp_dir / "array_config.json"
        with config_file.open("w") as f:
            json.dump(config_data, f)

        config = load_select_config(config_file)
        schema = build_extract_schema_for_select(config)

        assert "diseases" in schema["properties"]
        assert schema["properties"]["diseases"]["type"] == ["array", "null"]
        assert schema["properties"]["diseases"]["items"]["type"] == "string"


class TestBuildExtractPromptForSelect:
    """Test cases for build_extract_prompt_for_select function."""

    def test_prompt_generation(self, select_config_file: Path) -> None:
        """Test prompt generation for select mode."""
        config = load_select_config(select_config_file)
        prompts = build_extract_prompt_for_select(config)

        assert len(prompts) == 2
        assert prompts[0].role == "system"
        assert prompts[1].role == "user"
        assert "cell_line" in prompts[1].content


# === compute_processing_time ===


class TestComputeProcessingTime:
    """Tests for compute_processing_time."""

    def test_normal(self) -> None:
        """Normal case: end > start → positive seconds."""
        result = compute_processing_time("20240101_120000", "20240101_120130")
        assert result == 90.0

    def test_same_time(self) -> None:
        """Same start and end → 0.0."""
        result = compute_processing_time("20240101_120000", "20240101_120000")
        assert result == 0.0

    def test_one_hour_difference(self) -> None:
        """One hour difference → 3600.0."""
        result = compute_processing_time("20240101_120000", "20240101_130000")
        assert result == 3600.0

    def test_cross_day_boundary(self) -> None:
        """Crossing day boundary works correctly."""
        result = compute_processing_time("20240101_235900", "20240102_000100")
        assert result == 120.0

    def test_end_before_start_raises_value_error(self) -> None:
        """end_time < start_time raises ValueError."""
        with pytest.raises(ValueError, match="before start_time"):
            compute_processing_time("20240101_130000", "20240101_120000")

    def test_invalid_format_raises_value_error(self) -> None:
        """Invalid format string raises ValueError from strptime."""
        with pytest.raises(ValueError):
            compute_processing_time("not-a-date", "20240101_120000")

    def test_invalid_end_format_raises_value_error(self) -> None:
        """Invalid end_time format raises ValueError."""
        with pytest.raises(ValueError):
            compute_processing_time("20240101_120000", "invalid")

    def test_empty_strings_raise_value_error(self) -> None:
        """Empty strings raise ValueError."""
        with pytest.raises(ValueError):
            compute_processing_time("", "")


# === build_extract_schema_for_select (additional) ===


class TestBuildExtractSchemaForSelectAdditional:
    """Additional tests for build_extract_schema_for_select."""

    def test_unsupported_value_type_raises(self) -> None:
        """Unsupported value_type raises ValueError."""
        # Bypass Pydantic Literal validation by constructing field then patching
        field = SelectConfigField(prompt_description="Test")
        field.value_type = "integer"  # type: ignore[assignment]  # force invalid
        config = SelectConfig(fields={"count": field})
        with pytest.raises(ValueError, match="Unsupported value_type"):
            build_extract_schema_for_select(config)

    def test_mixed_string_and_array_fields(self) -> None:
        """Mix of string and array fields produces correct schema types."""
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(value_type="string"),
                "diseases": SelectConfigField(value_type="array"),
            },
        )
        schema = build_extract_schema_for_select(config)
        assert schema["properties"]["cell_line"]["type"] == ["string", "null"]
        assert schema["properties"]["diseases"]["type"] == ["array", "null"]
        assert schema["properties"]["diseases"]["items"]["type"] == "string"

    def test_required_matches_fields(self) -> None:
        """Required list matches field names."""
        config = SelectConfig(
            fields={
                "a": SelectConfigField(),
                "b": SelectConfigField(),
            },
        )
        schema = build_extract_schema_for_select(config)
        assert set(schema["required"]) == {"a", "b"}

    def test_additional_properties_false(self) -> None:
        """Field additionalProperties is always False."""
        config = SelectConfig(fields={"f": SelectConfigField()})
        schema = build_extract_schema_for_select(config)
        assert schema["additionalProperties"] is False


# === build_extract_prompt_for_select: mutation-killing ===


class TestBuildExtractPromptForSelectMutations:
    """Mutation-killing tests for build_extract_prompt_for_select."""

    def test_none_description_uses_fallback(self) -> None:
        """When prompt_description is None, a default fallback text is used.

        Kills mutation on the `if field_config.prompt_description:` guard.
        """
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    value_type="string",
                    prompt_description=None,
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "A biological attribute to be extracted" in prompts[1].content

    def test_custom_description_overrides_fallback(self) -> None:
        """When prompt_description is set, it replaces the fallback."""
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    value_type="string",
                    prompt_description="Name of the cell line used in the experiment",
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "Name of the cell line used in the experiment" in prompts[1].content
        assert "A biological attribute to be extracted" not in prompts[1].content

    def test_array_type_noted_in_prompt(self) -> None:
        """Array value_type produces 'multiple values (array)' in prompt."""
        config = SelectConfig(
            fields={
                "diseases": SelectConfigField(
                    value_type="array",
                    prompt_description="List of diseases",
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "multiple values (array)" in prompts[1].content

    def test_string_type_noted_in_prompt(self) -> None:
        """String value_type produces 'single value' in prompt."""
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    value_type="string",
                    prompt_description="Cell line",
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "single value" in prompts[1].content

    def test_system_prompt_content(self) -> None:
        """System prompt contains expected role description."""
        config = SelectConfig(fields={"f": SelectConfigField()})
        prompts = build_extract_prompt_for_select(config)
        assert prompts[0].role == "system"
        assert "smart curator" in prompts[0].content

    def test_field_name_appears_in_prompt(self) -> None:
        """Field name appears in the user prompt."""
        config = SelectConfig(
            fields={
                "my_custom_field": SelectConfigField(
                    value_type="string",
                    prompt_description="Custom description",
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "my_custom_field" in prompts[1].content

    def test_multiple_fields_all_in_prompt(self) -> None:
        """All fields appear in the prompt when config has multiple fields."""
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    value_type="string",
                    prompt_description="Cell line",
                ),
                "organism": SelectConfigField(
                    value_type="string",
                    prompt_description="Organism",
                ),
            },
        )
        prompts = build_extract_prompt_for_select(config)
        assert "cell_line" in prompts[1].content
        assert "organism" in prompts[1].content


# === build_extract_schema_for_select: mutation-killing ===


class TestBuildExtractSchemaForSelectMutations:
    """Mutation-killing tests for build_extract_schema_for_select."""

    def test_schema_has_json_schema_field(self) -> None:
        """Schema includes $schema field.

        Kills mutation that removes the $schema key.
        """
        config = SelectConfig(fields={"f": SelectConfigField()})
        schema = build_extract_schema_for_select(config)
        assert "$schema" in schema
        assert "json-schema.org" in schema["$schema"]

    def test_empty_fields_produces_empty_properties(self) -> None:
        """Empty fields dict produces schema with no properties or required."""
        config = SelectConfig(fields={})
        schema = build_extract_schema_for_select(config)
        assert schema["properties"] == {}
        assert schema["required"] == []


# === build_error_log ===


class TestBuildErrorLog:
    """Tests for build_error_log."""

    def test_captures_exception_type(self) -> None:
        """Error type is the exception class name."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            log = build_error_log(e)
        assert log.error.type == "ValueError"

    def test_captures_exception_message(self) -> None:
        """Error message is str(exception)."""
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError as e:
            log = build_error_log(e)
        assert log.error.message == "something went wrong"

    def test_captures_traceback(self) -> None:
        """Traceback is non-empty and contains the exception type."""
        try:
            raise TypeError("bad type")
        except TypeError as e:
            log = build_error_log(e)
        assert log.error.traceback
        assert "TypeError" in log.error.traceback

    def test_timestamp_format(self) -> None:
        """Timestamp is in YYYYMMDD_HHMMSS format."""
        try:
            raise Exception("test")
        except Exception as e:
            log = build_error_log(e)
        datetime.strptime(log.timestamp, "%Y%m%d_%H%M%S")


# === to_result ===


class TestToResult:
    """Tests for to_result."""

    def _make_run_metadata(self) -> RunMetadata:
        return RunMetadata(
            run_name="test_run",
            model="test-model",
            start_time="20240101_120000",
            status="completed",
        )

    def test_basic_construction(self) -> None:
        """All arguments are correctly wired into Result fields."""
        bs_entries = [{"accession": "SAMN001"}]
        prompt = [
            Prompt(role="system", content="sys"),
            Prompt(role="user", content="usr"),
        ]
        output = [make_llm_output("SAMN001", {"cell_line": "HeLa"})]
        config = Config(ollama_host="http://test:11434")
        run_metadata = self._make_run_metadata()
        format_ = {"type": "object"}

        result = to_result(
            bs_entries=bs_entries,
            prompt=prompt,
            model="test-model",
            output=output,
            config=config,
            run_metadata=run_metadata,
            format_=format_,
            thinking=True,
            args=None,
        )

        assert result.input.bs_entries == bs_entries
        assert result.input.prompt == prompt
        assert result.input.thinking is True
        assert result.input.format == format_
        assert result.input.config == config
        assert result.input.cli_args is None
        assert result.output == output

        assert result.run_metadata == run_metadata

    def test_optional_args_none(self) -> None:
        """Optional parameters default to None without error."""
        result = to_result(
            bs_entries=[],
            prompt=[Prompt(role="system", content="s")],
            model="m",
            output=[],
            config=Config(ollama_host="http://test:11434"),
            run_metadata=self._make_run_metadata(),
            format_=None,
            thinking=None,
            args=None,
        )

        assert result.input.format is None
        assert result.input.thinking is None
        assert result.input.cli_args is None

    def test_return_type_is_result(self) -> None:
        """Return value is a Result instance."""
        result = to_result(
            bs_entries=[],
            prompt=[Prompt(role="system", content="s")],
            model="m",
            output=[],
            config=Config(ollama_host="http://test:11434"),
            run_metadata=self._make_run_metadata(),
        )

        assert isinstance(result, Result)

    def test_input_contains_correct_model(self) -> None:
        """Input.model matches the provided model string."""
        result = to_result(
            bs_entries=[],
            prompt=[Prompt(role="system", content="s")],
            model="llama3.1:70b",
            output=[],
            config=Config(ollama_host="http://test:11434"),
            run_metadata=self._make_run_metadata(),
        )

        assert result.input.model == "llama3.1:70b"

    def test_output_matches_provided_list(self) -> None:
        """Output field contains exactly the provided LlmOutput list."""
        outputs = [
            make_llm_output("SAMN001", {"cell_line": "HeLa"}),
            make_llm_output("SAMN002", {"cell_line": "HEK293"}),
        ]

        result = to_result(
            bs_entries=[{"accession": "SAMN001"}, {"accession": "SAMN002"}],
            prompt=[Prompt(role="system", content="s")],
            model="m",
            output=outputs,
            config=Config(ollama_host="http://test:11434"),
            run_metadata=self._make_run_metadata(),
        )

        assert result.output == outputs
        assert len(result.output) == 2
        assert result.output[0].accession == "SAMN001"
        assert result.output[1].accession == "SAMN002"


# === populate_run_metadata ===


class TestPopulateRunMetadata:
    """Tests for populate_run_metadata."""

    @staticmethod
    def _make_metadata(
        start_time: str = "20240101_120000",
        end_time: str | None = "20240101_121000",
    ) -> RunMetadata:
        return RunMetadata(
            run_name="test_run",
            model="test-model",
            start_time=start_time,
            end_time=end_time,
            status="completed",
        )

    def test_processing_time_computed(self) -> None:
        md = self._make_metadata(start_time="20240101_120000", end_time="20240101_121000")
        output = [make_llm_output("SAMN001", {"cell_line": "HeLa"})]
        result = populate_run_metadata(md, output)
        assert result.processing_time == 600.0

    def test_accuracy_computed_from_select_metrics(self) -> None:
        md = self._make_metadata()
        output = [
            make_llm_output("SAMN001", {"cell_line": "HeLa"}),
            make_llm_output("SAMN002", {"cell_line": "HEK293"}),
        ]
        select_metrics = EvaluationMetrics(
            tp=1, fp=0, fn=1, tn=0, correct=1, total=2, accuracy=0.5,
        )
        result = populate_run_metadata(md, output, select_metrics=select_metrics)
        assert result.matched_entries == 1
        assert result.accuracy == 50.0

    def test_no_end_time_no_processing_time(self) -> None:
        md = self._make_metadata(end_time=None)
        result = populate_run_metadata(md, [])
        assert result.processing_time is None

    def test_no_select_metrics_no_accuracy(self) -> None:
        md = self._make_metadata()
        result = populate_run_metadata(md, [])
        assert result.accuracy is None
        assert result.matched_entries is None

    def test_immutable_original(self) -> None:
        md = self._make_metadata()
        output = [make_llm_output("SAMN001", {"cell_line": "HeLa"})]
        select_metrics = EvaluationMetrics(tp=1, correct=1, total=1, accuracy=1.0)
        result = populate_run_metadata(md, output, select_metrics=select_metrics)
        assert md.processing_time is None
        assert md.total_entries is None
        assert md.accuracy is None
        assert result is not md

    def test_total_entries_set(self) -> None:
        md = self._make_metadata()
        output = [
            make_llm_output("SAMN001", None),
            make_llm_output("SAMN002", None),
            make_llm_output("SAMN003", None),
        ]
        result = populate_run_metadata(md, output)
        assert result.total_entries == 3


# === extract_predicted_term_id ===


def _make_search_result(term_id: str) -> SearchResult:
    return SearchResult(
        term_uri=f"http://example.org/{term_id}",
        term_id=term_id,
        value="test",
        exact_match=True,
    )


class TestExtractPredictedTermId:
    """Tests for extract_predicted_term_id."""

    def test_normal_extraction(self) -> None:
        """Extracts term_id from first value in results[field_name]."""
        sr = SelectResult(
            accession="SAMN001",
            results={"cell_line": {"HeLa": _make_search_result("CVCL:0030")}},
        )
        assert extract_predicted_term_id(sr, "cell_line") == "CVCL:0030"

    def test_field_not_present(self) -> None:
        """Returns None when field_name is not in results."""
        sr = SelectResult(accession="SAMN001", results={})
        assert extract_predicted_term_id(sr, "cell_line") is None

    def test_field_not_dict(self) -> None:
        """Returns None when results[field_name] is not a dict (e.g. string)."""
        sr = SelectResult(accession="SAMN001", results={"cell_line": "some_string"})
        assert extract_predicted_term_id(sr, "cell_line") is None

    def test_all_values_none(self) -> None:
        """Returns None when the SearchResult value is None."""
        sr = SelectResult(
            accession="SAMN001",
            results={"cell_line": {"HeLa": None}},
        )
        assert extract_predicted_term_id(sr, "cell_line") is None


# === compute_classification_metrics ===


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    def test_all_correct(self) -> None:
        """All predictions correct → accuracy=1.0, precision=1.0, recall=1.0, f1=1.0."""
        predicted = {"A": "X", "B": "Y", "C": None}
        expected = {"A": "X", "B": "Y", "C": None}
        m = compute_classification_metrics(predicted, expected)
        assert m.correct == 3
        assert m.total == 3
        assert m.tp == 2
        assert m.tn == 1
        assert m.fp == 0
        assert m.fn == 0
        assert m.accuracy == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_empty_inputs(self) -> None:
        """Empty dicts → total=0, accuracy=None."""
        m = compute_classification_metrics({}, {})
        assert m.total == 0
        assert m.tn == 0
        assert m.accuracy is None
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None

    def test_none_equals_none(self) -> None:
        """None == None counts as correct and TN."""
        predicted: dict[str, str | None] = {"A": None}
        expected: dict[str, str | None] = {"A": None}
        m = compute_classification_metrics(predicted, expected)
        assert m.correct == 1
        assert m.accuracy == pytest.approx(1.0)
        assert m.tp == 0
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 1

    def test_no_positive_in_expected(self) -> None:
        """When expected has no non-None values → precision/recall are None."""
        predicted: dict[str, str | None] = {"A": None, "B": None}
        expected: dict[str, str | None] = {"A": None, "B": None}
        m = compute_classification_metrics(predicted, expected)
        assert m.tn == 2
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None

    def test_fp_and_fn(self) -> None:
        """Verify FP and FN counting."""
        predicted = {"A": "wrong", "B": None}
        expected = {"A": None, "B": "correct"}
        m = compute_classification_metrics(predicted, expected)
        assert m.tp == 0
        assert m.fp == 1  # A: expected=None, predicted="wrong"
        assert m.fn == 1  # B: expected="correct", predicted=None
        assert m.tn == 0
        assert m.correct == 0
        assert m.accuracy == pytest.approx(0.0)

    def test_key_only_in_predicted_counts_as_fp(self) -> None:
        """Key present in predicted but not in expected with non-None value → FP."""
        predicted: dict[str, str | None] = {"A": "X", "B": "Y"}
        expected: dict[str, str | None] = {"A": "X"}
        m = compute_classification_metrics(predicted, expected)
        assert m.total == 2
        assert m.tp == 1  # A: correct
        assert m.fp == 1  # B: expected=None (absent), predicted="Y"
        assert m.fn == 0
        assert m.tn == 0

    def test_key_only_in_predicted_none_counts_as_tn(self) -> None:
        """Key present in predicted but not in expected with None value → TN."""
        predicted: dict[str, str | None] = {"A": "X", "B": None}
        expected: dict[str, str | None] = {"A": "X"}
        m = compute_classification_metrics(predicted, expected)
        assert m.total == 2
        assert m.tp == 1  # A: correct
        assert m.tn == 1  # B: expected=None (absent), predicted=None
        assert m.fp == 0
        assert m.fn == 0
        assert m.correct == 2

    def test_key_only_in_expected(self) -> None:
        """Key present in expected but not in predicted → FN if non-None, TN if None."""
        predicted: dict[str, str | None] = {"A": "X"}
        expected: dict[str, str | None] = {"A": "X", "B": "Y", "C": None}
        m = compute_classification_metrics(predicted, expected)
        assert m.total == 3
        assert m.tp == 1  # A: correct
        assert m.fn == 1  # B: expected="Y", predicted=None (absent)
        assert m.tn == 1  # C: expected=None, predicted=None (absent)
        assert m.fp == 0


# === evaluate_select_output ===


class TestEvaluateSelectOutput:
    """Tests for evaluate_select_output."""

    def _make_mapping_value(
        self,
        mapping_answer_id: str | None = None,
    ) -> MappingValue:
        return MappingValue(
            experiment_type="RNA-seq",
            extraction_answer=None,
            mapping_answer_id=mapping_answer_id,
            mapping_answer_label=None,
        )

    def test_correct_match(self) -> None:
        """Predicted term_id matches answer → TP."""
        sr = SelectResult(
            accession="A",
            results={"cell_line": {"HeLa": _make_search_result("CVCL:0030")}},
        )
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([sr], mapping)
        assert m.tp == 1
        assert m.tn == 0
        assert m.correct == 1

    def test_mismatch(self) -> None:
        """Predicted term_id != answer → FN."""
        sr = SelectResult(
            accession="A",
            results={"cell_line": {"HeLa": _make_search_result("CVCL:9999")}},
        )
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([sr], mapping)
        assert m.tp == 0
        assert m.fn == 1

    def test_none_equals_none(self) -> None:
        """Both predicted and answer are None → correct, TN."""
        sr = SelectResult(accession="A", results={})
        mapping = {"A": self._make_mapping_value(None)}
        m = evaluate_select_output([sr], mapping)
        assert m.correct == 1
        assert m.tn == 1

    def test_false_positive(self) -> None:
        """Predicted something but answer is None → FP."""
        sr = SelectResult(
            accession="A",
            results={"cell_line": {"HeLa": _make_search_result("CVCL:0030")}},
        )
        mapping = {"A": self._make_mapping_value(None)}
        m = evaluate_select_output([sr], mapping)
        assert m.fp == 1

    def test_false_negative(self) -> None:
        """No prediction but answer exists → FN."""
        sr = SelectResult(accession="A", results={})
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([sr], mapping)
        assert m.fn == 1

    def test_empty_results(self) -> None:
        """Empty inputs → total=0."""
        m = evaluate_select_output([], {})
        assert m.total == 0

    def test_custom_field_name(self) -> None:
        """field_name parameter selects which field to evaluate."""
        sr = SelectResult(
            accession="A",
            results={"organism": {"Human": _make_search_result("NCBITaxon:9606")}},
        )
        mapping = {"A": self._make_mapping_value("NCBITaxon:9606")}
        m = evaluate_select_output([sr], mapping, field_name="organism")
        assert m.tp == 1
        assert m.correct == 1

    def test_precision_recall_f1(self) -> None:
        """Verify precision/recall/f1 calculation."""
        results = [
            SelectResult(
                accession="A",
                results={"cell_line": {"HeLa": _make_search_result("CVCL:0030")}},
            ),
            SelectResult(
                accession="B",
                results={"cell_line": {"HEK": _make_search_result("CVCL:9999")}},
            ),
            SelectResult(accession="C", results={}),
        ]
        mapping = {
            "A": self._make_mapping_value("CVCL:0030"),
            "B": self._make_mapping_value("CVCL:0045"),
            "C": self._make_mapping_value(None),
        }
        m = evaluate_select_output(results, mapping)
        assert m.tp == 1
        assert m.fp == 0
        assert m.fn == 1
        assert m.tn == 1  # C: None==None
        assert m.correct == 2  # A correct, C correct (None==None)
        assert m.total == 3
        assert m.precision == pytest.approx(1.0)  # 1/(1+0)
        assert m.recall == pytest.approx(0.5)  # 1/(1+1)

    def test_multiple_entries(self) -> None:
        """Multiple entries are processed correctly."""
        results = [
            SelectResult(
                accession="A",
                results={"cell_line": {"HeLa": _make_search_result("CVCL:0030")}},
            ),
            SelectResult(
                accession="B",
                results={"cell_line": {"K562": _make_search_result("CVCL:0004")}},
            ),
        ]
        mapping = {
            "A": self._make_mapping_value("CVCL:0030"),
            "B": self._make_mapping_value("CVCL:0004"),
        }
        m = evaluate_select_output(results, mapping)
        assert m.tp == 2
        assert m.correct == 2
        assert m.accuracy == pytest.approx(1.0)
