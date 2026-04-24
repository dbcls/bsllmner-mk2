"""Tests for pipeline module (evaluation + builders + time utils)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bsllmner2.io import load_select_config
from bsllmner2.models import (
    ExtractEntry,
    MappingValue,
    ResolvedValue,
    RunMetadata,
    SelectConfig,
    SelectConfigField,
    SelectEntry,
)
from bsllmner2.pipeline import (
    build_error_log,
    build_extract_prompt_for_select,
    build_extract_schema_for_select,
    compute_classification_metrics,
    compute_processing_time,
    evaluate_select_output,
    extract_predicted_term_id,
    get_now,
    populate_run_metadata,
)
from tests.py_tests.conftest import make_extract_entry


class TestGetNow:
    """Test cases for get_now function."""

    def test_returns_datetime(self) -> None:
        """Test that get_now returns a datetime instance."""
        result = get_now()
        assert isinstance(result, datetime)

    def test_returns_utc_timezone(self) -> None:
        """Test that get_now returns a UTC-aware datetime."""
        result = get_now()
        assert result.tzinfo == timezone.utc

    def test_returns_current_time(self) -> None:
        """Test that get_now returns a time between before and after snapshots."""
        before = datetime.now(timezone.utc)
        result = get_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


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
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 1, 30, tzinfo=timezone.utc)
        result = compute_processing_time(start, end)
        assert result == 90.0

    def test_same_time(self) -> None:
        """Same start and end → 0.0."""
        t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert compute_processing_time(t, t) == 0.0

    def test_one_hour_difference(self) -> None:
        """One hour difference → 3600.0."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        assert compute_processing_time(start, end) == 3600.0

    def test_cross_day_boundary(self) -> None:
        """Crossing day boundary works correctly."""
        start = datetime(2024, 1, 1, 23, 59, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 1, 0, tzinfo=timezone.utc)
        assert compute_processing_time(start, end) == 120.0

    def test_end_before_start_raises_value_error(self) -> None:
        """end_time < start_time raises ValueError."""
        start = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="before start_time"):
            compute_processing_time(start, end)


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


# === build_extract_prompt_for_select: Category assignment rules ===


class TestBuildExtractPromptForSelectCategoryRules:
    """The 'Category assignment rules' block guards against cross-field value leaks."""

    def _prompt(self) -> str:
        config = SelectConfig(
            fields={
                "cell_line": SelectConfigField(
                    value_type="string",
                    prompt_description="Cell line",
                ),
            },
        )
        return build_extract_prompt_for_select(config)[1].content

    def test_section_header_present(self) -> None:
        assert "Category assignment rules:" in self._prompt()

    def test_one_category_per_value_rule(self) -> None:
        content = self._prompt()
        assert "at most ONE category" in content
        assert "biological meaning" in content

    def test_biological_meaning_over_attribute_label(self) -> None:
        """The rule must give the 'drug attribute contains HeLa' worked example."""
        content = self._prompt()
        assert '"drug"' in content
        assert '"HeLa"' in content
        assert '"cell_line"' in content

    def test_experimental_control_exclusion(self) -> None:
        content = self._prompt()
        assert "Do NOT extract experimental control terms" in content
        # A representative subset of listed control terms must appear.
        for term in ("negative control", "vehicle", "mock", "scramble", "shControl", "siControl"):
            assert term in content

    def test_rules_placed_between_output_rules_and_metadata(self) -> None:
        """Category rules must be appended AFTER Output rules and BEFORE the input stub."""
        content = self._prompt()
        out_idx = content.index("Output rules:")
        cat_idx = content.index("Category assignment rules:")
        meta_idx = content.index("Here is the input metadata:")
        assert out_idx < cat_idx < meta_idx


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

    def test_timestamp_is_datetime(self) -> None:
        """Timestamp is a datetime instance."""
        try:
            raise Exception("test")
        except Exception as e:
            log = build_error_log(e)
        assert isinstance(log.timestamp, datetime)


# === populate_run_metadata ===


class TestPopulateRunMetadata:
    """Tests for populate_run_metadata."""

    @staticmethod
    def _make_metadata(
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> RunMetadata:
        if start_time is None:
            start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        if end_time is None:
            end_time = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
        return RunMetadata(
            run_name="test_run",
            model="test-model",
            start_time=start_time,
            end_time=end_time,
            status="completed",
        )

    def test_processing_time_computed(self) -> None:
        md = self._make_metadata()
        output = [make_extract_entry("SAMN001", {"cell_line": "HeLa"})]
        result = populate_run_metadata(md, output)
        assert result.processing_time_sec == 600.0

    def test_no_end_time_no_processing_time(self) -> None:
        md_no_end = RunMetadata(
            run_name="test_run",
            model="test-model",
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=None,
            status="completed",
        )
        result = populate_run_metadata(md_no_end, [])
        assert result.processing_time_sec is None

    def test_immutable_original(self) -> None:
        md = self._make_metadata()
        output = [make_extract_entry("SAMN001", {"cell_line": "HeLa"})]
        result = populate_run_metadata(md, output)
        assert md.processing_time_sec is None
        assert md.total_entries is None
        assert result is not md

    def test_total_entries_set(self) -> None:
        md = self._make_metadata()
        output = [make_extract_entry("SAMN001"), make_extract_entry("SAMN002"), make_extract_entry("SAMN003")]
        result = populate_run_metadata(md, output)
        assert result.total_entries == 3


# === extract_predicted_term_id ===


class TestExtractPredictedTermId:
    """Tests for extract_predicted_term_id."""

    def test_normal_extraction(self) -> None:
        """Extracts term_id from first ResolvedValue in results[field_name]."""
        se = SelectEntry(
            extract=ExtractEntry(accession="SAMN001"),
            results={"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:0030")]},
        )
        assert extract_predicted_term_id(se, "cell_line") == "CVCL:0030"

    def test_field_not_present(self) -> None:
        """Returns None when field_name is not in results."""
        se = SelectEntry(
            extract=ExtractEntry(accession="SAMN001"),
            results={},
        )
        assert extract_predicted_term_id(se, "cell_line") is None

    def test_field_empty_list(self) -> None:
        """Returns None when results[field_name] is an empty list."""
        se = SelectEntry(
            extract=ExtractEntry(accession="SAMN001"),
            results={"cell_line": []},
        )
        assert extract_predicted_term_id(se, "cell_line") is None

    def test_no_term_id_in_resolved_values(self) -> None:
        """Returns None when ResolvedValue has no term_id."""
        se = SelectEntry(
            extract=ExtractEntry(accession="SAMN001"),
            results={"cell_line": [ResolvedValue(value="HeLa")]},
        )
        assert extract_predicted_term_id(se, "cell_line") is None


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

    @staticmethod
    def _se(accession: str, results: dict[str, list[ResolvedValue]] | None = None) -> SelectEntry:
        return SelectEntry(
            extract=ExtractEntry(accession=accession),
            results=results if results is not None else {},
        )

    def test_correct_match(self) -> None:
        """Predicted term_id matches answer → TP."""
        se = self._se("A", {"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:0030")]})
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([se], mapping)
        assert m.tp == 1
        assert m.tn == 0
        assert m.correct == 1

    def test_mismatch(self) -> None:
        """Predicted term_id != answer → FN."""
        se = self._se("A", {"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:9999")]})
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([se], mapping)
        assert m.tp == 0
        assert m.fn == 1

    def test_none_equals_none(self) -> None:
        """Both predicted and answer are None → correct, TN."""
        se = self._se("A", {})
        mapping = {"A": self._make_mapping_value(None)}
        m = evaluate_select_output([se], mapping)
        assert m.correct == 1
        assert m.tn == 1

    def test_false_positive(self) -> None:
        """Predicted something but answer is None → FP."""
        se = self._se("A", {"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:0030")]})
        mapping = {"A": self._make_mapping_value(None)}
        m = evaluate_select_output([se], mapping)
        assert m.fp == 1

    def test_false_negative(self) -> None:
        """No prediction but answer exists → FN."""
        se = self._se("A", {})
        mapping = {"A": self._make_mapping_value("CVCL:0030")}
        m = evaluate_select_output([se], mapping)
        assert m.fn == 1

    def test_empty_results(self) -> None:
        """Empty inputs → total=0."""
        m = evaluate_select_output([], {})
        assert m.total == 0

    def test_custom_field_name(self) -> None:
        """field_name parameter selects which field to evaluate."""
        se = self._se("A", {"organism": [ResolvedValue(value="Human", term_id="NCBITaxon:9606")]})
        mapping = {"A": self._make_mapping_value("NCBITaxon:9606")}
        m = evaluate_select_output([se], mapping, field_name="organism")
        assert m.tp == 1
        assert m.correct == 1

    def test_precision_recall_f1(self) -> None:
        """Verify precision/recall/f1 calculation."""
        results = [
            self._se("A", {"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:0030")]}),
            self._se("B", {"cell_line": [ResolvedValue(value="HEK", term_id="CVCL:9999")]}),
            self._se("C", {}),
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
            self._se("A", {"cell_line": [ResolvedValue(value="HeLa", term_id="CVCL:0030")]}),
            self._se("B", {"cell_line": [ResolvedValue(value="K562", term_id="CVCL:0004")]}),
        ]
        mapping = {
            "A": self._make_mapping_value("CVCL:0030"),
            "B": self._make_mapping_value("CVCL:0004"),
        }
        m = evaluate_select_output(results, mapping)
        assert m.tp == 2
        assert m.correct == 2
        assert m.accuracy == pytest.approx(1.0)
