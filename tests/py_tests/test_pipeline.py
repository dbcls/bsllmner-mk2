"""Tests for pipeline module (evaluation + builders + time utils)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from ollama import ChatResponse, Message

from bsllmner2.io import load_select_config
from bsllmner2.models import (
    Evaluation,
    LlmOutput,
    MappingValue,
    SelectConfig,
    SelectConfigField,
)
from bsllmner2.pipeline import (
    build_extract_prompt_for_select,
    build_extract_schema_for_select,
    compute_processing_time,
    evaluate_output,
    get_now_str,
)


def _make_chat_response() -> ChatResponse:
    """Create a minimal ChatResponse for test data construction."""
    return ChatResponse(
        model="test",
        created_at="2024-01-01T00:00:00Z",
        message=Message(role="assistant", content=""),
        done=True,
    )


def _make_llm_output(
    accession: str,
    output: object = None,
) -> LlmOutput:
    """Create a LlmOutput with minimal boilerplate."""
    return LlmOutput(
        accession=accession,
        output=output,
        chat_response=_make_chat_response(),
    )


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


# === evaluate_output ===


class TestEvaluateOutput:
    """Tests for evaluate_output.

    Known limitation:
    - "cell_line" is hardcoded: other fields silently return actual=None
    """

    def test_matching_output(self) -> None:
        """Exact match: actual == expected → match=True."""
        output = [_make_llm_output("SAMN001", {"cell_line": "HeLa"})]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id="CVCL:0030",
                mapping_answer_label="HeLa",
            ),
        }
        evals = evaluate_output(output, mapping)
        assert len(evals) == 1
        assert evals[0].match is True
        assert evals[0].actual == "HeLa"
        assert evals[0].expected == "HeLa"

    def test_non_matching_output(self) -> None:
        """Mismatch: actual != expected → match=False."""
        output = [_make_llm_output("SAMN001", {"cell_line": "HEK293"})]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id="CVCL:0030",
                mapping_answer_label="HeLa",
            ),
        }
        evals = evaluate_output(output, mapping)
        assert evals[0].match is False
        assert evals[0].actual == "HEK293"
        assert evals[0].expected == "HeLa"

    def test_output_is_none(self) -> None:
        """When entry.output is None, actual becomes None."""
        output = [_make_llm_output("SAMN001", None)]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
        }
        evals = evaluate_output(output, mapping)
        assert evals[0].actual is None
        assert evals[0].match is False

    def test_output_not_dict(self) -> None:
        """When entry.output is not a dict (e.g. a string), actual becomes None."""
        output = [_make_llm_output("SAMN001", "just a string")]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
        }
        evals = evaluate_output(output, mapping)
        assert evals[0].actual is None
        assert evals[0].match is False

    def test_accession_not_in_mapping(self) -> None:
        """Accession absent from mapping → expected=None."""
        output = [_make_llm_output("SAMN_MISSING", {"cell_line": "HeLa"})]
        mapping: dict[str, MappingValue] = {}
        evals = evaluate_output(output, mapping)
        assert evals[0].expected is None
        assert evals[0].actual == "HeLa"
        assert evals[0].match is False

    def test_empty_output_list(self) -> None:
        """Empty output list → empty evaluations."""
        evals = evaluate_output([], {})
        assert evals == []

    def test_both_none_is_not_match(self) -> None:
        """When both actual and expected are None, match=False.

        actual=None means the LLM produced no output, so it should not
        count as a correct prediction even when expected is also None.
        """
        output = [_make_llm_output("SAMN_MISSING", None)]
        mapping: dict[str, MappingValue] = {}
        evals = evaluate_output(output, mapping)
        assert evals[0].actual is None
        assert evals[0].expected is None
        assert evals[0].match is False

    def test_hardcoded_cell_line_key(self) -> None:
        """Limitation documentation: evaluate_output only reads "cell_line" from output.

        If the schema uses a different field name (e.g. "organism"),
        actual is always None regardless of what the LLM produced.
        """
        output = [_make_llm_output("SAMN001", {"organism": "Homo sapiens"})]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="Homo sapiens",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
        }
        evals = evaluate_output(output, mapping)
        # The output dict has "organism" but evaluate_output only looks at "cell_line"
        assert evals[0].actual is None
        assert evals[0].expected == "Homo sapiens"
        assert evals[0].match is False

    def test_multiple_entries(self) -> None:
        """Multiple entries produce one evaluation per entry in order."""
        output = [
            _make_llm_output("SAMN001", {"cell_line": "HeLa"}),
            _make_llm_output("SAMN002", {"cell_line": "HEK293"}),
            _make_llm_output("SAMN003", None),
        ]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
            "SAMN002": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="K562",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
        }
        evals = evaluate_output(output, mapping)
        assert len(evals) == 3
        assert evals[0].accession == "SAMN001"
        assert evals[0].match is True
        assert evals[1].accession == "SAMN002"
        assert evals[1].match is False
        assert evals[2].accession == "SAMN003"

    def test_output_dict_without_cell_line_key(self) -> None:
        """Dict output without "cell_line" key → actual=None via .get()."""
        output = [_make_llm_output("SAMN001", {"tissue": "brain"})]
        mapping = {
            "SAMN001": MappingValue(
                experiment_type="RNA-seq",
                extraction_answer="HeLa",
                mapping_answer_id=None,
                mapping_answer_label=None,
            ),
        }
        evals = evaluate_output(output, mapping)
        assert evals[0].actual is None

    def test_output_is_list(self) -> None:
        """List output is not a dict → actual=None."""
        output = [_make_llm_output("SAMN001", ["HeLa", "HEK293"])]
        mapping: dict[str, MappingValue] = {}
        evals = evaluate_output(output, mapping)
        assert evals[0].actual is None

    def test_returns_evaluation_objects(self) -> None:
        """Return values are Evaluation model instances."""
        output = [_make_llm_output("SAMN001", {"cell_line": "HeLa"})]
        evals = evaluate_output(output, {})
        assert isinstance(evals[0], Evaluation)


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
        """additionalProperties is always False."""
        config = SelectConfig(fields={"f": SelectConfigField()})
        schema = build_extract_schema_for_select(config)
        assert schema["additionalProperties"] is False
