import contextlib
import traceback
from datetime import datetime, timezone

from pydantic.json_schema import JsonSchemaValue

from bsllmner2.models import (
    ErrorInfo,
    ErrorLog,
    EvaluationMetrics,
    ExtractEntry,
    ExtractResult,
    Mapping,
    PerformanceSummary,
    Prompt,
    RunMetadata,
    SelectConfig,
    SelectEntry,
)


def extract_predicted_term_id(select_entry: SelectEntry, field_name: str) -> str | None:
    """Extract term_id from SelectEntry.results[field_name].

    Returns the term_id of the first ResolvedValue found, or None.
    """
    field_info = select_entry.results.get(field_name)
    if field_info is None:
        return None
    for rv in field_info:
        if rv.term_id is not None:
            return rv.term_id
    return None


def compute_classification_metrics(
    predicted: dict[str, str | None],
    expected: dict[str, str | None],
) -> EvaluationMetrics:
    """Compute accuracy/precision/recall/f1 from predicted vs expected mappings.

    None is treated as a valid class (None == None is a correct prediction).
    Precision/recall are computed over the "positive" class (non-None values).
    Keys are taken from the union of predicted and expected to avoid missing FPs.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    all_keys = expected.keys() | predicted.keys()
    total = len(all_keys)

    for key in all_keys:
        answer = expected.get(key)
        pred = predicted.get(key)

        if answer is not None:
            if pred == answer:
                tp += 1
            else:
                fn += 1
        elif pred is not None:
            fp += 1
        else:
            tn += 1

    correct = tp + tn
    accuracy = correct / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return EvaluationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total=total,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_select_output(
    select_entries: list[SelectEntry],
    mapping: Mapping,
    field_name: str = "cell_line",
) -> EvaluationMetrics:
    """Evaluate Select results by comparing term_id against mapping_answer_id."""
    predicted: dict[str, str | None] = {}
    expected: dict[str, str | None] = {}
    for se in select_entries:
        accession = se.extract.accession
        predicted[accession] = extract_predicted_term_id(se, field_name)
        if accession in mapping:
            expected[accession] = mapping[accession].mapping_answer_id
        else:
            expected[accession] = None
    return compute_classification_metrics(predicted, expected)


def build_extract_result(
    entries: list[ExtractEntry],
    run_metadata: RunMetadata,
    performance: PerformanceSummary | None = None,
    errors: list[ErrorLog] | None = None,
) -> ExtractResult:
    return ExtractResult(
        entries=entries,
        run_metadata=run_metadata,
        performance=performance,
        errors=errors or [],
    )


def get_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def compute_processing_time(start_time: datetime, end_time: datetime) -> float:
    seconds = (end_time - start_time).total_seconds()
    if seconds < 0:
        raise ValueError(f"end_time ({end_time}) is before start_time ({start_time})")
    return seconds


def populate_run_metadata(
    run_metadata: RunMetadata,
    entries: list[ExtractEntry],
) -> RunMetadata:
    """Compute and fill unused RunMetadata fields. Returns a new instance."""
    updates: dict[str, object] = {
        "total_entries": len(entries),
    }

    if run_metadata.end_time is not None:
        with contextlib.suppress(ValueError):
            updates["processing_time_sec"] = compute_processing_time(
                run_metadata.start_time,
                run_metadata.end_time,
            )

    return run_metadata.model_copy(update=updates)


def build_error_log(
    exc: Exception,
) -> ErrorLog:
    return ErrorLog(
        timestamp=get_now(),
        error=ErrorInfo(
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        ),
    )


def build_extract_schema_for_select(config: SelectConfig) -> JsonSchemaValue:
    properties: dict[str, JsonSchemaValue] = {}

    for field_name, field_config in config.fields.items():
        if field_config.value_type == "string":
            properties[field_name] = {
                "type": ["string", "null"],
            }
        elif field_config.value_type == "array":
            properties[field_name] = {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                },
            }
        else:
            raise ValueError(f"Unsupported value_type {field_config.value_type} for field {field_name}")

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": list(config.fields.keys()),
        "additionalProperties": False,
    }


def build_extract_prompt_for_select(config: SelectConfig) -> list[Prompt]:
    category_lines: list[str] = []
    for field_name, field_config in config.fields.items():
        if field_config.value_type == "array":
            type_note = "multiple values (array)"
        else:
            type_note = "single value"
        if field_config.prompt_description:
            desc = field_config.prompt_description
        else:
            desc = "A biological attribute to be extracted from the metadata."

        category_lines.append(f'- "{field_name} ({type_note})":\n  - {desc}')

    categories_block = "\n".join(category_lines)

    system_content = "You are a smart curator of biological data."

    user_content = (
        "I will input JSON formatted metadata of a sample for a biological experiment.\n"
        "Your task is to extract relevant biological information (if present) from the input data and format it according to the specified schema.\n"
        "\n"
        "---\n"
        "Categories to extract:\n"
        f"{categories_block}\n"
        "\n"
        "---\n"
        "Output rules:\n"
        "  - Return only JSON, matching the provided schema (via the `format` option).\n"
        "  - For categories with `string` value_type:\n"
        "      - If you can identify a value, output a single concise canonical name (not a free-form description).\n"
        "      - If no value can be found, output null.\n"
        "  - For categories with `array` value_type:\n"
        "      - If you can identify one or more values, output an array of concise canonical names.\n"
        "      - If no values can be found, output null (not an empty array).\n"
        "  - Prefer exact mentions in the input; if multiple candidates exist, pick the most specific and widely recognized.\n"
        "  - Do not hallucinate or infer values that are absent from the input.\n"
        "  - Do not mix different kinds of concepts in the same category.\n"
        "\n"
        "---\n"
        "Category assignment rules:\n"
        "  - Each extracted value must belong to at most ONE category. If a value could fit multiple\n"
        "    categories, pick the single most appropriate one based on its biological meaning.\n"
        "  - Classify values by their biological meaning, not by attribute keys or labels in the input\n"
        '    metadata (e.g., if an attribute labeled "drug" actually contains "HeLa", extract it as\n'
        '    "cell_line", not "drug").\n'
        "  - Do NOT extract experimental control terms into any category. They are experimental\n"
        '    conditions, not biological entities (e.g., "negative control", "NC", "vehicle", "mock",\n'
        '    "empty vector", "scramble", "non-targeting", "shControl", "siControl").\n'
        "\n"
        "---\n"
        "Here is the input metadata:\n"
    )

    return [
        Prompt(role="system", content=system_content),
        Prompt(role="user", content=user_content),
    ]
