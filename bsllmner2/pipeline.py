import traceback
from datetime import datetime, timezone

from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import Config
from bsllmner2.metrics import Metrics
from bsllmner2.models import (
    BsEntries,
    CliExtractArgs,
    CliSelectArgs,
    ErrorInfo,
    ErrorLog,
    Evaluation,
    LlmOutput,
    Mapping,
    Prompt,
    Result,
    RunMetadata,
    SelectConfig,
    WfInput,
)


def evaluate_output(output: list[LlmOutput], mapping: Mapping) -> list[Evaluation]:
    """Evaluate the LLM outputs against the mapping.

    Returns a list of Evaluation objects containing the results.
    """
    evaluations = []
    for entry in output:
        accession = entry.accession
        if entry.output is None or not isinstance(entry.output, dict):
            actual = None
        else:
            actual = entry.output.get("cell_line", None)
        if accession not in mapping:
            expected = None
        else:
            expected = mapping[accession].extraction_answer
        evaluation = Evaluation(
            accession=accession,
            expected=expected,
            actual=actual,
            match=(actual is not None and actual == expected),
        )
        evaluations.append(evaluation)

    return evaluations


def to_result(
    bs_entries: BsEntries,
    mapping: Mapping | None,
    prompt: list[Prompt],
    model: str,
    output: list[LlmOutput],
    evaluation: list[Evaluation],
    config: Config,
    run_metadata: RunMetadata,
    format_: JsonSchemaValue | None = None,
    thinking: bool | None = None,
    args: CliExtractArgs | CliSelectArgs | None = None,
    metrics: list[Metrics] | None = None,
) -> Result:
    return Result(
        input=WfInput(
            bs_entries=bs_entries,
            mapping=mapping,
            prompt=prompt,
            model=model,
            thinking=thinking,
            format=format_,
            config=config,
            cli_args=args,
        ),
        output=output,
        evaluation=evaluation,
        metrics=metrics,
        run_metadata=run_metadata,
    )


def get_now_str() -> str:
    """Return current UTC time as string in YYYYMMDD_HHMMSS format."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def compute_processing_time(start_time: str, end_time: str) -> float:
    dt_format = "%Y%m%d_%H%M%S"
    start_dt = datetime.strptime(start_time, dt_format)
    end_dt = datetime.strptime(end_time, dt_format)

    seconds = (end_dt - start_dt).total_seconds()
    if seconds < 0:
        raise ValueError(f"end_time ({end_time}) is before start_time ({start_time})")
    return seconds


def build_error_log(
    exc: Exception,
) -> ErrorLog:
    return ErrorLog(
        timestamp=get_now_str(),
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
        "Here is the input metadata:\n"
    )

    return [
        Prompt(role="system", content=system_content),
        Prompt(role="user", content=user_content),
    ]
