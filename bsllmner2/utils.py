import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import ijson
import yaml
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import EXTRACT_RESULT_DIR, LOGGER, PROGRESS_DIR, SELECT_RESULT_DIR, Config
from bsllmner2.metrics import Metrics
from bsllmner2.schema import (
    BsEntries,
    CliExtractArgs,
    CliSelectArgs,
    ErrorInfo,
    ErrorLog,
    Evaluation,
    LlmOutput,
    Mapping,
    MappingValue,
    Prompt,
    Result,
    RunMetadata,
    SelectConfig,
    SelectResult,
    WfInput,
)


def load_bs_entries(path: Path) -> BsEntries:
    """Load and return a list of BioSample entries from a JSON or JSONL file.

    If the file is JSONL, each line is treated as a separate JSON object.
    If the file is JSON, it is expected to be a list of dictionaries.

    Raises:
        ValueError: If the file is neither JSON nor JSONL.

    """
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        try:
            # Try to load as JSON
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return data
            raise ValueError(
                f"Invalid format in {path}:\n"
                f"  JSON file must contain a list of dictionaries.\n"
                f'  Expected: [{{"accession": "SAMD00000001", "title": "...", ...}}]',
            )
        except json.JSONDecodeError as outer_e:
            # If JSON fails, try to load as JSONL
            f.seek(0)
            jl_data: list[dict[str, Any]] = []
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as inner_e:
                    raise ValueError(
                        f"Invalid JSON format in {path} at line {line_no}:\n"
                        f"  Error: {inner_e}\n"
                        f"  Line content: {line[:100]}{'...' if len(line) > 100 else ''}\n"
                        f"  Expected formats:\n"
                        f'    - JSON array: [{{"accession": "SAMD00000001", ...}}]\n'
                        f"    - JSONL: one JSON object per line",
                    ) from inner_e
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Invalid entry in {path} at line {line_no}:\n"
                        f"  Each line in JSONL file must be a JSON object (dictionary).\n"
                        f"  Got: {type(entry).__name__}",
                    ) from outer_e
                jl_data.append(entry)
            if not jl_data:
                raise ValueError(
                    f"No valid entries in {path}:\n"
                    f"  File contains no valid JSON objects.\n"
                    f"  Expected formats:\n"
                    f'    - JSON array: [{{"accession": "SAMD00000001", ...}}]\n'
                    f"    - JSONL: one JSON object per line",
                ) from outer_e
            return jl_data


def load_prompt_file(path: Path) -> list[Prompt]:
    """Load a prompt file from the given path.

    The file should be in YAML format, containing a dictionary where each key is a number as a string.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Prompt file {path} must contain a list of prompts.")

    return [Prompt(**item) for item in raw_data]


def load_mapping(path: Path) -> Mapping:
    expected_headers = [
        "BioSample ID",
        "Experiment type",
        "extraction answer",
        "mapping answer ID",
        "mapping answer label",
    ]

    mapping: Mapping = {}

    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return {}

    header_fields = lines[0].split("\t")
    if header_fields != expected_headers:
        raise ValueError(f"Header mismatch: expected {expected_headers}, got {header_fields}")

    for lineno, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(expected_headers):
            raise ValueError(
                f"The number of fields in line {lineno} does not match the header: {len(fields)} != {len(expected_headers)}",
            )
        accession, experiment_type, extraction_answer, mapping_answer_id, mapping_answer_label = fields
        if accession in mapping:
            raise ValueError(f"Duplicate accession found: {accession} at line {lineno}")

        mapping[accession] = MappingValue(
            experiment_type=experiment_type,
            extraction_answer=extraction_answer or None,
            mapping_answer_id=mapping_answer_id or None,
            mapping_answer_label=mapping_answer_label or None,
        )

    return mapping


def load_format_schema(path: Path) -> JsonSchemaValue:
    """Load a JSON schema file from the given path.

    The file should contain a valid JSON schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Format schema file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        try:
            schema = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema in file {path}: {e}") from e

    return cast("JsonSchemaValue", schema)


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
            match=(actual == expected),
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
    return (end_dt - start_dt).total_seconds()


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


def dump_extract_result(result: Result, run_name: str) -> Path:
    EXTRACT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = EXTRACT_RESULT_DIR.joinpath(f"{run_name}.json")
    with result_file.open("w", encoding="utf-8") as f:
        json_str = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        f.write(json_str.encode("utf-8", errors="replace").decode("utf-8"))

    return result_file


def dump_select_result(result: list[SelectResult], run_name: str) -> Path:
    SELECT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = SELECT_RESULT_DIR.joinpath(f"select_{run_name}.json")
    with result_file.open("w", encoding="utf-8") as f:
        json_str = json.dumps([res.model_dump() for res in result], ensure_ascii=False, indent=2)
        f.write(json_str.encode("utf-8", errors="replace").decode("utf-8"))

    return result_file


def load_extract_result(path: Path) -> Result:
    if not path.exists():
        raise FileNotFoundError(f"Result file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    result = Result.model_validate_json(content)
    result.run_metadata.completed_count = load_progress_count(result.run_metadata.run_name)

    return result


def load_run_metadata(path: Path) -> RunMetadata:
    if not path.exists():
        raise FileNotFoundError(f"Run metadata file {path} does not exist.")

    with path.open("rb") as f:
        iterator = ijson.items(f, "run_metadata")
        try:
            data: Any = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"No run metadata found in file {path}") from exc

    return RunMetadata.model_validate(data)


def list_run_metadata() -> list[RunMetadata]:
    if not EXTRACT_RESULT_DIR.exists():
        return []

    run_metadata_list = []
    for file in EXTRACT_RESULT_DIR.glob("*.json"):
        try:
            run_name = file.name.removesuffix(".json")
            run_metadata = load_run_metadata(file)
            run_metadata.completed_count = load_progress_count(run_name)
            run_metadata_list.append(run_metadata)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping file {file}: {e}")

    return run_metadata_list


def list_run_names() -> list[str]:
    """List all run names from the result directory.

    Returns:
        A list of run names (without file extensions).

    """
    if not EXTRACT_RESULT_DIR.exists():
        return []

    return [file.name.removesuffix(".json") for file in EXTRACT_RESULT_DIR.glob("*.json") if file.is_file()]


def load_progress_count(run_name: str) -> int | None:
    """Load the progress count from a file in the PROGRESS_DIR.

    The file is named after the run_name and contains one accession per line.

    Returns:
        The number of processed entries.

    """
    progress_file = PROGRESS_DIR.joinpath(f"{run_name}.txt")
    if not progress_file.exists():
        return None

    with progress_file.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_select_config(path: Path) -> SelectConfig:
    if not path.exists():
        raise FileNotFoundError(f"Select configuration file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in select configuration file {path}: {e}") from e

    return SelectConfig.model_validate(data)


def build_select_schema(config: SelectConfig) -> JsonSchemaValue:
    properties = {
        field_name: {
            "type": ["string", "null"],
            "description": f"Selected term ID for {field_name}",
        }
        for field_name in config.fields
    }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": list(config.fields.keys()),
        "additionalProperties": False,
    }


def build_select_prompt(config: SelectConfig) -> list[Prompt]:
    description = "\n".join(
        f"- {field_name}: {field_config.prompt_description}"
        for field_name, field_config in config.fields.items()
        if field_config.prompt_description
    )
    system_content = (
        "You are an expert in biomedical ontologies. "
        "Based on the provided information, select the most appropriate term for each of the following fields:\n"
        f"{description}\n"
        "Provide your answers in JSON format with the field names as keys and the selected term IDs as values. "
        "If no appropriate term is found, use null."
    )
    return [
        Prompt(role="system", content=system_content),
        Prompt(role="user", content="Here is the information: {{ner_extraction}}"),
    ]


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


def load_extract_resume_file(run_name: str) -> list[LlmOutput]:
    extract_resume_file_path = EXTRACT_RESULT_DIR.joinpath(f"{run_name}_resume.json")
    if not extract_resume_file_path.exists():
        return []

    with extract_resume_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return []
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError(f"Resume file {extract_resume_file_path} must contain a list of LLM outputs.")
    outputs: list[LlmOutput] = []
    for item in data:
        output = LlmOutput.model_validate(item)
        outputs.append(output)

    return outputs


def validate_extract_resume_file(
    extract_outputs: list[LlmOutput],
    run_name: str,
) -> set[str]:
    """Validate extract resume data and return done IDs.

    Check for duplicates and log warnings.

    Returns:
        Set of accession IDs that have been processed.

    """
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for output in extract_outputs:
        if output.accession in seen_ids:
            duplicates.append(output.accession)
        seen_ids.add(output.accession)

    if duplicates:
        LOGGER.warning(
            "Found %d duplicate entries in extract resume file for run '%s': %s%s",
            len(duplicates),
            run_name,
            duplicates[:5],
            "..." if len(duplicates) > 5 else "",
        )

    return seen_ids


def dump_extract_resume_file(outputs: list[LlmOutput], run_name: str) -> Path:
    EXTRACT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    resume_file = EXTRACT_RESULT_DIR.joinpath(f"{run_name}_resume.json")

    fd, tmp_path = tempfile.mkstemp(dir=EXTRACT_RESULT_DIR, prefix=f"{run_name}_resume_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json_str = json.dumps([output.model_dump() for output in outputs], ensure_ascii=False, indent=2)
            f.write(json_str.encode("utf-8", errors="replace").decode("utf-8"))
        shutil.move(tmp_path, resume_file)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise

    return resume_file


def load_select_resume_file(run_name: str) -> list[SelectResult]:
    select_resume_file_path = SELECT_RESULT_DIR.joinpath(f"select_{run_name}_resume.json")
    if not select_resume_file_path.exists():
        return []
    with select_resume_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return []
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError(f"Resume file {select_resume_file_path} must contain a list of SelectResult objects.")
    results: list[SelectResult] = []
    for item in data:
        result = SelectResult.model_validate(item)
        results.append(result)

    return results


def dump_select_resume_file(results: list[SelectResult], run_name: str) -> Path:
    SELECT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    resume_file = SELECT_RESULT_DIR.joinpath(f"select_{run_name}_resume.json")

    fd, tmp_path = tempfile.mkstemp(dir=SELECT_RESULT_DIR, prefix=f"select_{run_name}_resume_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json_str = json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2)
            f.write(json_str.encode("utf-8", errors="replace").decode("utf-8"))
        shutil.move(tmp_path, resume_file)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise

    return resume_file


def remove_resume_files(run_name: str) -> None:
    extract_resume_file_path = EXTRACT_RESULT_DIR.joinpath(f"{run_name}_resume.json")
    if extract_resume_file_path.exists():
        extract_resume_file_path.unlink()

    select_resume_file_path = SELECT_RESULT_DIR.joinpath(f"select_{run_name}_resume.json")
    if select_resume_file_path.exists():
        select_resume_file_path.unlink()


def validate_resume_consistency(
    extract_outputs: list[LlmOutput],
    select_results: list[SelectResult],
    run_name: str,
) -> tuple[set[str], set[str]]:
    """Validate consistency between extract and select resume data.

    Detect orphaned entries (extract completed but select not completed)
    and invalid entries (select exists but extract missing).

    Returns:
        Tuple of (done_ids, orphan_ids):
        - done_ids: IDs that have both extract and select completed
        - orphan_ids: IDs that have extract but not select (need reprocessing)

    Raises:
        ResumeDataError: If select has entries that extract doesn't have

    """
    from bsllmner2.errors import ResumeDataError

    extract_ids = {output.accession for output in extract_outputs}
    select_ids = {result.accession for result in select_results}

    # IDs in select but not in extract = data corruption
    invalid_ids = select_ids - extract_ids
    if invalid_ids:
        raise ResumeDataError(
            run_name,
            f"Select resume contains {len(invalid_ids)} entries not found in extract resume: "
            f"{sorted(invalid_ids)[:5]}{'...' if len(invalid_ids) > 5 else ''}",
        )

    # IDs in both = completed
    done_ids = extract_ids & select_ids

    # IDs in extract but not in select = orphans (need reprocessing)
    orphan_ids = extract_ids - select_ids

    return done_ids, orphan_ids
