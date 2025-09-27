import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ijson
import yaml
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import PROGRESS_DIR, RESULT_DIR, Config
from bsllmner2.metrics import Metrics
from bsllmner2.schema import (BsEntries, CliExtractArgs, ErrorInfo, ErrorLog,
                              Evaluation, LlmOutput, Mapping, MappingValue,
                              Prompt, Result, RunMetadata, WfInput)


def load_bs_entries(path: Path) -> BsEntries:
    """
    Load and return a list of BioSample entries from a JSON or JSONL file.
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
            else:
                raise ValueError("JSON file must contain a list of dictionaries.")
        except json.JSONDecodeError as outer_e:
            # If JSON fails, try to load as JSONL
            f.seek(0)
            jl_data: List[Dict[str, Any]] = []
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as inner_e:
                    raise ValueError(f"Invalid JSONL: failed to parse line {line!r}") from inner_e
                if not isinstance(entry, dict):
                    raise ValueError("Each line in JSONL file must be a JSON object.") from outer_e
                jl_data.append(entry)
            if not jl_data:
                raise ValueError("JSONL file contains no valid JSON objects.") from outer_e
            return jl_data


def load_prompt_file(path: Path) -> List[Prompt]:
    """
    Load a prompt file from the given path.
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
    HEADERS = ["BioSample ID", "Experiment type", "extraction answer", "mapping answer ID", "mapping answer label"]

    mapping: Mapping = {}

    if path.is_file():
        with path.open("rt", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
            if not lines:
                return mapping
    else:
        return mapping

    header_fields = lines[0].split("\t")
    if header_fields != HEADERS:
        raise ValueError(f"Header mismatch: expected {HEADERS}, got {header_fields}")

    for lineno, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(HEADERS):
            raise ValueError(f"The number of fields in line {lineno} does not match the header: {len(fields)} != {len(HEADERS)}")
        accession, experiment_type, extraction_answer, mapping_answer_id, mapping_answer_label = fields
        if accession in mapping:
            raise ValueError(f"Duplicate accession found: {accession} at line {lineno}")

        mapping[accession] = MappingValue(
            experiment_type=experiment_type,
            extraction_answer=extraction_answer if extraction_answer else None,
            mapping_answer_id=mapping_answer_id if mapping_answer_id else None,
            mapping_answer_label=mapping_answer_label if mapping_answer_label else None,
        )

    return mapping


def load_format_schema(path: Path) -> JsonSchemaValue:
    """
    Load a JSON schema file from the given path.
    The file should contain a valid JSON schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Format schema file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        try:
            schema = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema in file {path}: {e}") from e

    return schema  # type: ignore


def evaluate_output(output: List[LlmOutput], mapping: Mapping) -> List[Evaluation]:
    """
    Evaluate the LLM outputs against the mapping.
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
    mapping: Mapping,
    prompt: List[Prompt],
    model: str,
    output: List[LlmOutput],
    evaluation: List[Evaluation],
    config: Config,
    run_metadata: RunMetadata,
    format_: Optional[JsonSchemaValue] = None,
    thinking: Optional[bool] = None,
    args: Optional[CliExtractArgs] = None,
    metrics: Optional[List[Metrics]] = None,
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
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def dump_result(result: Result, run_name: Optional[str] = None) -> Path:
    if run_name is None:
        run_name = f"extract_{result.input.cli_args.model}_{get_now_str()}"  # type: ignore

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULT_DIR.joinpath(f"{run_name}.json")
    with result_file.open("w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return result_file


def load_result(path: Path) -> Result:
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


def list_run_metadata() -> List[RunMetadata]:
    if not RESULT_DIR.exists():
        return []

    run_metadata_list = []
    for file in RESULT_DIR.glob("*.json"):
        try:
            run_name = file.name.removesuffix(".json")
            run_metadata = load_run_metadata(file)
            run_metadata.completed_count = load_progress_count(run_name)
            run_metadata_list.append(run_metadata)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping file {file}: {e}")

    return run_metadata_list


def list_run_names() -> List[str]:
    """
    List all run names from the result directory.
    Returns:
        A list of run names (without file extensions).
    """
    if not RESULT_DIR.exists():
        return []

    return [file.name.removesuffix(".json") for file in RESULT_DIR.glob("*.json") if file.is_file()]


def load_progress_count(run_name: str) -> Optional[int]:
    """
    Load the progress count from a file in the PROGRESS_DIR.
    The file is named after the run_name and contains one accession per line.
    Returns:
        The number of processed entries.
    """
    progress_file = PROGRESS_DIR.joinpath(f"{run_name}.txt")
    if not progress_file.exists():
        return None

    with progress_file.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)
