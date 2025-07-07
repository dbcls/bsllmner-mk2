import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bsllmner2.config import RESULT_DIR, Config
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

    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return {}

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
    args: Optional[CliExtractArgs] = None,
    metrics: Optional[List[Metrics]] = None,
) -> Result:
    return Result(
        input=WfInput(
            bs_entries=bs_entries,
            mapping=mapping,
            prompt=prompt,
            model=model,
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
