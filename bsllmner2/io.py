import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import ijson
import yaml
from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import EXTRACT_RESULT_DIR, LOGGER, PROGRESS_DIR, SELECT_RESULT_DIR
from bsllmner2.errors import ResumeDataError
from bsllmner2.models import (
    BsEntries,
    ExtractEntry,
    ExtractResult,
    Mapping,
    MappingValue,
    Prompt,
    RunMetadata,
    SelectConfig,
    SelectEntry,
    SelectResult,
)

_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _replace_surrogates(s: str) -> str:
    """Replace lone surrogate characters with U+FFFD replacement character."""
    return _SURROGATE_RE.sub("\ufffd", s)


def load_bs_entries(path: Path) -> BsEntries:
    """Load and return a list of BioSample entries from a JSON or JSONL file.

    If the file is JSONL, each line is treated as a separate JSON object.
    If the file is JSON, it is expected to be a list of dictionaries.

    Raises:
        ValueError: If the file is neither JSON nor JSONL.

    """
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    with path.open("r", encoding="utf-8", errors="replace") as f:
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

    with path.open("r", encoding="utf-8", errors="replace") as f:
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


def load_select_config(path: Path) -> SelectConfig:
    if not path.exists():
        raise FileNotFoundError(f"Select configuration file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in select configuration file {path}: {e}") from e

    return SelectConfig.model_validate(data)


def dump_extract_result(result: ExtractResult, run_name: str) -> Path:
    EXTRACT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = EXTRACT_RESULT_DIR.joinpath(f"{run_name}.json")
    with result_file.open("w", encoding="utf-8") as f:
        json_str = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        f.write(_replace_surrogates(json_str))

    return result_file


def dump_select_result(result: SelectResult, run_name: str) -> Path:
    SELECT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = SELECT_RESULT_DIR.joinpath(f"select_{run_name}.json")
    with result_file.open("w", encoding="utf-8") as f:
        json_str = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        f.write(_replace_surrogates(json_str))

    return result_file


def load_extract_result(path: Path) -> ExtractResult:
    if not path.exists():
        raise FileNotFoundError(f"Result file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    return ExtractResult.model_validate_json(content)


def load_select_result(path: Path) -> SelectResult:
    if not path.exists():
        raise FileNotFoundError(f"Result file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    return SelectResult.model_validate_json(content)


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
            run_metadata = load_run_metadata(file)
            run_metadata_list.append(run_metadata)
        except (FileNotFoundError, ValueError) as e:
            LOGGER.warning("Skipping file %s: %s", file, e)

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


_EXTRACT_ENTRY_LIST_ADAPTER = TypeAdapter(list[ExtractEntry])


def load_extract_resume_file(run_name: str) -> list[ExtractEntry]:
    extract_resume_file_path = EXTRACT_RESULT_DIR.joinpath(f"{run_name}_resume.json")
    if not extract_resume_file_path.exists():
        return []

    with extract_resume_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return []

    return _EXTRACT_ENTRY_LIST_ADAPTER.validate_json(content)


def validate_extract_resume_file(
    extract_entries: list[ExtractEntry],
    run_name: str,
) -> set[str]:
    """Validate extract resume data and return done IDs.

    Check for duplicates and log warnings.

    Returns:
        Set of accession IDs that have been processed.

    """
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for entry in extract_entries:
        if entry.accession in seen_ids:
            duplicates.append(entry.accession)
        seen_ids.add(entry.accession)

    if duplicates:
        LOGGER.warning(
            "Found %d duplicate entries in extract resume file for run '%s': %s%s",
            len(duplicates),
            run_name,
            duplicates[:5],
            "..." if len(duplicates) > 5 else "",
        )

    return seen_ids


def _atomic_dump_json(
    data: Sequence[BaseModel],
    result_dir: Path,
    filename: str,
    prefix: str,
) -> Path:
    """Atomically write a list of Pydantic models as JSON to a file."""
    result_dir.mkdir(parents=True, exist_ok=True)
    resume_file = result_dir.joinpath(filename)

    fd, tmp_path = tempfile.mkstemp(dir=result_dir, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json_str = json.dumps([item.model_dump(mode="json") for item in data], ensure_ascii=False, indent=2)
            f.write(_replace_surrogates(json_str))
        shutil.move(tmp_path, resume_file)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise

    return resume_file


def dump_extract_resume_file(outputs: list[ExtractEntry], run_name: str) -> Path:
    return _atomic_dump_json(outputs, EXTRACT_RESULT_DIR, f"{run_name}_resume.json", f"{run_name}_resume_")


_SELECT_ENTRY_LIST_ADAPTER = TypeAdapter(list[SelectEntry])


def load_select_resume_file(run_name: str) -> list[SelectEntry]:
    select_resume_file_path = SELECT_RESULT_DIR.joinpath(f"select_{run_name}_resume.json")
    if not select_resume_file_path.exists():
        return []
    with select_resume_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return []

    return _SELECT_ENTRY_LIST_ADAPTER.validate_json(content)


def dump_select_resume_file(results: list[SelectEntry], run_name: str) -> Path:
    return _atomic_dump_json(results, SELECT_RESULT_DIR, f"select_{run_name}_resume.json", f"select_{run_name}_resume_")


def remove_resume_files(run_name: str) -> None:
    extract_resume_file_path = EXTRACT_RESULT_DIR.joinpath(f"{run_name}_resume.json")
    if extract_resume_file_path.exists():
        extract_resume_file_path.unlink()

    select_resume_file_path = SELECT_RESULT_DIR.joinpath(f"select_{run_name}_resume.json")
    if select_resume_file_path.exists():
        select_resume_file_path.unlink()


def validate_resume_consistency(
    extract_entries: list[ExtractEntry],
    select_entries: list[SelectEntry],
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
    extract_ids = {entry.accession for entry in extract_entries}
    select_ids = {entry.extract.accession for entry in select_entries}

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
