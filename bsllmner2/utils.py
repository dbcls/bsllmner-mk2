import json
from pathlib import Path
from typing import Any, Dict, List


def load_bs_entries(path: Path) -> List[Dict[str, Any]]:
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
