import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bsllmner2.config import FILTER_KEYS_PATH


class FilterKeys(BaseModel):
    filter_keys: list[str] = []


def _load_filter_keys(path: Path) -> FilterKeys:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return FilterKeys(**data)


def is_ebi_format(bs_entry: dict[str, Any]) -> bool:
    """Check if the BioSample entry is in EBI format.

    EBI format is identified by the presence of a 'characteristics' key that is a dictionary.
    """
    return "characteristics" in bs_entry and isinstance(bs_entry["characteristics"], dict)


def construct_llm_input_json(entry: dict[str, Any]) -> dict[str, Any]:
    """Construct minimized input JSON for LLM calls from a BioSample JSON object.

    Filter out keys that are not relevant for LLM processing.

    Args:
        entry: A single BioSample JSON object.

    Returns:
        A filtered dictionary containing only relevant keys for LLM processing.

    """
    filter_keys = _load_filter_keys(FILTER_KEYS_PATH)

    filtered_entry = {}
    if is_ebi_format(entry):
        attrs = entry.get("characteristics", {})
    else:
        attrs = entry
    for key, value in attrs.items():
        if key not in filter_keys.filter_keys:
            if is_ebi_format(entry):
                if (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], dict)
                    and "text" in value[0]
                ):
                    filtered_entry[key] = value[0]["text"]
            else:
                filtered_entry[key] = value

    return filtered_entry
