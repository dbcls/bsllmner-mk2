import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from bsllmner2.config import FILTER_KEY_VAL_RULES_PATH


class FilterKeyValRules(BaseModel):
    filter_keys: List[str] = []
    filter_values: List[str] = []


def _load_filter_key_val_rules(path: Path) -> FilterKeyValRules:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return FilterKeyValRules(**data)


def is_ebi_format(bs_entry: Dict[str, Any]) -> bool:
    """
    Check if the BioSample entry is in EBI format.
    EBI format is identified by the presence of a 'characteristics' key that is a dictionary.
    """
    return "characteristics" in bs_entry and isinstance(bs_entry["characteristics"], dict)


def construct_llm_input_json(bs_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Construct minimized input JSON for LLM calls from a list of BioSample JSON objects.
    This function filters out keys that are not relevant for LLM processing.

    Args:
        bs_json (List[Dict[str, Any]]): List of BioSample JSON objects.

    Returns:
        List[Dict[str, Any]]: Filtered list of BioSample JSON objects ready for LLM input.
    """
    filter_key_val_rules = _load_filter_key_val_rules(FILTER_KEY_VAL_RULES_PATH)

    is_input_ebi_format = False
    if "characteristics" in bs_json[0] and isinstance(bs_json[0]["characteristics"], dict):
        is_input_ebi_format = True

    filtered_entries = []
    for entry in bs_json:
        if is_ebi_format(entry):
            attrs = entry.get("characteristics", {})
        else:
            attrs = entry
        filtered_entry = {}
        for key, value in attrs.items():
            if key not in filter_key_val_rules.filter_keys:
                filtered_entry[key] = value
        filtered_entries.append(filtered_entry)

    return filtered_entries
