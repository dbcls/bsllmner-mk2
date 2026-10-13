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


_FILTER_KEYS = _load_filter_keys(FILTER_KEYS_PATH)
_FILTER_KEY_SET = frozenset(_FILTER_KEYS.filter_keys)


def is_ebi_format(bs_entry: dict[str, Any]) -> bool:
    """Check if the BioSample entry is in EBI format.

    EBI format is identified by the presence of a 'characteristics' key that is a dictionary.
    """
    return "characteristics" in bs_entry and isinstance(bs_entry["characteristics"], dict)


def _ncbi_record(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return the NCBI-format record of the entry, or None if the entry is not in NCBI format.

    The record sits under a 'BioSample' key in converter-derived entries and at the top level
    in Bulk API-derived entries. A top-level 'BioSample' object identifies the format on its own;
    otherwise the entry must carry 'Attributes' or 'Description'.
    """
    wrapped = entry.get("BioSample")
    if isinstance(wrapped, dict):
        return wrapped
    if isinstance(entry.get("Attributes"), dict) or isinstance(entry.get("Description"), dict):
        return entry
    return None


def _add_value(target: dict[str, Any], key: str, value: str) -> None:
    """Add one attribute value, keeping every distinct value of a repeated attribute name."""
    if key in _FILTER_KEY_SET:
        return
    if key not in target:
        target[key] = value
        return
    current = target[key]
    if isinstance(current, list):
        if value not in current:
            current.append(value)
    elif current != value:
        target[key] = [current, value]


def _construct_from_ebi(entry: dict[str, Any]) -> dict[str, Any]:
    filtered_entry: dict[str, Any] = {}
    for key, value in entry.get("characteristics", {}).items():
        if key in _FILTER_KEY_SET:
            continue
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "text" in value[0]:
            filtered_entry[key] = value[0]["text"]

    return filtered_entry


def _construct_from_ncbi(record: dict[str, Any]) -> dict[str, Any]:
    filtered_entry: dict[str, Any] = {}

    attributes = record.get("Attributes")
    if isinstance(attributes, dict):
        attribute = attributes.get("Attribute")
        items = attribute if isinstance(attribute, list) else [attribute]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("attribute_name")
            content = item.get("content")
            if isinstance(name, str) and name != "" and isinstance(content, str) and content != "":
                _add_value(filtered_entry, name, content)

    description = record.get("Description")
    if isinstance(description, dict):
        title = description.get("Title")
        if isinstance(title, str) and title != "":
            _add_value(filtered_entry, "title", title)

        organism = description.get("Organism")
        if isinstance(organism, dict):
            for key in ("OrganismName", "taxonomy_name"):
                organism_name = organism.get(key)
                if isinstance(organism_name, str) and organism_name != "":
                    _add_value(filtered_entry, "organism", organism_name)
                    break

        comment = description.get("Comment")
        if isinstance(comment, dict):
            paragraph = comment.get("Paragraph")
            if isinstance(paragraph, str) and paragraph != "":
                _add_value(filtered_entry, "description", paragraph)

    return filtered_entry


def construct_llm_input_json(entry: dict[str, Any]) -> dict[str, Any]:
    """Construct minimized input JSON for LLM calls from a BioSample JSON object.

    Attributes are flattened into a single object that maps an attribute name to its value, and
    keys listed in filter_keys.json are dropped. Record-level keys of an NCBI-format entry
    (identifiers, dates, owner, submission package) are dropped as well. An entry that is in
    neither EBI nor NCBI format keeps its top-level keys.

    Args:
        entry: A single BioSample JSON object.

    Returns:
        A filtered dictionary containing only relevant keys for LLM processing.

    """
    if is_ebi_format(entry):
        return _construct_from_ebi(entry)

    record = _ncbi_record(entry)
    if record is not None:
        return _construct_from_ncbi(record)

    return {key: value for key, value in entry.items() if key not in _FILTER_KEY_SET}
