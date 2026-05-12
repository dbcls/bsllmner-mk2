"""Prompt and JSON-schema construction for Stage 3 LLM selection.

Split out of :mod:`bsllmner2.select` so the orchestration module stays focused
on the asyncio coordination. These functions are pure (no I/O, no mutation of
the input ``SelectEntry``) and therefore are exercised directly in unit tests.
"""

import json
from typing import Any

from ollama import Message
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.models import SearchResult, SelectConfig, SelectEntry
from bsllmner2.ontology_search import is_label_prop


def _string_values(value: Any) -> list[str]:
    """Coerce a string or list-of-strings field value into a plain list of strings.

    Duplicated from ``bsllmner2.select`` so this module remains free of inbound
    imports from orchestration code (which would create a cycle).
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _build_select_schema(
    candidates: list[SearchResult],
    reasoning: bool = True,
) -> JsonSchemaValue:
    enum = [res.term_id for res in candidates]

    properties: dict[str, Any] = {
        "id": {
            "anyOf": [
                {"type": "string", "enum": enum},
                {"type": "null"},
            ],
        },
    }
    required = ["id"]

    if reasoning:
        properties["reasoning"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        }
        required.append("reasoning")

    schema: JsonSchemaValue = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    return schema


def _serialize_candidates_for_llm(candidates: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        c.model_dump(exclude={"exact_match", "text2term_score", "reasoning"}, exclude_none=True) for c in candidates
    ]


def _collect_candidates_for_field(
    field_name: str,
    value: str,
    select_entry: SelectEntry,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    merged.extend(select_entry.search_results.get(field_name, {}).get(value, []))
    merged.extend(select_entry.text2term_results.get(field_name, {}).get(value, []))

    # Remove duplicates based on term_id. Prefer the candidate whose prop_uri is
    # a label property (rdfs:label / skos:prefLabel) when multiple hit the same term.
    by_term_id: dict[str, SearchResult] = {}
    for result in merged:
        prev = by_term_id.get(result.term_id)
        if prev is None or (is_label_prop(result.prop_uri) and not is_label_prop(prev.prop_uri)):
            by_term_id[result.term_id] = result

    return list(by_term_id.values())


def _build_select_system_message(reasoning: bool) -> Message:
    base = (
        "You are a smart curator of biological metadata.\n"
        "Pick the best ontology term ID from the provided candidates, or return null if uncertain.\n"
        "Rules:\n"
        "- Prefer exact string matches or canonical labels present in the metadata.\n"
        "- Prefer widely recognized and specific terms.\n"
        "- Do NOT invent IDs. Choose only from the provided candidates.\n"
        "- Do NOT use outside knowledge; decide only from the provided context.\n"
        "- Output ONLY valid JSON matching the schema. No extra text.\n"
    )
    if reasoning:
        base += (
            "- Also return a 'reasoning' that describes your decision process step by step: "
            "cite the exact evidence from the provided text, compare the top candidates, "
            "and state why others were rejected do not use outside knowledge."
        )

    return Message(role="system", content=base)


def _build_select_prompt_and_schema(
    bs_entry: dict[str, Any],
    select_entry: SelectEntry,
    select_config: SelectConfig,
    reasoning: bool,
) -> dict[tuple[str, str], tuple[list[Message], JsonSchemaValue]]:
    """Build per-field (messages, schema) for LLM selection (choose term_id).

    Only includes fields that still need a selection.
    """
    results: dict[tuple[str, str], tuple[list[Message], JsonSchemaValue]] = {}
    bs_ctx_json = json.dumps(bs_entry, ensure_ascii=False)
    system_msg = _build_select_system_message(reasoning)

    extracted = select_entry.extract.extracted

    for field_name, field_config in select_config.fields.items():
        if extracted is None:
            continue
        values = _string_values(extracted.get(field_name))
        if not values:
            continue

        for value in values:
            existing = select_entry.results.get(field_name)
            if isinstance(existing, list) and any(rv.value == value for rv in existing):
                continue

            candidates = _collect_candidates_for_field(field_name, value, select_entry)
            if not candidates:
                continue

            schema = _build_select_schema(candidates, reasoning=reasoning)

            reasoning_instr = ""
            if reasoning:
                reasoning_instr = (
                    "For 'reasoning', provide: "
                    "(1) exact evidence text, "
                    "(2) a brief comparison of the top 2-3 candidates, "
                    "(3) explicit rejection reasons for the others."
                )

            user_msg = Message(
                role="user",
                content=(
                    f"Field: {field_name}\n"
                    f"Value: {value}\n\n"
                    f"Description: {(field_config.prompt_description or field_name)}\n\n"
                    "Provenance:\n"
                    "- The 'value' below was produced by an earlier NER step and may be noisy.\n"
                    "- The 'ontology candidates' were assembled by ontology search (and possibly text2term) and are the ONLY allowed choices.\n"
                    "- Decide strictly from the provided metadata and candidates; do not use outside knowledge.\n\n"
                    f"Original extracted value: {value}\n\n"
                    f"BioSample metadata (context):\n{bs_ctx_json}\n\n"
                    f"Ontology candidates (JSON array):\n"
                    f"{json.dumps(_serialize_candidates_for_llm(candidates), ensure_ascii=False, indent=2)}\n\n"
                    "Return ONLY JSON that matches the schema.\n"
                    f"{reasoning_instr}"
                ),
            )

            results[(field_name, value)] = (
                [system_msg, user_msg],
                schema,
            )

    return results
