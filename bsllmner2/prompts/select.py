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
        "Choose the ontology term whose description best matches the provided sample context, "
        "or return null if none of the candidates is consistent with the context.\n"
        "\n"
        "How to decide:\n"
        "- Read each candidate's label, comments, and definitions as the ontology's authoritative description of what that term refers to.\n"
        "- Read the provided sample metadata as the authoritative description of the actual sample.\n"
        "- Pick the single candidate whose description is consistent with the sample's biological context (origin, lineage, condition, and other attributes recorded in the metadata).\n"
        "- If no candidate is consistent with the context, return null.\n"
        "\n"
        "Hard rules:\n"
        "- Choose only from the provided candidates; do not invent term IDs.\n"
        "- Do not rely on outside knowledge or term popularity; decide only from the provided context and candidate descriptions.\n"
        "- Do not pick a candidate just because its label string resembles the input value; the input value can be ambiguous and shared across multiple unrelated terms.\n"
        "- Output ONLY valid JSON matching the schema. No extra text.\n"
    )
    if reasoning:
        base += (
            "- Also return a 'reasoning' describing how you decided: "
            "quote the exact evidence from the provided sample metadata, "
            "compare it against the top candidates' descriptions, "
            "and state why the others were rejected. Do not use outside knowledge."
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
                    "(1) the exact evidence text from the sample metadata, "
                    "(2) a brief comparison of the top 2-3 candidates' descriptions against that evidence, "
                    "(3) explicit rejection reasons for the others."
                )

            user_msg = Message(
                role="user",
                content=(
                    f"Field: {field_name}\n"
                    f"Value: {value}\n\n"
                    f"Description: {(field_config.prompt_description or field_name)}\n\n"
                    "Provenance:\n"
                    "- 'Value' was produced by an earlier NER step and may be noisy or shared across multiple unrelated ontology terms.\n"
                    "- The 'ontology candidates' below were assembled by ontology search and text2term lookups, and are the ONLY allowed choices.\n"
                    "- Decide strictly from the provided sample metadata and the candidate descriptions; do not use outside knowledge.\n"
                    "- Do not pick a candidate just because its label or synonym resembles the value; pick the one whose description (label, comments, definitions) is consistent with the sample metadata.\n\n"
                    f"Original extracted value: {value}\n\n"
                    f"Sample metadata (context):\n{bs_ctx_json}\n\n"
                    "Ontology candidates (JSON array):\n"
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
