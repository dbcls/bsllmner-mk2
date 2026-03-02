"""Select mode: ontology search, text2term mapping, and LLM-based selection."""

import asyncio
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import ollama
from ollama import ChatResponse, Message
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import LOGGER
from bsllmner2.llm import OLLAMA_OPTIONS, LlmBackend, _parse_response_json
from bsllmner2.models import BsEntries, LlmOutput, OntologyIndex, SearchResult, SelectConfig, SelectResult
from bsllmner2.ontology_search import (
    build_index_from_file,
    is_label_prop,
    search_terms,
    search_terms_with_text2term,
)

INDEX_CACHE_DIR = Path(os.environ.get("BSLLMNER2_INDEX_CACHE_DIR", "/app/ontology/index_cache"))


def _pick_exact_match_search_result(
    search_results: list[SearchResult],
) -> SearchResult | None:
    exact_matches = [res for res in search_results if res.exact_match]
    if not exact_matches:
        return None
    if len(exact_matches) == 1:
        return exact_matches[0]

    term_ids = {search_result.term_id for search_result in exact_matches}
    if len(term_ids) > 1:
        return None

    for search_result in exact_matches:
        if is_label_prop(search_result.prop_uri):
            return search_result

    # Here, only one unique term_id exists, but no preferred property found.
    # Return the first exact match as a fallback.
    return exact_matches[0]


def _compute_filter_hash(ontology_filter: dict[str, str] | None) -> str:
    """Compute a hash of the ontology filter for cache key."""
    if ontology_filter is None:
        return "nofilter"
    filter_str = json.dumps(ontology_filter, sort_keys=True)

    return hashlib.sha256(filter_str.encode()).hexdigest()[:16]


def build_index_map(select_config: SelectConfig) -> dict[Path, OntologyIndex]:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mapping: dict[Path, OntologyIndex] = {}

    for field_config in select_config.fields.values():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue
        if ontology_file_path in mapping:
            continue

        filter_hash = _compute_filter_hash(field_config.ontology_filter)
        cache_file_path = INDEX_CACHE_DIR.joinpath(f"{ontology_file_path.name}_{filter_hash}.pkl")
        if cache_file_path.exists():
            try:
                with cache_file_path.open("rb") as f:
                    index = pickle.load(f)
                mapping[ontology_file_path] = index
                continue
            except (OSError, EOFError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
                LOGGER.warning("Failed to load cache %s", cache_file_path, exc_info=True)

        index = build_index_from_file(ontology_file_path, ontology_filter=field_config.ontology_filter)
        mapping[ontology_file_path] = index

        try:
            with cache_file_path.open("wb") as f:
                pickle.dump(index, f)
        except OSError:
            LOGGER.warning("Failed to save cache %s", cache_file_path, exc_info=True)

    return mapping


def _collect_queries(
    select_results: list[SelectResult],
    field_name: str,
) -> set[str]:
    """Collect unique query strings from select results for a given field.

    Skips entries that already have final results for the field.
    """
    queries: set[str] = set()
    for res in select_results:
        if not isinstance(res.extract_output, dict):
            continue
        if field_name not in res.extract_output:
            continue
        if res.results.get(field_name):
            continue
        query_value = res.extract_output[field_name]
        if isinstance(query_value, str):
            queries.add(query_value)
        elif isinstance(query_value, list):
            queries.update(v for v in query_value if isinstance(v, str))

    return queries


def _distribute_results(
    select_results: list[SelectResult],
    field_name: str,
    all_results: dict[str, list[SearchResult]],
    result_attr: str,
) -> None:
    """Distribute search results back into SelectResult objects.

    For each SelectResult, stores per-query candidates into the attribute
    specified by *result_attr* (``"search_results"`` or ``"text2term_results"``)
    and sets exact-match results into ``results``.
    """
    for res in select_results:
        if not isinstance(res.extract_output, dict):
            continue
        if field_name not in res.extract_output:
            continue
        if res.results.get(field_name):
            continue

        query_value = res.extract_output[field_name]
        if isinstance(query_value, str):
            values = [query_value]
        elif isinstance(query_value, list):
            values = [v for v in query_value if isinstance(v, str)]
        else:
            continue

        per_query_store: dict[str, Any] = getattr(res, result_attr)
        field_specific = per_query_store.get(field_name)
        if not isinstance(field_specific, dict):
            field_specific = {}
            per_query_store[field_name] = field_specific

        field_results = res.results.get(field_name)
        if not isinstance(field_results, dict):
            field_results = {}
            res.results[field_name] = field_results

        for value in values:
            candidates = all_results.get(value, [])
            field_specific[value] = candidates

            exact_match_result = _pick_exact_match_search_result(candidates)
            if exact_match_result is not None:
                field_results[value] = exact_match_result


def _ontology_search_wrapper(
    select_results: list[SelectResult],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectResult]:
    """Perform ontology search for each field in the select configuration."""
    for field_name, field_config in select_config.fields.items():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue

        if index_map is not None:
            index = index_map.get(ontology_file_path)
            if index is None:
                continue
        else:
            index = build_index_from_file(ontology_file_path, ontology_filter=field_config.ontology_filter)

        LOGGER.info("Searching ontology for field: %s", field_name)

        queries = _collect_queries(select_results, field_name)
        if not queries:
            continue

        results = search_terms(index, queries)
        _distribute_results(select_results, field_name, results, "search_results")

    return select_results


def _text2term_wrapper(
    select_results: list[SelectResult],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectResult]:
    """Perform text2term search for each field in the select configuration."""
    for field_name, field_config in select_config.fields.items():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue

        if ontology_file_path.suffix != ".owl":
            LOGGER.warning(
                "Text2Term currently supports only OWL files. Skipping field: %s",
                field_name,
            )
            continue

        LOGGER.info("text2term for field: %s", field_name)

        queries = _collect_queries(select_results, field_name)
        if not queries:
            continue

        index = index_map.get(ontology_file_path) if index_map is not None else None
        try:
            results = search_terms_with_text2term(queries, ontology_file_path, index=index)
        except (OSError, ValueError, RuntimeError) as e:
            LOGGER.exception(
                "text2term failed. field: %s, error: %s",
                field_name,
                e,
            )
            results = {}

        _distribute_results(select_results, field_name, results, "text2term_results")

    return select_results


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
    return [c.model_dump(exclude={"exact_match", "text2term_score", "reasoning"}) for c in candidates]


def _collect_candidates_for_field(
    field_name: str,
    value: str,
    select_result: SelectResult,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    merged.extend(select_result.search_results.get(field_name, {}).get(value, []))
    merged.extend(select_result.text2term_results.get(field_name, {}).get(value, []))

    # Remove duplicates based on term_id.
    by_term_id: dict[str, SearchResult] = {}
    for result in merged:
        prev = by_term_id.get(result.term_id)
        if prev is None or (is_label_prop(result.prop_uri) and not is_label_prop(prev.prop_uri)):
            by_term_id[result.term_id] = result

    return list(by_term_id.values())


def _pick_search_result_by_id(
    select_result: SelectResult,
    field_name: str,
    value: str,
    term_id: str,
) -> SearchResult | None:
    candidates = _collect_candidates_for_field(field_name, value, select_result)
    for candidate in candidates:
        if candidate.term_id == term_id and is_label_prop(candidate.prop_uri):
            return candidate
    for candidate in candidates:
        if candidate.term_id == term_id:
            return candidate

    return None


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
    select_result: SelectResult,
    select_config: SelectConfig,
    reasoning: bool,
) -> dict[tuple[str, str], tuple[list[Message], JsonSchemaValue]]:
    """Build per-field (messages, schema) for LLM selection (choose term_id).

    Only includes fields that still need a selection (i.e., results[field] is None).
    """
    results: dict[tuple[str, str], tuple[list[Message], JsonSchemaValue]] = {}
    bs_ctx_json = json.dumps(bs_entry, ensure_ascii=False)
    system_msg = _build_select_system_message(reasoning)

    for field_name, field_config in select_config.fields.items():
        raw = select_result.extract_output.get(field_name) if isinstance(select_result.extract_output, dict) else None

        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, list):
            values = [v for v in raw if isinstance(v, str)]
        else:
            continue

        for value in values:
            existing = select_result.results.get(field_name)
            if isinstance(existing, dict) and value in existing:
                continue

            candidates = _collect_candidates_for_field(field_name, value, select_result)
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


def _parse_output_object(chat_response: ChatResponse) -> dict[str, Any] | None:
    """Parse a ChatResponse into a dict, or None if not a valid JSON object."""
    parsed = _parse_response_json(chat_response)
    if isinstance(parsed, dict):
        return parsed

    return None


# === Select function ===


async def select(
    backend: LlmBackend,
    bs_entries: BsEntries,
    model: str,
    extract_outputs: list[LlmOutput],
    select_config: SelectConfig,
    thinking: bool | None = None,
    include_reasoning: bool = True,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectResult]:
    # Ensure model is available, pull if necessary
    await backend.ensure_model(model)

    fields = select_config.fields.keys()
    no_select_fields = [f for f in fields if select_config.fields[f].ontology_file is None]

    intermediate_results: list[SelectResult] = []
    for obj in extract_outputs:
        sr = SelectResult(
            accession=obj.accession,
            extract_output=obj.output,
            search_results={field: {} for field in fields},
            text2term_results={field: {} for field in fields},
            llm_chat_response={field: {} for field in fields},
            results={},
        )
        for field in no_select_fields:
            if isinstance(obj.output, dict):
                sr.results[field] = obj.output.get(field, None)
        intermediate_results.append(sr)

    # 1. Perform ontology search for each field specified in the select configuration.
    _ontology_search_wrapper(intermediate_results, select_config, index_map=index_map)

    # 2. Perform text2term search for each field specified in the select configuration.
    _text2term_wrapper(intermediate_results, select_config, index_map=index_map)

    # 3. For fields that still have multiple matches or no matches, use the LLM to select the best match.

    async def _process_field_selection(
        accession: str,
        field_name: str,
        value: str,
        messages: list[Message],
        schema: JsonSchemaValue,
    ) -> tuple[str, str, str, ChatResponse | None]:
        try:
            LOGGER.debug("[Select] Processing entry: %s, field: %s", accession, field_name)
            response: ChatResponse | None = await backend.chat(
                model=model,
                messages=messages,
                options=OLLAMA_OPTIONS,
                think=thinking,
                format_=schema,
            )
        except (ollama.ResponseError, OSError) as e:
            LOGGER.error("Error during select step: %s", e)
            response = None

        return (accession, field_name, value, response)

    coros = []
    bs_entry_map = {e.get("accession"): e for e in bs_entries if e.get("accession") is not None}
    for select_result in intermediate_results:
        accession = select_result.accession
        bs_entry = bs_entry_map.get(accession)
        if bs_entry is None:
            continue
        field_prompts_and_schemas = _build_select_prompt_and_schema(
            bs_entry,
            select_result,
            select_config,
            include_reasoning,
        )
        for (field_name, value), (messages, schema) in field_prompts_and_schemas.items():
            coros.append(_process_field_selection(accession, field_name, value, messages, schema))

    if coros:
        LOGGER.info(
            "Performing LLM selection for %d fields across %d entries...",
            len(coros),
            len(intermediate_results),
        )
        acc_to_result_map = {result.accession: result for result in intermediate_results}
        llm_results = await asyncio.gather(*coros)

        for accession, field_name, value, chat_response in llm_results:
            select_result = acc_to_result_map[accession]
            if select_result is None or chat_response is None:
                continue

            select_result.llm_chat_response.setdefault(field_name, {})[value] = chat_response

            output_obj = _parse_output_object(chat_response)
            chosen_id = output_obj.get("id", None) if output_obj else None
            reasoning = output_obj.get("reasoning", None) if output_obj else None

            if not isinstance(chosen_id, str):
                continue

            picked_result = _pick_search_result_by_id(select_result, field_name, value, chosen_id.strip())
            if picked_result is None:
                continue

            picked_copy = picked_result.model_copy(deep=True)
            if isinstance(reasoning, str):
                picked_copy.reasoning = reasoning

            field_dict = select_result.results.get(field_name)
            if not isinstance(field_dict, dict):
                field_dict = {}
                select_result.results[field_name] = field_dict
            field_dict[value] = picked_copy

    return intermediate_results
