"""Select mode: ontology search, text2term mapping, and LLM-based selection."""

import asyncio
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, TypedDict

from ollama import ChatResponse, Message
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.benchmark import stage_timer
from bsllmner2.config import LOGGER
from bsllmner2.llm import LlmBackend, build_ollama_options, parse_response_json
from bsllmner2.models import (
    BsEntries,
    DiskIoTimings,
    ExtractEntry,
    OntologyIndex,
    ResolvedValue,
    SearchResult,
    SelectConfig,
    SelectEntry,
    llm_timing_from_chat_response,
)
from bsllmner2.ontology_search import (
    build_index_from_file,
    is_label_prop,
    search_terms,
    search_terms_with_text2term,
)

class SelectStageTimings(TypedDict):
    ontology_search_sec: float
    text2term_sec: float
    llm_select_sec: float


INDEX_CACHE_DIR = Path(os.environ.get("BSLLMNER2_INDEX_CACHE_DIR", "ontology/index_cache"))


def _resolved_from_search_result(
    value: str,
    search_result: SearchResult,
    reasoning: str | None = None,
) -> ResolvedValue:
    return ResolvedValue(
        value=value,
        term_id=search_result.term_id,
        term_uri=search_result.term_uri,
        label=search_result.label,
        exact_match=search_result.exact_match,
        reasoning=reasoning if reasoning is not None else search_result.reasoning,
    )


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


def build_index_map(select_config: SelectConfig) -> tuple[dict[Path, OntologyIndex], DiskIoTimings]:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mapping: dict[Path, OntologyIndex] = {}
    disk_io = DiskIoTimings()

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
                with stage_timer("cache_load") as t, cache_file_path.open("rb") as f:
                    index = pickle.load(f)
                disk_io.index_cache_load_sec.append(t.elapsed_sec)
                mapping[ontology_file_path] = index
                continue
            except (OSError, EOFError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
                LOGGER.warning("Failed to load cache %s", cache_file_path, exc_info=True)

        with stage_timer("index_build") as t:
            index = build_index_from_file(ontology_file_path, ontology_filter=field_config.ontology_filter)
        disk_io.index_build_from_file_sec.append(t.elapsed_sec)
        mapping[ontology_file_path] = index

        try:
            with stage_timer("cache_save") as t, cache_file_path.open("wb") as f:
                pickle.dump(index, f)
            disk_io.index_cache_save_sec.append(t.elapsed_sec)
        except OSError:
            LOGGER.warning("Failed to save cache %s", cache_file_path, exc_info=True)

    return mapping, disk_io


def _collect_queries(
    select_entries: list[SelectEntry],
    field_name: str,
) -> set[str]:
    """Collect unique query strings from select entries for a given field.

    Skips entries that already have final results for the field.
    """
    queries: set[str] = set()
    for entry in select_entries:
        extracted = entry.extract.extracted
        if not isinstance(extracted, dict):
            if extracted is not None:
                LOGGER.warning(
                    "Skipping non-dict extracted for accession %s in _collect_queries: got %s",
                    entry.extract.accession,
                    type(extracted).__name__,
                )
            continue
        if field_name not in extracted:
            continue
        if entry.results.get(field_name):
            continue
        query_value = extracted[field_name]
        if isinstance(query_value, str):
            queries.add(query_value)
        elif isinstance(query_value, list):
            queries.update(v for v in query_value if isinstance(v, str))

    return queries


def _distribute_results(
    select_entries: list[SelectEntry],
    field_name: str,
    all_results: dict[str, list[SearchResult]],
    result_attr: str,
) -> None:
    """Distribute search results back into SelectEntry objects.

    For each SelectEntry, stores per-query candidates into the attribute
    specified by *result_attr* (``"search_results"`` or ``"text2term_results"``)
    and sets exact-match results into ``results`` as ``list[ResolvedValue]``.
    """
    for entry in select_entries:
        extracted = entry.extract.extracted
        if not isinstance(extracted, dict):
            if extracted is not None:
                LOGGER.warning(
                    "Skipping non-dict extracted for accession %s in _distribute_results: got %s",
                    entry.extract.accession,
                    type(extracted).__name__,
                )
            continue
        if field_name not in extracted:
            continue
        if entry.results.get(field_name):
            continue

        query_value = extracted[field_name]
        if isinstance(query_value, str):
            values = [query_value]
        elif isinstance(query_value, list):
            values = [v for v in query_value if isinstance(v, str)]
        else:
            continue

        per_query_store: dict[str, Any] = getattr(entry, result_attr)
        field_specific = per_query_store.setdefault(field_name, {})

        existing_resolved = entry.results.get(field_name, [])

        for value in values:
            candidates = all_results.get(value, [])
            field_specific[value] = candidates

            exact_match_result = _pick_exact_match_search_result(candidates)
            if exact_match_result is not None:
                existing_resolved.append(_resolved_from_search_result(value, exact_match_result))

        if existing_resolved:
            entry.results[field_name] = existing_resolved


def _ontology_search_wrapper(
    select_entries: list[SelectEntry],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectEntry]:
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

        queries = _collect_queries(select_entries, field_name)
        if not queries:
            continue

        results = search_terms(index, queries)
        _distribute_results(select_entries, field_name, results, "search_results")

    return select_entries


def _text2term_wrapper(
    select_entries: list[SelectEntry],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectEntry]:
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

        queries = _collect_queries(select_entries, field_name)
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

        _distribute_results(select_entries, field_name, results, "text2term_results")

    return select_entries


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
    select_entry: SelectEntry,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    merged.extend(select_entry.search_results.get(field_name, {}).get(value, []))
    merged.extend(select_entry.text2term_results.get(field_name, {}).get(value, []))

    # Remove duplicates based on term_id.
    by_term_id: dict[str, SearchResult] = {}
    for result in merged:
        prev = by_term_id.get(result.term_id)
        if prev is None or (is_label_prop(result.prop_uri) and not is_label_prop(prev.prop_uri)):
            by_term_id[result.term_id] = result

    return list(by_term_id.values())


def _pick_search_result_by_id(
    select_entry: SelectEntry,
    field_name: str,
    value: str,
    term_id: str,
) -> SearchResult | None:
    candidates = _collect_candidates_for_field(field_name, value, select_entry)
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
        raw = extracted.get(field_name) if isinstance(extracted, dict) else None

        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, list):
            values = [v for v in raw if isinstance(v, str)]
        else:
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


def _parse_output_object(chat_response: ChatResponse) -> dict[str, Any] | None:
    """Parse a ChatResponse into a dict, or None if not a valid JSON object."""
    parsed = parse_response_json(chat_response)
    if isinstance(parsed, dict):
        return parsed

    return None


# === Select function ===


async def select(
    backend: LlmBackend,
    bs_entries: BsEntries,
    model: str,
    extract_outputs: list[ExtractEntry],
    select_config: SelectConfig,
    thinking: bool | None = None,
    include_reasoning: bool = True,
    index_map: dict[Path, OntologyIndex] | None = None,
    num_ctx: int | None = None,
) -> tuple[list[SelectEntry], list[ChatResponse], SelectStageTimings]:
    # Ensure model is available, pull if necessary
    await backend.ensure_model(model)

    fields = select_config.fields.keys()
    no_select_fields = [f for f in fields if select_config.fields[f].ontology_file is None]

    intermediate_entries: list[SelectEntry] = []
    for obj in extract_outputs:
        se = SelectEntry(
            extract=obj,
            search_results={field: {} for field in fields},
            text2term_results={field: {} for field in fields},
            select_timings={field: {} for field in fields},
            results={},
        )

        # Step 6: Boundary validation — non-dict, non-None extracted gets warning + None
        extract = obj
        if extract.extracted is not None and not isinstance(extract.extracted, dict):
            LOGGER.warning(
                "Non-dict extracted for accession %s: got %s, treating as None",
                extract.accession,
                type(extract.extracted).__name__,
            )
            extract = extract.model_copy(update={"extracted": None})
            se = se.model_copy(update={"extract": extract})

        for field in no_select_fields:
            if isinstance(extract.extracted, dict):
                raw_val = extract.extracted.get(field)
                if raw_val is not None:
                    if isinstance(raw_val, str):
                        se.results[field] = [ResolvedValue(value=raw_val)]
                    elif isinstance(raw_val, list):
                        se.results[field] = [ResolvedValue(value=v) for v in raw_val if isinstance(v, str)]

        # Step 7: Explicitly set empty list for fields with None value in extracted
        if isinstance(extract.extracted, dict):
            for field in fields:
                if field not in se.results and extract.extracted.get(field) is None and field in extract.extracted:
                    se.results[field] = []

        intermediate_entries.append(se)

    # 1. Perform ontology search for each field specified in the select configuration.
    with stage_timer("ontology_search") as t_ontology:
        _ontology_search_wrapper(intermediate_entries, select_config, index_map=index_map)

    # 2. Perform text2term search for each field specified in the select configuration.
    with stage_timer("text2term") as t_text2term:
        _text2term_wrapper(intermediate_entries, select_config, index_map=index_map)

    # 3. For fields that still have multiple matches or no matches, use the LLM to select the best match.

    all_select_chat_responses: list[ChatResponse] = []
    ollama_options = build_ollama_options(num_ctx)

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
                options=ollama_options,
                think=thinking,
                format_=schema,
            )
        except Exception as e:
            LOGGER.error("Error during select step: %s", e)
            response = None

        return (accession, field_name, value, response)

    coros = []
    bs_entry_map = {e.get("accession"): e for e in bs_entries if e.get("accession") is not None}
    for select_entry in intermediate_entries:
        accession = select_entry.extract.accession
        bs_entry = bs_entry_map.get(accession)
        if bs_entry is None:
            continue
        field_prompts_and_schemas = _build_select_prompt_and_schema(
            bs_entry,
            select_entry,
            select_config,
            include_reasoning,
        )
        for (field_name, value), (messages, schema) in field_prompts_and_schemas.items():
            coros.append(_process_field_selection(accession, field_name, value, messages, schema))

    with stage_timer("llm_select") as t_llm_select:
        if coros:
            LOGGER.info(
                "Performing LLM selection for %d fields across %d entries...",
                len(coros),
                len(intermediate_entries),
            )
            acc_to_entry_map = {e.extract.accession: e for e in intermediate_entries}
            llm_results = await asyncio.gather(*coros)

            for accession, field_name, value, chat_response in llm_results:
                select_entry = acc_to_entry_map[accession]
                if select_entry is None or chat_response is None:
                    continue

                all_select_chat_responses.append(chat_response)
                select_entry.select_timings.setdefault(field_name, {})[value] = llm_timing_from_chat_response(
                    chat_response
                )

                output_obj = _parse_output_object(chat_response)
                chosen_id = output_obj.get("id", None) if output_obj else None
                reasoning = output_obj.get("reasoning", None) if output_obj else None

                if not isinstance(chosen_id, str):
                    continue

                picked_result = _pick_search_result_by_id(select_entry, field_name, value, chosen_id.strip())
                if picked_result is None:
                    continue

                resolved = _resolved_from_search_result(
                    value,
                    picked_result,
                    reasoning=reasoning if isinstance(reasoning, str) else None,
                )

                existing_list = select_entry.results.setdefault(field_name, [])
                existing_list.append(resolved)

    timings = SelectStageTimings(
        ontology_search_sec=t_ontology.elapsed_sec,
        text2term_sec=t_text2term.elapsed_sec,
        llm_select_sec=t_llm_select.elapsed_sec,
    )

    return intermediate_entries, all_select_chat_responses, timings
