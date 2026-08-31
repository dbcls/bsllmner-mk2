"""Select mode coordinator.

Stage 3 (LLM selection) lives here together with the helpers that distribute
ontology-search results across :class:`SelectEntry` instances. Pure prompt /
JSON-schema construction lives in :mod:`bsllmner2.prompts.select`, while
filesystem-bound cache I/O lives in :mod:`bsllmner2.select_index_cache`. The
public surface of those modules is re-exported below so existing tests that
patch attributes on ``bsllmner2.select`` continue to work unchanged.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from ollama import ChatResponse, Message
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.benchmark import stage_timer
from bsllmner2.config import LOGGER
from bsllmner2.errors import OllamaProcessingError
from bsllmner2.llm import LlmBackend, build_ollama_options, parse_response_json
from bsllmner2.models import (
    BsEntries,
    ErrorLog,
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
from bsllmner2.pipeline import build_error_log
from bsllmner2.prompts.select import (
    _build_select_prompt_and_schema,
    _build_select_schema,  # noqa: F401  re-exported for tests
    _build_select_system_message,  # noqa: F401  re-exported for tests
    _collect_candidates_for_field,
    _serialize_candidates_for_llm,  # noqa: F401  re-exported for tests
)
from bsllmner2.select_index_cache import (
    _CACHE_KEY_SUFFIX,  # noqa: F401  re-exported for tests
    INDEX_CACHE_DIR,
    TEXT2TERM_CACHE_DIR,
    _text2term_acronym,
    build_index_map,
    build_text2term_cache,
)


class SelectStageTimings(TypedDict):
    ontology_search_sec: float
    text2term_sec: float
    llm_select_sec: float


# Run-scoped memoization for ontology / text2term search. Key is (field_name,
# extracted value); reused across batches so identical queries are resolved
# once per run. Single-threaded asyncio makes lock-free sharing safe.
SearchMemo = dict[tuple[str, str], list[SearchResult]]


MatchKind = Literal["no_match", "single", "ambiguous"]


@dataclass(frozen=True, slots=True)
class MatchResolution:
    """Verdict on whether ontology-search candidates can be auto-picked.

    - ``no_match``: no candidates returned. Stage falls through to text2term / LLM.
    - ``single``: every candidate points to one ontology term and at least one
      hit is an exact synonym/label match. Auto-pick is safe — the term is
      uniquely identified by curator-validated ontology metadata.
    - ``ambiguous``: candidates span multiple terms, or all hits are non-exact
      n-gram subset matches. The LLM must disambiguate from context.
    """

    kind: MatchKind
    picked: SearchResult | None = None


__all__ = [
    "INDEX_CACHE_DIR",
    "TEXT2TERM_CACHE_DIR",
    "MatchKind",
    "MatchResolution",
    "SearchMemo",
    "SelectStageTimings",
    "build_index_map",
    "build_text2term_cache",
    "select",
]


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


def _resolve_search_candidates(
    candidates: list[SearchResult],
) -> MatchResolution:
    """Evaluate an ontology-search candidate list for a single (field, value).

    Auto-pick is only allowed when **every** candidate (exact and non-exact
    alike) resolves to the same ontology term *and* at least one hit is an
    exact synonym/label match. Any spread across multiple term_ids — or a
    candidate list made up only of non-exact n-gram subset hits — is treated
    as ambiguous so the LLM stage can disambiguate from the BioSample context.
    """
    if not candidates:
        return MatchResolution(kind="no_match")

    term_ids = {c.term_id for c in candidates}
    if len(term_ids) > 1:
        return MatchResolution(kind="ambiguous")

    exact_matches = [c for c in candidates if c.exact_match]
    if not exact_matches:
        # Only non-exact subset hits — too weak to commit without LLM review.
        return MatchResolution(kind="ambiguous")

    if len(exact_matches) == 1:
        return MatchResolution(kind="single", picked=exact_matches[0])

    for sr in exact_matches:
        if is_label_prop(sr.prop_uri):
            return MatchResolution(kind="single", picked=sr)
    return MatchResolution(kind="single", picked=exact_matches[0])


def _string_values(value: Any) -> list[str]:
    """Coerce a string or list-of-strings field value into a plain list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _resolved_values(entry: SelectEntry, field_name: str) -> set[str]:
    """Values of ``field_name`` that already carry a final term and need no further search.

    Array-typed fields hold one entry per extracted value, so resolution is tracked per
    value: a resolved ``drug`` value must not stop its siblings from being searched.
    """
    return {rv.value for rv in entry.results.get(field_name, [])}


def _collect_queries(
    select_entries: list[SelectEntry],
    field_name: str,
) -> set[str]:
    """Collect unique query strings from select entries for a given field.

    Skips values that are already resolved for the field.
    """
    queries: set[str] = set()
    for entry in select_entries:
        extracted = entry.extract.extracted
        if extracted is None or field_name not in extracted:
            continue
        resolved = _resolved_values(entry, field_name)
        queries.update(v for v in _string_values(extracted[field_name]) if v not in resolved)

    return queries


def _record_search_candidates(
    select_entries: list[SelectEntry],
    field_name: str,
    all_results: dict[str, list[SearchResult]],
) -> None:
    """Attach ontology-search candidates to SelectEntries and resolve where unambiguous.

    For each entry/value, the full candidate list is stored under
    ``search_results[field_name][value]``. The candidates are then evaluated:

    - ``MatchResolution.kind == "single"`` → append the picked term to
      ``results[field_name]`` so the LLM step skips it.
    - ``MatchResolution.kind == "ambiguous"`` → record the value under
      ``ambiguous_fields[field_name]`` so the LLM step picks it up.
    - ``MatchResolution.kind == "no_match"`` → leave both ``results`` and
      ``ambiguous_fields`` untouched; downstream text2term and LLM stages get
      a chance to fill in.

    Values already resolved for the field are skipped individually, so one resolved
    value in an array-typed field does not suppress its siblings.
    """
    for entry in select_entries:
        extracted = entry.extract.extracted
        if extracted is None or field_name not in extracted:
            continue

        values = _string_values(extracted[field_name])
        if not values:
            continue

        already_resolved = _resolved_values(entry, field_name)
        pending = [v for v in values if v not in already_resolved]
        if not pending:
            continue

        field_candidates = entry.search_results.setdefault(field_name, {})
        resolved = entry.results.get(field_name, [])
        ambiguous_for_field = entry.ambiguous_fields.setdefault(field_name, {})

        for value in pending:
            candidates = all_results.get(value, [])
            field_candidates[value] = candidates

            resolution = _resolve_search_candidates(candidates)
            if resolution.kind == "single" and resolution.picked is not None:
                resolved.append(_resolved_from_search_result(value, resolution.picked))
            elif resolution.kind == "ambiguous":
                ambiguous_for_field[value] = sorted({c.term_id for c in candidates})

        if resolved:
            entry.results[field_name] = resolved


def _record_text2term_candidates(
    select_entries: list[SelectEntry],
    field_name: str,
    all_results: dict[str, list[SearchResult]],
) -> None:
    """Attach text2term candidates to SelectEntries without auto-picking.

    text2term is a fuzzy mapper (TF-IDF / embeddings) that supplements the
    ontology-search candidate pool. Its hits feed the LLM stage; they never
    decide a term_id on their own — letting a fuzzy matcher commit would
    silently override ambiguity already detected by ontology search.

    Values already resolved by Stage 2a are skipped individually. Skipping the whole
    entry instead would strip fuzzy candidates from the unresolved values of an
    array-typed field — exactly the values that need them most.
    """
    for entry in select_entries:
        extracted = entry.extract.extracted
        if extracted is None or field_name not in extracted:
            continue

        values = _string_values(extracted[field_name])
        if not values:
            continue

        already_resolved = _resolved_values(entry, field_name)
        pending = [v for v in values if v not in already_resolved]
        if not pending:
            continue

        field_candidates = entry.text2term_results.setdefault(field_name, {})
        for value in pending:
            field_candidates[value] = all_results.get(value, [])


def _ontology_search_wrapper(
    select_entries: list[SelectEntry],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
    search_memo: SearchMemo | None = None,
) -> list[SelectEntry]:
    """Perform ontology search for each field in the select configuration.

    When ``search_memo`` is provided, previously seen ``(field, value)`` pairs
    reuse their cached candidates instead of re-scanning the ontology index.
    """
    memo = search_memo if search_memo is not None else {}
    for field_name, field_config in select_config.fields.items():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue

        if index_map is not None:
            index = index_map.get(ontology_file_path)
            if index is None:
                continue
        else:
            index = build_index_from_file(ontology_file_path)

        LOGGER.info("Searching ontology for field: %s", field_name)

        queries = _collect_queries(select_entries, field_name)
        if not queries:
            continue

        uncached = {q for q in queries if (field_name, q) not in memo}
        if uncached:
            new_results = search_terms(index, uncached)
            for q in uncached:
                memo[(field_name, q)] = new_results.get(q, [])

        results = {q: memo[(field_name, q)] for q in queries}
        _record_search_candidates(select_entries, field_name, results)

    return select_entries


def _text2term_wrapper(
    select_entries: list[SelectEntry],
    select_config: SelectConfig,
    index_map: dict[Path, OntologyIndex] | None = None,
    cache_folder: Path | None = None,
    t2t_memo: SearchMemo | None = None,
) -> list[SelectEntry]:
    """Perform text2term search for each field in the select configuration.

    When ``t2t_memo`` is provided, previously seen ``(field, value)`` pairs
    reuse their cached candidates instead of re-invoking text2term. Failed
    queries are memoized as empty lists so the same inputs do not repeatedly
    hammer text2term across batches sharing the memo.
    """
    memo = t2t_memo if t2t_memo is not None else {}
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
        acronym = _text2term_acronym(ontology_file_path) if cache_folder is not None else None

        uncached = {q for q in queries if (field_name, q) not in memo}
        if uncached:
            try:
                new_results = search_terms_with_text2term(
                    uncached,
                    ontology_file_path,
                    index=index,
                    acronym=acronym,
                    cache_folder=cache_folder,
                )
            except (OSError, ValueError, RuntimeError) as e:
                LOGGER.exception(
                    "text2term failed. field: %s, error: %s",
                    field_name,
                    e,
                )
                for q in uncached:
                    memo[(field_name, q)] = []
            else:
                for q in uncached:
                    memo[(field_name, q)] = new_results.get(q, [])

        results = {q: memo.get((field_name, q), []) for q in queries}
        _record_text2term_candidates(select_entries, field_name, results)

    return select_entries


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
    thinking: bool = False,
    include_reasoning: bool = True,
    index_map: dict[Path, OntologyIndex] | None = None,
    text2term_cache_folder: Path | None = None,
    num_ctx: int | None = None,
    search_memo: SearchMemo | None = None,
    t2t_memo: SearchMemo | None = None,
) -> tuple[list[SelectEntry], list[ChatResponse], SelectStageTimings, list[ErrorLog]]:
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
            ambiguous_fields={field: {} for field in fields},
        )

        # ``ExtractEntry.extracted`` is now ``dict | None`` (see model_validator),
        # so we can use it directly.
        extracted_dict = obj.extracted

        if extracted_dict is not None:
            for field in no_select_fields:
                values = _string_values(extracted_dict.get(field))
                if values:
                    se.results[field] = [ResolvedValue(value=v) for v in values]
            # Explicitly set empty list for fields with None value in extracted
            for field in fields:
                if field not in se.results and field in extracted_dict and extracted_dict[field] is None:
                    se.results[field] = []

        intermediate_entries.append(se)

    # 1. Perform ontology search for each field specified in the select configuration.
    with stage_timer("ontology_search") as t_ontology:
        _ontology_search_wrapper(
            intermediate_entries,
            select_config,
            index_map=index_map,
            search_memo=search_memo,
        )

    # 2. Perform text2term search for each field specified in the select configuration.
    with stage_timer("text2term") as t_text2term:
        _text2term_wrapper(
            intermediate_entries,
            select_config,
            index_map=index_map,
            cache_folder=text2term_cache_folder,
            t2t_memo=t2t_memo,
        )

    # 3. For fields that still have multiple matches or no matches, use the LLM to select the best match.

    all_select_chat_responses: list[ChatResponse] = []
    select_errors: list[ErrorLog] = []
    ollama_options = build_ollama_options(num_ctx)

    async def _process_field_selection(
        accession: str,
        field_name: str,
        value: str,
        messages: list[Message],
        schema: JsonSchemaValue,
    ) -> tuple[str, str, str, ChatResponse | None, ErrorLog | None]:
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
            LOGGER.exception(
                "Error during select step for %s/%s/%r",
                accession,
                field_name,
                value,
            )
            wrapped = OllamaProcessingError(
                f"{accession} [select:{field_name}={value!r}]",
                e,
            )
            return (accession, field_name, value, None, build_error_log(wrapped))

        return (accession, field_name, value, response, None)

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

            for accession, field_name, value, chat_response, err in llm_results:
                if err is not None:
                    select_errors.append(err)
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

    return intermediate_entries, all_select_chat_responses, timings, select_errors
