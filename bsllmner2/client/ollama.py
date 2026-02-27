import asyncio
import copy
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import IO, Any

import ollama
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.bs import construct_llm_input_json, is_ebi_format
from bsllmner2.config import LOGGER, Config
from bsllmner2.ontology_search import (
    OntologyIndex,
    SearchResult,
    _is_label_prop,
    build_index_from_owl,
    build_index_from_table,
    search_terms,
    search_terms_with_text2term,
)
from bsllmner2.schema import BsEntries, LlmOutput, Prompt, SelectConfig, SelectResult

OLLAMA_OPTIONS = Options(
    seed=0,
    temperature=0.0,
)


def fetch_ollama_models(config: Config) -> list[str]:
    """Fetch the list of available models from the Ollama server."""
    client = ollama.Client(host=config.ollama_host)
    models = client.list()
    return [model.name for model in models]  # type: ignore[attr-defined]


async def ensure_model_available(config: Config, model: str) -> None:
    """Ensure the specified model is available on the Ollama server.

    If not available, pull it automatically.
    """
    client = ollama.AsyncClient(host=config.ollama_host)

    # Check if model exists
    models_response = await client.list()
    available_models = [m.model for m in models_response.models]

    # Normalize model name for comparison (e.g., "llama3.1:70b" matches "llama3.1:70b")
    if model in available_models:
        LOGGER.debug("Model %s is already available", model)
        return

    # Model not found, pull it
    LOGGER.info("Model %s not found locally, pulling...", model)
    try:
        async for progress in await client.pull(model, stream=True):
            if progress.status:
                if progress.completed and progress.total:
                    pct = (progress.completed / progress.total) * 100
                    LOGGER.info("Pulling %s: %s (%.1f%%)", model, progress.status, pct)
                else:
                    LOGGER.info("Pulling %s: %s", model, progress.status)
        LOGGER.info("Model %s pulled successfully", model)
    except ollama.ResponseError as e:
        LOGGER.error("Failed to pull model %s: %s", model, e)
        raise


def _construct_messages(prompts: list[Prompt]) -> list[Message]:
    """Construct a list of messages from the prompt file content."""
    return [Message(role=prompt.role, content=prompt.content) for prompt in prompts]


def _extract_last_json(text: str) -> str | None:
    json_candidates = re.findall(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)

    if not json_candidates:
        return None

    for candidate in reversed(json_candidates):
        try:
            json_obj = json.loads(candidate)
            return json.dumps(json_obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue

    return None


def _construct_output(bs_entry: dict[str, Any], chat_response: ChatResponse) -> LlmOutput:
    try:
        res_text = chat_response["message"]["content"]
        res_text_json = _extract_last_json(res_text)
    except Exception as e:
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None
    if res_text_json is not None:
        try:
            output_obj = json.loads(res_text_json) if res_text_json else None
            if output_obj is not None:
                for k, v in output_obj.items():
                    if isinstance(v, str) and v in ("null", "None"):
                        output_obj[k] = None
        except Exception as e:
            LOGGER.error("Error parsing JSON response: %s", e)
            output_obj = None
    else:
        output_obj = None

    output_ins = LlmOutput(
        accession=bs_entry["accession"],
        output=output_obj,
        output_full=res_text_json,
        chat_response=chat_response,
    )

    # add "characteristics" and "taxId"
    if isinstance(output_ins.output, dict) and is_ebi_format(bs_entry):
        output_ins.characteristics = {
            key: ({"text": value} if not isinstance(value, list) else [{"text": v} for v in value])
            for key, value in output_ins.output.items()
        }
        if "taxId" in bs_entry:
            output_ins.taxId = bs_entry["taxId"]

    return output_ins


async def ner(
    config: Config,
    bs_entries: BsEntries,
    prompt: list[Prompt],
    format_: JsonSchemaValue | None,
    model: str,
    thinking: bool | None = None,
    progress_file_path: Path | None = None,
) -> list[LlmOutput]:
    from bsllmner2.errors import OllamaConnectionError

    # Ensure model is available, pull if necessary
    await ensure_model_available(config, model)

    client = ollama.AsyncClient(host=config.ollama_host)
    messages = _construct_messages(prompt)
    outputs: list[LlmOutput] = []
    error_count = 0
    connection_tested = False

    progress_file: IO[str] | None = None
    if progress_file_path:
        progress_file = progress_file_path.open("w", encoding="utf-8")

    sem = asyncio.Semaphore(256)

    async def _process_entry(entry: dict[str, Any]) -> LlmOutput | None:
        nonlocal error_count, connection_tested
        async with sem:
            accession = entry.get("accession")
            if accession is None:
                LOGGER.warning("Entry without accession found, skipping.")
                return None
            LOGGER.debug("[NER] Processing entry: %s", accession)
            entry_str = json.dumps(construct_llm_input_json(entry), ensure_ascii=False)
            messages_copy = copy.deepcopy(messages)
            if messages_copy[-1].content is not None:
                messages_copy[-1].content += "\n" + entry_str
            try:
                response: ChatResponse = await client.chat(
                    model=model,
                    messages=messages_copy,
                    options=OLLAMA_OPTIONS,
                    think=thinking,
                    format=format_,
                )
                connection_tested = True
            except (ConnectionError, OSError) as e:
                if not connection_tested:
                    raise OllamaConnectionError(config.ollama_host, e) from e
                LOGGER.error("Connection error for entry %s: %s", accession, e)
                error_count += 1
                return None
            except Exception as e:
                LOGGER.error("Error processing entry %s: %s", accession, e)
                error_count += 1
                return None

            output = _construct_output(entry, response)

            if progress_file:
                progress_file.write(f"{accession}\n")
                progress_file.flush()

            return output

    try:
        results = await asyncio.gather(*(_process_entry(entry) for entry in bs_entries))
        outputs.extend([res for res in results if res is not None])
    finally:
        if progress_file:
            progress_file.close()

    if error_count > 0:
        LOGGER.warning(
            "Completed with %d errors out of %d entries (%.1f%% success rate)",
            error_count,
            len(bs_entries),
            (len(bs_entries) - error_count) / len(bs_entries) * 100,
        )

    return outputs


# === select mode ===


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
        if _is_label_prop(search_result.prop_uri):
            return search_result

    # Here, only one unique term_id exists, but no preferred property found.
    # Return the first exact match as a fallback.
    return exact_matches[0]


INDEX_CACHE_DIR = Path(os.environ.get("BSLLMNER2_INDEX_CACHE_DIR", "/app/ontology/index_cache"))
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _compute_filter_hash(ontology_filter: dict[str, str] | None) -> str:
    """Compute a hash of the ontology filter for cache key."""
    if ontology_filter is None:
        return "nofilter"
    filter_str = json.dumps(ontology_filter, sort_keys=True)
    return hashlib.md5(filter_str.encode()).hexdigest()[:12]


def build_index_map(select_config: SelectConfig) -> dict[Path, OntologyIndex]:
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
            except Exception:
                pass

        if ontology_file_path.suffix == ".owl":
            index = build_index_from_owl(ontology_file_path, additional_conditions=field_config.ontology_filter)
        elif ontology_file_path.suffix in [".tsv", ".csv"]:
            index = build_index_from_table(ontology_file_path)
        else:
            raise ValueError(f"Unsupported ontology file format: {ontology_file_path}")
        mapping[ontology_file_path] = index

        try:
            with cache_file_path.open("wb") as f:
                pickle.dump(index, f)
        except Exception:
            pass

    return mapping


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
        elif ontology_file_path.suffix == ".owl":
            index = build_index_from_owl(
                ontology_file_path,
                additional_conditions=field_config.ontology_filter,
            )
        elif ontology_file_path.suffix in [".tsv", ".csv"]:
            index = build_index_from_table(ontology_file_path)
        else:
            raise ValueError(f"Unsupported ontology file format: {ontology_file_path}")

        LOGGER.info("Searching ontology for field: %s", field_name)

        queries: set[str] = set()
        for res in select_results:
            if not isinstance(res.extract_output, dict):
                continue
            if field_name not in res.extract_output:
                continue

            # skip only if final result already exists
            if res.results.get(field_name):
                continue

            query_value = res.extract_output[field_name]
            if isinstance(query_value, str):
                queries.add(query_value)
            elif isinstance(query_value, list):
                queries.update(v for v in query_value if isinstance(v, str))

        if not queries:
            continue

        search_results = search_terms(index, queries)

        for res in select_results:
            if not isinstance(res.extract_output, dict):
                continue
            if field_name not in res.extract_output:
                continue

            # skip only if final result already exists
            if res.results.get(field_name):
                continue

            query_value = res.extract_output[field_name]
            if isinstance(query_value, str):
                values = [query_value]
            elif isinstance(query_value, list):
                values = [v for v in query_value if isinstance(v, str)]
            else:
                continue

            field_search_results = res.search_results.get(field_name)
            if not isinstance(field_search_results, dict):
                field_search_results = {}
                res.search_results[field_name] = field_search_results

            field_results = res.results.get(field_name)
            if field_results is None:
                field_results = {}
                res.results[field_name] = field_results

            for value in values:
                candidates = search_results.get(value, [])
                field_search_results[value] = candidates

                exact_match_result = _pick_exact_match_search_result(candidates)
                if exact_match_result is not None:
                    field_results[value] = exact_match_result

    return select_results


def _text2term_wrapper(
    select_results: list[SelectResult],
    select_config: SelectConfig,
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

        queries: set[str] = set()
        for res in select_results:
            if not isinstance(res.extract_output, dict):
                continue
            if field_name not in res.extract_output:
                continue

            # skip only if final result already exists
            if res.results.get(field_name):
                continue

            query_value = res.extract_output[field_name]
            if isinstance(query_value, str):
                queries.add(query_value)
            elif isinstance(query_value, list):
                queries.update(v for v in query_value if isinstance(v, str))

        if not queries:
            continue

        try:
            text2term_results = search_terms_with_text2term(queries, ontology_file_path)
        except Exception as e:
            LOGGER.exception(
                "text2term failed. field: %s, error: %s",
                field_name,
                e,
            )
            text2term_results = {}

        for res in select_results:
            if not isinstance(res.extract_output, dict):
                continue
            if field_name not in res.extract_output:
                continue

            # skip only if final result already exists
            if res.results.get(field_name):
                continue

            query_value = res.extract_output[field_name]
            if isinstance(query_value, str):
                values = [query_value]
            elif isinstance(query_value, list):
                values = [v for v in query_value if isinstance(v, str)]
            else:
                continue

            field_text2term_results = res.text2term_results.get(field_name)
            if not isinstance(field_text2term_results, dict):
                field_text2term_results = {}
                res.text2term_results[field_name] = field_text2term_results

            field_results = res.results.get(field_name)
            if field_results is None:
                field_results = {}
                res.results[field_name] = field_results

            for value in values:
                candidates = text2term_results.get(value, [])
                field_text2term_results[value] = candidates

                exact_match_result = _pick_exact_match_search_result(candidates)
                if exact_match_result is not None:
                    field_results[value] = exact_match_result

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
        if prev is None or (_is_label_prop(result.prop_uri) and not _is_label_prop(prev.prop_uri)):
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
        if candidate.term_id == term_id and _is_label_prop(candidate.prop_uri):
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
            if field_name in select_result.results and value in select_result.results[field_name]:
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
    try:
        res_text = chat_response["message"]["content"]
        res_text_json = _extract_last_json(res_text)
    except Exception as e:
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None
    if res_text_json is not None:
        try:
            output_obj = json.loads(res_text_json) if res_text_json else None
            if output_obj is not None:
                for k, v in output_obj.items():
                    if v in ("null", "None"):
                        output_obj[k] = None
        except Exception as e:
            LOGGER.error("Error parsing JSON response: %s", e)
            return None
        else:
            return output_obj
    return None


async def select(
    config: Config,
    bs_entries: BsEntries,
    model: str,
    extract_outputs: list[LlmOutput],
    select_config: SelectConfig,
    thinking: bool | None = None,
    include_reasoning: bool = True,
    index_map: dict[Path, OntologyIndex] | None = None,
) -> list[SelectResult]:
    # Ensure model is available, pull if necessary
    await ensure_model_available(config, model)

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
    #   1.1 If no matches are found, proceed to step 2.
    #   1.2 If exactly one match is found, use that result as the final result for that field.
    #   1.3 If multiple matches are found, proceed to step 2.
    _ontology_search_wrapper(intermediate_results, select_config, index_map=index_map)

    # 2. Perform text2term search for each field specified in the select configuration.
    #   2.1 If no matches are found, proceed to step 3.
    #   2.2 If exactly one match is found, use that result as the final result for that field.
    #   2.3 If multiple matches are found, proceed to step 3.
    _text2term_wrapper(intermediate_results, select_config)

    # 3. For fields that still have multiple matches or no matches, use the LLM to select the best match.
    #   The LLM prompt should include the original field value, the list of candidate matches from steps 1 and 2, and bs_entry context.
    client = ollama.AsyncClient(host=config.ollama_host)
    sem = asyncio.Semaphore(256)

    async def _process_field_selection(
        accession: str,
        field_name: str,
        value: str,
        messages: list[Message],
        schema: JsonSchemaValue,
    ) -> tuple[str, str, str, ChatResponse | None]:
        async with sem:
            try:
                LOGGER.debug("[Select] Processing entry: %s, field: %s", accession, field_name)
                response: ChatResponse | None = await client.chat(
                    model=model,
                    messages=messages,
                    options=OLLAMA_OPTIONS,
                    think=thinking,
                    format=schema,
                )
            except Exception as e:
                LOGGER.error("Error during select step: %s", e)
                response = None

            return (accession, field_name, value, response)

    tasks = []
    # bs_entries and intermediate_results are aligned by accession
    for bs_entry, select_result in zip(bs_entries, intermediate_results, strict=False):
        accession = select_result.accession
        field_prompts_and_schemas = _build_select_prompt_and_schema(
            bs_entry,
            select_result,
            select_config,
            include_reasoning,
        )
        for (field_name, value), (messages, schema) in field_prompts_and_schemas.items():
            tasks.append(asyncio.create_task(_process_field_selection(accession, field_name, value, messages, schema)))

    if tasks:
        LOGGER.info(
            "Performing LLM selection for %d fields across %d entries...",
            len(tasks),
            len(intermediate_results),
        )
        acc_to_result_map = {result.accession: result for result in intermediate_results}
        llm_results = await asyncio.gather(*tasks)

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

            if field_name not in select_result.results:
                select_result.results[field_name] = {}
            select_result.results[field_name][value] = picked_copy

    return intermediate_results
