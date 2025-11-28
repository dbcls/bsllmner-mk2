import asyncio
import copy
import json
import re
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Tuple

import ollama
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.bs import construct_llm_input_json, is_ebi_format
from bsllmner2.config import LOGGER, Config
from bsllmner2.ontology_search import (SearchResult, _is_label_prop,
                                       build_index_from_owl,
                                       build_index_from_table, search_terms,
                                       search_terms_with_text2term)
from bsllmner2.schema import (BsEntries, LlmOutput, Prompt, SelectConfig,
                              SelectResult)

OLLAMA_OPTIONS = Options(
    seed=0,
    temperature=0.0,
)


def fetch_ollama_models(config: Config) -> List[str]:
    """
    Fetch the list of available models from the Ollama server.
    """
    client = ollama.Client(host=config.ollama_host)
    models = client.list()
    model_names = [model.name for model in models]  # type: ignore
    return model_names


def _construct_messages(prompts: List[Prompt]) -> List[Message]:
    """
    Construct a list of messages from the prompt file content.
    """
    return [Message(role=prompt.role, content=prompt.content) for prompt in prompts]


def _extract_last_json(text: str) -> Optional[str]:
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


def _construct_output(bs_entry: Dict[str, Any], chat_response: ChatResponse) -> LlmOutput:
    try:
        res_text = chat_response["message"]["content"]
        res_text_json = _extract_last_json(res_text)
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None
    if res_text_json is not None:
        try:
            output_obj = json.loads(res_text_json) if res_text_json else None
            if output_obj is not None:
                for k, v in output_obj.items():
                    if v in ("null", "None"):
                        output_obj[k] = None
        except Exception as e:  # pylint: disable=broad-except
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
            key: {"text": value} for key, value in output_ins.output.items()
        }
        if "taxId" in bs_entry:
            output_ins.taxId = bs_entry["taxId"]

    return output_ins


async def ner(
    config: Config,
    bs_entries: BsEntries,
    prompt: List[Prompt],
    format_: Optional[JsonSchemaValue],
    model: str,
    thinking: Optional[bool] = None,
    progress_file_path: Optional[Path] = None,
) -> List[LlmOutput]:
    client = ollama.AsyncClient(host=config.ollama_host)
    messages = _construct_messages(prompt)
    outputs: List[LlmOutput] = []

    progress_file: Optional[IO[str]] = None
    if progress_file_path:
        progress_file = progress_file_path.open("w", encoding="utf-8")

    sem = asyncio.Semaphore(256)

    async def _process_entry(entry: Dict[str, Any]) -> Optional[LlmOutput]:
        async with sem:
            accession = entry.get("accession", None)
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
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.error("Error processing entry %s: %s", accession, e)
                return None

            output = _construct_output(entry, response)

            if progress_file:
                progress_file.write(f"{accession}\n")
                progress_file.flush()

            return output

    try:
        results = await asyncio.gather(
            *(_process_entry(entry) for entry in bs_entries)
        )
        outputs.extend([res for res in results if res is not None])
    finally:
        if progress_file:
            progress_file.close()

    return outputs


# === select mode ===


def _pick_exact_match_search_result(
    search_results: List[SearchResult],
) -> Optional[SearchResult]:
    exact_matches = [res for res in search_results if res.exact_match]
    if not exact_matches:
        return None
    if len(exact_matches) == 1:
        return exact_matches[0]

    term_ids = set(search_result.term_id for search_result in exact_matches)
    if len(term_ids) > 1:
        return None

    for search_result in exact_matches:
        if _is_label_prop(search_result.prop_uri):
            return search_result

    # Here, only one unique term_id exists, but no preferred property found.
    # Return the first exact match as a fallback.
    return exact_matches[0]


def _ontology_search_wrapper(
    select_results: List[SelectResult],
    select_config: SelectConfig,
) -> List[SelectResult]:
    """\
    Wrapper function to perform ontology search for each field in the select configuration.

    Args:
        select_results (List[SelectResult]): List of SelectResult objects containing LLM outputs.
        select_config (SelectConfig): Configuration for the select mode.
    """
    for field_name, field_config in select_config.fields.items():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path.suffix == ".owl":
            index = build_index_from_owl(ontology_file_path, additional_conditions=field_config.ontology_filter)
        elif ontology_file_path.suffix in [".tsv", ".csv"]:
            index = build_index_from_table(ontology_file_path)
        else:
            raise ValueError(f"Unsupported ontology file format: {ontology_file_path}")

        LOGGER.info("Searching ontology for field: %s", field_name)
        queries: Set[str] = set()
        for res in select_results:
            if res.extract_output is None or field_name not in res.extract_output:
                continue
            # For idempotency, skip if results already exist
            if res.results.get(field_name, None) is not None:
                continue
            query_value = res.extract_output[field_name]
            if not isinstance(query_value, str):
                continue
            queries.add(query_value)

        search_results = search_terms(index, queries)

        for res in select_results:
            if res.extract_output is None or field_name not in res.extract_output:
                continue
            query_value = res.extract_output[field_name]
            if not isinstance(query_value, str):
                continue
            results_for_query = search_results.get(query_value, [])
            res.search_results[field_name] = list(results_for_query)

            # If exactly one exact match is found, use it directly.
            exact_match_result = _pick_exact_match_search_result(results_for_query)
            if exact_match_result is not None:
                res.results[field_name] = exact_match_result

    return select_results


def _text2term_wrapper(
    select_results: List[SelectResult],
    select_config: SelectConfig,
) -> List[SelectResult]:
    """\
    Wrapper function to perform text2term search for each field in the select configuration.

    Args:
        select_results (List[SelectResult]): List of SelectResult objects containing LLM outputs.
        select_config (SelectConfig): Configuration for the select mode.
    """
    for field_name, field_config in select_config.fields.items():
        ontology_file_path = field_config.ontology_file
        if not ontology_file_path.suffix == ".owl":
            LOGGER.warning("Text2Term currently supports only OWL files. Skipping field: %s", field_name)

        LOGGER.info("text2term for field: %s", field_name)
        queries: Set[str] = set()
        for res in select_results:
            if res.extract_output is None or field_name not in res.extract_output:
                continue
            # For idempotency, skip if results already exist
            if res.results.get(field_name, None) is not None:
                continue
            query_value = res.extract_output[field_name]
            if not isinstance(query_value, str):
                continue
            queries.add(query_value)

        if not queries:
            continue

        text2term_results = search_terms_with_text2term(queries, ontology_file_path)

        for res in select_results:
            if res.extract_output is None or field_name not in res.extract_output:
                continue
            query_value = res.extract_output[field_name]
            if not isinstance(query_value, str):
                continue
            results_for_query = text2term_results.get(query_value, [])
            res.text2term_results[field_name] = list(results_for_query)

            # If exactly one exact match is found, use it directly.
            exact_match_result = _pick_exact_match_search_result(results_for_query)
            if exact_match_result is not None:
                res.results[field_name] = exact_match_result

    return select_results


def _build_select_schema(
    candidates: List[SearchResult],
    reasoning: bool = True,
) -> JsonSchemaValue:
    enum = [res.term_id for res in candidates]

    properties: Dict[str, Any] = {
        "id": {
            "anyOf": [
                {"type": "string", "enum": enum},
                {"type": "null"},
            ]
        },
    }
    required = ["id"]

    if reasoning:
        properties["reasoning"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }
        required.append("reasoning")

    schema: JsonSchemaValue = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    return schema


def _serialize_candidates_for_llm(candidates: List[SearchResult]) -> List[Dict[str, Any]]:
    return [
        c.model_dump(exclude={"exact_match", "text2term_score", "reasoning"})
        for c in candidates
    ]


def _collect_candidates_for_field(
    field_name: str,
    select_result: SelectResult,
) -> List[SearchResult]:
    merged: List[SearchResult] = []
    merged.extend(select_result.search_results.get(field_name, []))
    merged.extend(select_result.text2term_results.get(field_name, []))

    # Remove duplicates based on term_id.
    by_term_id: Dict[str, SearchResult] = {}
    for result in merged:
        prev = by_term_id.get(result.term_id, None)
        if prev is None:
            by_term_id[result.term_id] = result
        else:
            if _is_label_prop(result.prop_uri) and not _is_label_prop(prev.prop_uri):
                by_term_id[result.term_id] = result

    return list(by_term_id.values())


def _pick_search_result_by_id(
    select_result: SelectResult,
    field_name: str,
    term_id: str,
) -> Optional[SearchResult]:
    candidates = _collect_candidates_for_field(field_name, select_result)
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
    bs_entry: Dict[str, Any],
    select_result: SelectResult,
    select_config: SelectConfig,
    reasoning: bool,
) -> Dict[str, Tuple[List[Message], JsonSchemaValue]]:
    """
    Build per-field (messages, schema) for LLM selection (choose term_id)
    Only includes fields that still need a selection (i.e., results[field] is None)
    """
    results: Dict[str, Tuple[List[Message], JsonSchemaValue]] = {}
    bs_ctx_json = json.dumps(bs_entry, ensure_ascii=False)
    system_msg = _build_select_system_message(reasoning)

    for field_name, field_config in select_config.fields.items():
        # Skip fields that already have a result
        if select_result.results.get(field_name, None) is not None:
            continue

        # Skip fields without extract_output
        extracted_value = None
        if isinstance(select_result.extract_output, dict):
            value = select_result.extract_output.get(field_name, None)
            if isinstance(value, str):
                extracted_value = value
        if extracted_value is None:
            continue

        candidates = _collect_candidates_for_field(field_name, select_result)
        if not candidates:
            continue

        schema = _build_select_schema(candidates, reasoning=reasoning)

        reasoning_instr = ""
        if reasoning:
            reasoning_instr = "For 'reasoning', provide: (1) exact evidence text, (2) a brief comparison of the top 2-3 candidates, (3) explicit rejection reasons for the others."

        user_msg = Message(
            role="user",
            content=(
                f"Field: {field_name}\n\n"
                f"Description: {(field_config.prompt_description or field_name)}\n\n"
                "Provenance:\n"
                "- The 'extracted value' below was produced by an earlier NER step and may be noisy.\n"
                "- The 'ontology candidates' were assembled by ontology search (and possibly text2term) and are the ONLY allowed choices.\n"
                "- Decide strictly from the provided metadata and candidates; do not use outside knowledge.\n\n"
                f"Original extracted value: {extracted_value}\n\n"
                f"BioSample metadata (context):\n{bs_ctx_json}\n\n"
                f"Ontology candidates (JSON array):\n"
                f"{json.dumps(_serialize_candidates_for_llm(candidates), ensure_ascii=False, indent=2)}\n\n"
                "Return ONLY JSON that matches the schema.\n"
                f"{reasoning_instr}"
            ),
        )

        results[field_name] = (
            [system_msg, user_msg],
            schema,
        )

    return results


def _parse_output_object(chat_response: ChatResponse) -> Optional[Dict[str, Any]]:
    try:
        res_text = chat_response["message"]["content"]
        res_text_json = _extract_last_json(res_text)
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None
    if res_text_json is not None:
        try:
            output_obj = json.loads(res_text_json) if res_text_json else None
            if output_obj is not None:
                for k, v in output_obj.items():
                    if v in ("null", "None"):
                        output_obj[k] = None
            return output_obj
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error("Error parsing JSON response: %s", e)
            return None
    return None


async def select(
    config: Config,
    bs_entries: BsEntries,
    model: str,
    extract_outputs: List[LlmOutput],
    select_config: SelectConfig,
    thinking: Optional[bool] = None,
    include_reasoning: bool = True,
) -> List[SelectResult]:
    fields = select_config.fields.keys()

    intermediate_results: List[SelectResult] = []
    for obj in extract_outputs:
        intermediate_results.append(SelectResult(
            accession=obj.accession,
            extract_output=obj.output,
            search_results={field: [] for field in fields},
            text2term_results={field: [] for field in fields},
            llm_chat_response={field: None for field in fields},
            results={field: None for field in fields},
        ))

    # 1. Perform ontology search for each field specified in the select configuration.
    #   1.1 If no matches are found, proceed to step 2.
    #   1.2 If exactly one match is found, use that result as the final result for that field.
    #   1.3 If multiple matches are found, proceed to step 2.
    _ontology_search_wrapper(intermediate_results, select_config)

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
        messages: List[Message],
        schema: JsonSchemaValue
    ) -> Tuple[str, str, Optional[ChatResponse]]:
        async with sem:
            try:
                LOGGER.debug("[Select] Processing entry: %s, field: %s", accession, field_name)
                response: Optional[ChatResponse] = await client.chat(
                    model=model,
                    messages=messages,
                    options=OLLAMA_OPTIONS,
                    think=thinking,
                    format=schema,
                )
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.error("Error during select step: %s", e)
                response = None

            return (accession, field_name, response)

    tasks = []
    # bs_entries and intermediate_results are aligned by accession
    for bs_entry, select_result in zip(bs_entries, intermediate_results):
        accession = select_result.accession
        field_prompts_and_schemas = _build_select_prompt_and_schema(bs_entry, select_result, select_config, include_reasoning)
        for field_name, (messages, schema) in field_prompts_and_schemas.items():
            tasks.append(asyncio.create_task(
                _process_field_selection(accession, field_name, messages, schema)
            ))

    if tasks:
        LOGGER.info("Performing LLM selection for %d fields across %d entries...", len(tasks), len(intermediate_results))
        acc_to_result_map = {result.accession: result for result in intermediate_results}
        llm_results = await asyncio.gather(*tasks)
        for accession, field_name, chat_response in llm_results:
            select_result = acc_to_result_map[accession]
            if select_result is None or chat_response is None:
                continue

            select_result.llm_chat_response[field_name] = chat_response

            output_obj = _parse_output_object(chat_response)
            chosen_id = output_obj.get("id", None) if output_obj else None
            reasoning = output_obj.get("reasoning", None) if output_obj else None
            if not isinstance(chosen_id, str) or not chosen_id.strip():
                continue
            chosen_id = chosen_id.strip()
            picked_result = _pick_search_result_by_id(select_result, field_name, chosen_id)
            if picked_result is None:
                continue

            picked_copy = picked_result.model_copy(deep=True)
            if isinstance(reasoning, str):
                picked_copy.reasoning = reasoning

            select_result.results[field_name] = picked_copy

    return intermediate_results
