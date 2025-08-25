import asyncio
import copy
import json
import re
from pathlib import Path
from typing import IO, Any, Dict, Iterable, List, Optional

import ollama
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.bs import construct_llm_input_json, is_ebi_format
from bsllmner2.config import LOGGER, Config
from bsllmner2.schema import BsEntries, LlmOutput, Prompt

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


def _normalize_null_like_inplace(obj: Dict[str, Any]) -> None:
    for k, v in list(obj.items()):
        if isinstance(v, str) and v.strip() in ("null", "None"):
            obj[k] = None


def _count_prompt_tokens(config: Config, model: str, messages: list) -> int:
    client = ollama.Client(host=config.ollama_host)
    resp = client.chat(
        model=model,
        messages=messages,
        options={"num_predict": 0, "seed": 0, "temperature": 0.0},
    )
    return int(resp.get("prompt_eval_count", 0))


def _construct_output(
    batch_entries: List[Dict[str, Any]],
    chat_response: ChatResponse
) -> List[LlmOutput]:
    try:
        res_text = chat_response["message"]["content"]
        res_text_json = _extract_last_json(res_text)
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None

    print(res_text_json)

    outputs: List[LlmOutput] = []
    if not res_text_json:
        for entry in batch_entries:
            outputs.append(
                LlmOutput(
                    accession=entry.get("accession", "Unknown"),
                    output=None,
                    output_full=None,
                    chat_response=chat_response,
                )
            )
        return outputs

    try:
        parsed = json.loads(res_text_json)
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.error("Error parsing batch JSON: %s", e)
        parsed = None

    acc2entry = {e.get("accession", "Unknown"): e for e in batch_entries}

    if not isinstance(parsed, list):
        LOGGER.warning("Batch response is not a list; wrapping to list")
        parsed = [parsed] if parsed is not None else []

    for item in parsed:
        if not isinstance(item, dict):
            continue
        _normalize_null_like_inplace(item)
        acc = item.get("accession", "Unknown")
        # 出力は accession 以外のキーを categories とみなす
        categories = {k: v for k, v in item.items() if k != "accession"}

        out = LlmOutput(
            accession=acc,
            output=categories if categories else None,
            output_full=json.dumps(item, ensure_ascii=False),
            chat_response=chat_response,
        )

        # EBI 形式なら characteristics/taxId を付与
        src_entry = acc2entry.get(acc)
        if isinstance(out.output, dict) and src_entry and is_ebi_format(src_entry):
            out.characteristics = {k: {"text": v} for k, v in out.output.items()}
            if "taxId" in src_entry:
                out.taxId = src_entry["taxId"]

        outputs.append(out)

    # 応答に載っていない accession のフォールバック（欠落検知）
    covered = {o.accession for o in outputs}
    for entry in batch_entries:
        acc = entry.get("accession", "Unknown")
        if acc not in covered:
            LOGGER.warning("Missing in batch response: %s", acc)
            outputs.append(
                LlmOutput(
                    accession=acc,
                    output=None,
                    output_full=res_text_json,
                    chat_response=chat_response,
                )
            )

    return outputs


def _split_batches(seq: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


async def ner(
    config: Config,
    bs_entries: BsEntries,
    prompt: List[Prompt],
    item_schema: JsonSchemaValue,
    model: str,
    thinking: Optional[bool] = None,
    progress_file_path: Optional[Path] = None,
    max_concurrency: int = 64,
    batch_size: int = 2,
) -> List[LlmOutput]:
    client = ollama.AsyncClient(host=config.ollama_host)
    base_messages = _construct_messages(prompt)

    batch_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": item_schema,
    }

    outputs: List[LlmOutput] = []

    progress_file: Optional[IO[str]] = None
    if progress_file_path:
        progress_file = progress_file_path.open("w", encoding="utf-8")

    sem = asyncio.Semaphore(max_concurrency)

    async def _process_batch(batch: List[Dict[str, Any]]) -> List[LlmOutput]:
        async with sem:
            accessions = [entry.get("accession", "Unknown") for entry in batch]
            LOGGER.debug("Processing batch: %s", ", ".join(accessions))

            input_payload = [construct_llm_input_json(entry) for entry in batch]
            messages_copy = copy.deepcopy(base_messages)
            if messages_copy[-1].content is not None:
                messages_copy[-1].content += "\n" + json.dumps(input_payload, ensure_ascii=False)

            # count = _count_prompt_tokens(config, model, messages_copy)
            # print(count)
            # return []

            try:
                response: ChatResponse = await client.chat(
                    model=model,
                    messages=messages_copy,
                    options=OLLAMA_OPTIONS,
                    think=thinking,
                    format=batch_schema,
                )
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.error("Error processing batch %s: %s", ", ".join(accessions), e)
                return [
                    LlmOutput(
                        accession=acc,
                        output=None,
                        output_full=None,
                        chat_response=response,
                    ) for acc in accessions
                ]

            batch_outputs = _construct_output(batch, response)

            if progress_file:
                for acc in accessions:
                    progress_file.write(f"{acc}\n")
                progress_file.flush()

            return batch_outputs

    try:
        tasks = [_process_batch(batch) for batch in _split_batches(bs_entries, batch_size)]
        results_nested = await asyncio.gather(*tasks)
        for chunk in results_nested:
            outputs.extend(chunk)
    finally:
        if progress_file:
            progress_file.close()

    return outputs
