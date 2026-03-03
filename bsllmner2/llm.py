import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import ollama
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.biosample import construct_llm_input_json, is_ebi_format
from bsllmner2.config import LOGGER
from bsllmner2.errors import OllamaConnectionError
from bsllmner2.models import BsEntries, LlmOutput, Prompt

OLLAMA_OPTIONS = Options(
    seed=0,
    temperature=0.0,
)


# === LLM Backend Protocol ===


@runtime_checkable
class LlmBackend(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool | None = None,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse: ...

    async def ensure_model(self, model: str) -> None: ...

    def list_models(self) -> list[str]: ...


# === Ollama implementation ===


class OllamaBackend:
    def __init__(self, host: str, semaphore_limit: int = 256) -> None:
        self._host = host
        self._semaphore = asyncio.Semaphore(semaphore_limit)

    @property
    def host(self) -> str:
        return self._host

    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool | None = None,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse:
        async with self._semaphore:
            client = ollama.AsyncClient(host=self._host)

            return await client.chat(
                model=model,
                messages=messages,
                options=options,
                think=think,
                format=format_,
            )

    async def ensure_model(self, model: str) -> None:
        """Ensure the specified model is available on the Ollama server.

        If not available, pull it automatically.
        """
        client = ollama.AsyncClient(host=self._host)

        # Check if model exists
        models_response = await client.list()
        available_models = [m.model for m in models_response.models]

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

    def list_models(self) -> list[str]:
        """Fetch the list of available models from the Ollama server."""
        client = ollama.Client(host=self._host)
        models_response = client.list()

        return [m.model for m in models_response.models if m.model is not None]


# === Private helpers ===


def _construct_messages(prompts: list[Prompt]) -> list[Message]:
    """Construct a list of messages from the prompt file content."""
    return [Message(role=prompt.role, content=prompt.content) for prompt in prompts]


def _extract_last_json(text: str) -> str | None:
    decoder = json.JSONDecoder()
    last_obj = None
    i = 0
    while i < len(text):
        if text[i] in ("{", "["):
            try:
                obj, end = decoder.raw_decode(text, i)
                last_obj = obj
                i = end
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1

    if last_obj is not None:
        return json.dumps(last_obj, ensure_ascii=False)

    return None


def parse_response_json(chat_response: ChatResponse) -> dict[str, Any] | list[Any] | None:
    """Parse JSON from a ChatResponse, normalizing string 'null'/'None' values to None."""
    try:
        res_text = chat_response.message.content
        res_text_json = _extract_last_json(res_text) if res_text is not None else None
    except (AttributeError, TypeError) as e:
        LOGGER.error("Error extracting JSON from response text: %s", e)
        res_text_json = None
    if res_text_json is not None:
        try:
            output_obj = json.loads(res_text_json) if res_text_json else None
            if isinstance(output_obj, dict):
                for k, v in output_obj.items():
                    if isinstance(v, str) and v in ("null", "None"):
                        output_obj[k] = None
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            LOGGER.error("Error parsing JSON response: %s", e)

            return None
        else:
            return output_obj

    return None


def _construct_output(bs_entry: dict[str, Any], chat_response: ChatResponse) -> LlmOutput:
    output_obj = parse_response_json(chat_response)
    res_text_json = _extract_last_json(chat_response.message.content or "") if chat_response.message.content else None

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


# === NER function ===


async def ner(
    backend: LlmBackend,
    bs_entries: BsEntries,
    prompt: list[Prompt],
    format_: JsonSchemaValue | None,
    model: str,
    thinking: bool | None = None,
    progress_file_path: Path | None = None,
) -> list[LlmOutput]:
    # Ensure model is available, pull if necessary
    await backend.ensure_model(model)

    messages = _construct_messages(prompt)
    outputs: list[LlmOutput] = []
    error_count = 0
    connection_tested = False

    with contextlib.ExitStack() as stack:
        progress_file = (
            stack.enter_context(progress_file_path.open("w", encoding="utf-8")) if progress_file_path else None
        )

        async def _process_entry(entry: dict[str, Any]) -> LlmOutput | None:
            nonlocal error_count, connection_tested
            accession = entry.get("accession")
            if accession is None:
                LOGGER.warning("Entry without accession found, skipping.")

                return None
            LOGGER.debug("[NER] Processing entry: %s", accession)
            entry_str = json.dumps(construct_llm_input_json(entry), ensure_ascii=False)
            last_msg = messages[-1]
            base_content = last_msg.content or ""
            messages_copy = [
                *messages[:-1],
                Message(role=last_msg.role, content=base_content + "\n" + entry_str),
            ]
            try:
                response: ChatResponse = await backend.chat(
                    model=model,
                    messages=messages_copy,
                    options=OLLAMA_OPTIONS,
                    think=thinking,
                    format_=format_,
                )
                connection_tested = True
            except (ConnectionError, OSError) as e:
                if not connection_tested:
                    host = getattr(backend, "host", "unknown")
                    raise OllamaConnectionError(host, e) from e
                LOGGER.error("Connection error for entry %s: %s", accession, e)
                error_count += 1

                return None
            except (ollama.ResponseError, RuntimeError) as e:
                LOGGER.error("Error processing entry %s: %s", accession, e)
                error_count += 1

                return None

            output = _construct_output(entry, response)

            if progress_file:
                progress_file.write(f"{accession}\n")
                progress_file.flush()

            return output

        # Process the first entry serially to verify the connection
        if bs_entries:
            first_result = await _process_entry(bs_entries[0])
            connection_tested = True
            remaining = bs_entries[1:]
        else:
            first_result = None
            remaining = []

        # Process the remaining entries in parallel
        if remaining:
            rest_results = await asyncio.gather(*(_process_entry(entry) for entry in remaining))
        else:
            rest_results = []

        all_results = [first_result, *rest_results]
        outputs.extend([res for res in all_results if res is not None])

    if error_count > 0 and len(bs_entries) > 0:
        LOGGER.error(
            "Completed with %d errors out of %d entries (%.1f%% success rate)",
            error_count,
            len(bs_entries),
            (len(bs_entries) - error_count) / len(bs_entries) * 100,
        )

    return outputs
