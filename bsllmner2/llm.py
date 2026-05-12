import asyncio
import json
from typing import Any, Protocol, runtime_checkable

import ollama
from ollama import ChatResponse, Message, Options
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.biosample import construct_llm_input_json
from bsllmner2.config import LOGGER, resolve_default_ollama_concurrency
from bsllmner2.errors import OllamaConnectionError, OllamaProcessingError
from bsllmner2.models import BsEntries, ErrorLog, ExtractEntry, Prompt, llm_timing_from_chat_response
from bsllmner2.pipeline import build_error_log


def build_ollama_options(
    num_ctx: int | None = None,
    *,
    seed: int = 0,
    temperature: float = 0.0,
    num_predict: int | None = None,
) -> Options:
    """Return Ollama call options.

    ``num_predict`` defaults to ``None``, which leaves Ollama's own default
    (``-1`` = unlimited within the context window) in effect. Earlier code set
    ``num_predict = num_ctx`` which forced Ollama to pre-allocate buffers for
    the full context window per call, hurting throughput; the override is now
    opt-in.
    """
    opts = Options(seed=seed, temperature=temperature)
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    if num_predict is not None:
        opts["num_predict"] = num_predict
    return opts


# === LLM Backend Protocol ===


@runtime_checkable
class LlmBackend(Protocol):
    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool = False,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse: ...

    async def ensure_model(self, model: str) -> None: ...

    def list_models(self) -> list[str]: ...


# === Ollama implementation ===


class OllamaBackend:
    def __init__(self, host: str, semaphore_limit: int | None = None) -> None:
        self._host = host
        self._semaphore_limit = semaphore_limit if semaphore_limit is not None else resolve_default_ollama_concurrency()
        self._semaphore = asyncio.Semaphore(self._semaphore_limit)
        self._async_client = ollama.AsyncClient(host=host)

    @property
    def host(self) -> str:
        return self._host

    @property
    def semaphore_limit(self) -> int:
        return self._semaphore_limit

    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        options: Options | None = None,
        think: bool = False,
        format_: JsonSchemaValue | None = None,
    ) -> ChatResponse:
        async with self._semaphore:
            return await self._async_client.chat(
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
        # Check if model exists
        models_response = await self._async_client.list()
        available_models = [m.model for m in models_response.models]

        if model in available_models:
            LOGGER.debug("Model %s is already available", model)

            return

        # Model not found, pull it
        LOGGER.info("Model %s not found locally, pulling...", model)
        try:
            async for progress in await self._async_client.pull(model, stream=True):
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


def _extract_last_json_match(
    text: str,
) -> tuple[dict[str, Any] | list[Any], int, int] | None:
    """Return ``(parsed_obj, start, end)`` for the last decodable top-level JSON.

    Returns ``None`` when no valid JSON is found. Shared engine used by both
    :func:`_extract_last_json` (returns the parsed object) and
    :func:`_extract_last_json_str` (returns the raw substring).
    """
    decoder = json.JSONDecoder()
    last: tuple[dict[str, Any] | list[Any], int, int] | None = None
    i = 0
    while i < len(text):
        if text[i] in ("{", "["):
            try:
                obj, end = decoder.raw_decode(text, i)
                last = (obj, i, end)
                i = end
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1
    return last


def _extract_last_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract and return the last valid JSON object/array from *text*."""
    match = _extract_last_json_match(text)
    return match[0] if match is not None else None


def _extract_last_json_str(text: str) -> str | None:
    """Extract the last valid JSON substring from *text* without re-serializing."""
    match = _extract_last_json_match(text)
    if match is None:
        return None
    _obj, start, end = match
    return text[start:end]


def _normalize_null_strings(obj: Any) -> Any:
    """Recursively replace string ``"null"`` and ``"None"`` with ``None``."""
    if isinstance(obj, dict):
        return {k: _normalize_null_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_null_strings(v) for v in obj]
    if isinstance(obj, str) and obj in ("null", "None"):
        return None
    return obj


def parse_response_json(chat_response: ChatResponse) -> dict[str, Any] | list[Any] | None:
    """Parse JSON from a ChatResponse, normalizing string 'null'/'None' values to None."""
    try:
        res_text = chat_response.message.content
        output_obj = _extract_last_json(res_text) if res_text is not None else None
    except (AttributeError, TypeError) as e:
        LOGGER.error("Error extracting JSON from response text: %s", e)
        output_obj = None
    if output_obj is not None:
        normalized: dict[str, Any] | list[Any] = _normalize_null_strings(output_obj)
        return normalized

    return None


def _construct_output(bs_entry: dict[str, Any], chat_response: ChatResponse) -> ExtractEntry:
    output_obj = parse_response_json(chat_response)
    raw_output = _extract_last_json_str(chat_response.message.content or "") if chat_response.message.content else None

    # The schema for `extracted` is ``dict | None``; coerce array outputs to None
    # with a warning so downstream code does not need to defensively type-check.
    if isinstance(output_obj, list):
        LOGGER.warning(
            "Discarding list-shaped LLM output for entry %s (expected object).",
            bs_entry.get("accession"),
        )
        extracted_dict: dict[str, Any] | None = None
    else:
        extracted_dict = output_obj

    return ExtractEntry(
        accession=bs_entry["accession"],
        extracted=extracted_dict,
        raw_output=raw_output,
        llm_timing=llm_timing_from_chat_response(chat_response),
    )


# === NER function ===


async def ner(
    backend: LlmBackend,
    bs_entries: BsEntries,
    prompt: list[Prompt],
    format_: JsonSchemaValue | None,
    model: str,
    thinking: bool = False,
    num_ctx: int | None = None,
) -> tuple[list[ExtractEntry], list[ChatResponse], int, list[ErrorLog]]:
    """Run the NER LLM call across *bs_entries*.

    Returns ``(outputs, chat_responses, error_count, errors_log)``.

    Connection failures on ``ensure_model`` or on the very first entry are
    re-raised as :class:`OllamaConnectionError` so the caller can decide to
    abort the whole run. Per-entry failures after the connection is verified
    are captured in ``errors_log`` (one :class:`ErrorLog` per failed entry)
    and ``error_count`` is incremented; processing continues with the
    remaining entries.
    """
    host = getattr(backend, "host", "unknown")

    # Surface "server is unreachable / model unavailable" before we start
    # the gather loop so callers can fail fast instead of treating every
    # entry as a per-entry processing error.
    try:
        await backend.ensure_model(model)
    except (ConnectionError, OSError, ollama.ResponseError, ollama.RequestError) as e:
        raise OllamaConnectionError(host, e) from e

    ollama_options = build_ollama_options(num_ctx)
    messages = _construct_messages(prompt)
    outputs: list[ExtractEntry] = []
    chat_responses: list[ChatResponse] = []
    errors_log: list[ErrorLog] = []
    connection_tested = False

    async def _process_entry(entry: dict[str, Any]) -> tuple[ExtractEntry, ChatResponse] | None:
        nonlocal connection_tested
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
                options=ollama_options,
                think=thinking,
                format_=format_,
            )
            connection_tested = True
        except (ConnectionError, OSError) as e:
            if not connection_tested:
                # First chat call to the server failed at transport level.
                # Lift to OllamaConnectionError so the caller aborts cleanly.
                raise OllamaConnectionError(host, e) from e
            LOGGER.exception("Connection error for entry %s", accession)
            errors_log.append(build_error_log(OllamaProcessingError(accession, e)))

            return None
        except Exception as e:
            LOGGER.exception("Error processing entry %s", accession)
            errors_log.append(build_error_log(OllamaProcessingError(accession, e)))

            return None

        return _construct_output(entry, response), response

    # Process the first entry serially so that a transport-level failure surfaces
    # as OllamaConnectionError before we issue 256 parallel requests against a
    # broken server. Once ``connection_tested`` flips to True the parallel
    # entries treat per-entry failures as recoverable.
    if bs_entries:
        first_result = await _process_entry(bs_entries[0])
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
    for res in all_results:
        if res is not None:
            outputs.append(res[0])
            chat_responses.append(res[1])

    error_count = len(errors_log)
    if error_count > 0 and len(bs_entries) > 0:
        LOGGER.error(
            "Completed with %d errors out of %d entries (%.1f%% success rate)",
            error_count,
            len(bs_entries),
            (len(bs_entries) - error_count) / len(bs_entries) * 100,
        )

    return outputs, chat_responses, error_count, errors_log
