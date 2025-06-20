import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import ollama
from ollama import ChatResponse, Message, Options
from pydantic import BaseModel

from bsllmner2.bs import is_ebi_format
from bsllmner2.config import LOGGER, Config
from bsllmner2.prompt import Prompt

# === Paste from ollama.Options ===
# class RuntimeOptions:
#     num_keep: Optional[int] = None            # Number of past tokens to keep cached (used for context continuation)
#     seed: Optional[int] = None                # Random seed for reproducible outputs
#     num_predict: Optional[int] = None         # Maximum number of tokens to generate (-1 = no limit)
#     top_k: Optional[int] = None               # Sample from top K most probable tokens (conservative/diverse control)
#     top_p: Optional[float] = None             # Nucleus sampling: cumulative probability threshold p
#     tfs_z: Optional[float] = None             # Tail-free sampling z-cutoff to trim low‑prob tokens
#     typical_p: Optional[float] = None         # Typical sampling threshold balancing quality vs frequency
#     repeat_last_n: Optional[int] = None       # Penalize repeating tokens in last N outputs (0 = disabled)
#     temperature: Optional[float] = None       # Sampling temperature (0 = deterministic, higher = more random)
#     repeat_penalty: Optional[float] = None    # Penalty factor for token repetition (>1 reduces repeats)
#     presence_penalty: Optional[float] = None  # Penalize based on whether token has appeared before
#     frequency_penalty: Optional[float] = None # Penalize based on token frequency in output
#     mirostat: Optional[int] = None            # Enable Mirostat sampling (0 = off, 1 = Mirostat, 2 = Mirostat 2.0)
#     mirostat_tau: Optional[float] = None      # Target entropy for Mirostat
#     mirostat_eta: Optional[float] = None      # Learning rate for Mirostat parameter adjustments
#     penalize_newline: Optional[bool] = None   # Apply penalty to newline tokens (to discourage blank lines)
#     stop: Optional[Sequence[str]] = None      # List of stop-sequences; generation halts when any is encountered

OLLAMA_OPTIONS = Options(
    seed=0,
    temperature=0.0,
)
OLLAMA_MODELS = [
    "llama3.1:70b",
    "deepseek-r1:70b",
]


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


class Output(BaseModel):
    accession: str
    output: Optional[Any] = None
    output_full: Optional[str] = None
    characteristics: Optional[Dict[str, Any]] = None
    taxId: Optional[Any] = None


def _construct_output(bs_entry: Dict[str, Any], res_text: str) -> Output:
    res_text_json = _extract_last_json(res_text)
    output_json = Output(
        accession=bs_entry["accession"],
        output=json.loads(res_text_json) if res_text_json else None,
        output_full=res_text_json,
    )

    # add "characteristics" and "taxId"
    if isinstance(output_json.output, dict) and is_ebi_format(bs_entry):
        output_json.characteristics = {
            key: {"text": value} for key, value in output_json.output.items()
        }
        if "taxId" in bs_entry:
            output_json.taxId = bs_entry["taxId"]

    return output_json


def ner(
    config: Config,
    bs_entries: List[Dict[str, Any]],
    prompts: List[Prompt],
    model: str
) -> List[Tuple[ChatResponse, Output]]:
    client = ollama.Client(host=config.ollama_host)
    messages = _construct_messages(prompts)
    results = []
    for entry in bs_entries:
        LOGGER.debug("Processing entry: %s", entry.get("accession", "Unknown"))
        entry_str = json.dumps(entry, ensure_ascii=False)
        messages_copy = copy.deepcopy(messages)
        if messages_copy[-1].content is not None:
            messages_copy[-1].content += "\n" + entry_str
        LOGGER.debug("Messages: %s", [msg.model_dump() for msg in messages_copy])
        response: ChatResponse = client.chat(
            model=model,
            messages=messages_copy,
            options=OLLAMA_OPTIONS
        )
        LOGGER.debug("Response: %s", response.model_dump())
        res_text = response["message"]["content"]
        output = _construct_output(entry, res_text)
        results.append((response, output))

    return results
