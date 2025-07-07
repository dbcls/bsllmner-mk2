import copy
import json
import re
from typing import Any, Dict, List, Optional

import ollama
from ollama import ChatResponse, Message, Options

from bsllmner2.bs import construct_llm_input_json, is_ebi_format
from bsllmner2.config import LOGGER, Config
from bsllmner2.schema import BsEntries, LlmOutput, Prompt

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


def _construct_output(bs_entry: Dict[str, Any], res_text: str, chat_response: ChatResponse) -> LlmOutput:
    res_text_json = _extract_last_json(res_text)
    output_obj = json.loads(res_text_json) if res_text_json else None
    if output_obj is not None:
        for k, v in output_obj.items():
            if v in ("null", "None"):
                output_obj[k] = None
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


def ner(
    config: Config,
    bs_entries: BsEntries,
    prompt: List[Prompt],
    model: str
) -> List[LlmOutput]:
    client = ollama.Client(host=config.ollama_host)
    messages = _construct_messages(prompt)
    outputs = []
    for entry in bs_entries:
        LOGGER.debug("Processing entry: %s", entry.get("accession", "Unknown"))
        entry_str = json.dumps(construct_llm_input_json(entry), ensure_ascii=False)
        messages_copy = copy.deepcopy(messages)
        if messages_copy[-1].content is not None:
            messages_copy[-1].content += "\n" + entry_str
        response: ChatResponse = client.chat(
            model=model,
            messages=messages_copy,
            options=OLLAMA_OPTIONS
        )
        res_text = response["message"]["content"]
        output = _construct_output(entry, res_text, response)
        outputs.append(output)

    return outputs
