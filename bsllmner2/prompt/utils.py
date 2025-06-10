from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Represents a prompt for LLM processing.
    """
    role: str
    text: str


def load_prompt_file(path: Path) -> Dict[int, Prompt]:
    """
    Load a prompt file from the given path.
    The file should be in YAML format, containing a dictionary where each key is a number as a string.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f)

    if not isinstance(prompt, dict):
        raise ValueError(f"Prompt file {path} must contain a dictionary.")
    prompts: Dict[int, Prompt] = {}
    for key, value in prompt.items():
        if not isinstance(value, dict):
            raise ValueError(f"Prompt entry {key} must be a dictionary.")
        try:
            int_key = int(key)  # Ensure the key is a valid integer string
        except ValueError as e:
            raise ValueError(f"Prompt key {key} must be a string representing an integer.") from e
        prompts[int_key] = Prompt(**value)

    return prompts
