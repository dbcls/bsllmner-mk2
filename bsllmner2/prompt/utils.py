from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Represents a prompt for LLM processing.
    """
    role: str
    content: str


def load_prompt_file(path: Path) -> List[Prompt]:
    """
    Load a prompt file from the given path.
    The file should be in YAML format, containing a dictionary where each key is a number as a string.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Prompt file {path} must contain a list of prompts.")

    return [Prompt(**item) for item in raw_data]
