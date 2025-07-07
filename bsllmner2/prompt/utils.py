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
