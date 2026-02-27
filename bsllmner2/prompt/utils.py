from pydantic import BaseModel


class Prompt(BaseModel):
    """Represents a prompt for LLM processing."""

    role: str
    content: str
