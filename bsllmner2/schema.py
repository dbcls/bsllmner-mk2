from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ollama import ChatResponse
from pydantic import BaseModel, Field

from bsllmner2.config import Config
from bsllmner2.metrics import Metrics

API_VERSION = "1.0.0"


class CliExtractArgs(BaseModel):
    """
    Command-line arguments for the bsllmner2 CLI extract mode.
    """
    bs_entries: Path = Field(
        ...,
        description="Path to the input JSON or JSONL file containing BioSample entries.",
        examples=["data/bs_entries.json", "data/bs_entries.jsonl"],
    )
    mapping: Path = Field(
        ...,
        description="Path to the mapping file in TSV format.",
        examples=["mapping/mapping.tsv"],
    )
    prompt: Path = Field(
        ...,
        description="Path to the prompt file in YAML format.",
        examples=["prompt/prompt_extract.yml"],
    )
    model: str = "llama3.1:70b"
    max_entries: Optional[int] = None
    with_metrics: bool = False


class ServiceInfo(BaseModel):
    api_version: str = Field(
        ...,
        description="API version of this api service",
        examples=[API_VERSION],
    )


class Prompt(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the message in the conversation",
        examples=["system", "user", "assistant"],
    )
    content: str = Field(
        ...,
        description="Content of the message",
        examples=["You are a helpful assistant.", "What is the capital of France?"],
    )


BsEntries = List[Dict[str, Any]]


class MappingValue(BaseModel):
    experiment_type: str
    extraction_answer: Optional[str]
    mapping_answer_id: Optional[str]
    mapping_answer_label: Optional[str]


Mapping = Dict[str, MappingValue]  # key: bs_entry accession


class WfInput(BaseModel):
    bs_entries: BsEntries
    mapping: Mapping
    prompt: List[Prompt]
    model: str
    config: Config
    cli_args: Optional[CliExtractArgs] = None


class LlmOutput(BaseModel):
    accession: str
    output: Optional[Any] = None
    output_full: Optional[str] = None
    characteristics: Optional[Dict[str, Any]] = None
    taxId: Optional[Any] = None
    chat_response: ChatResponse


class Evaluation(BaseModel):
    accession: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    match: bool = False


class RunMetadata(BaseModel):
    run_name: str
    model: str
    username: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None
    status: Literal["running", "completed", "failed"] = "running"
    processing_time: Optional[float] = None
    matched_entries: Optional[int] = None
    total_entries: Optional[int] = None
    accuracy: Optional[float] = None


class ErrorInfo(BaseModel):
    type: str
    message: str
    traceback: str


class ErrorLog(BaseModel):
    timestamp: str
    error: ErrorInfo


class Result(BaseModel):
    input: WfInput
    output: List[LlmOutput] = []
    evaluation: List[Evaluation] = []
    metrics: Optional[List[Metrics]] = None
    run_metadata: RunMetadata
    error_log: Optional[ErrorLog] = None
