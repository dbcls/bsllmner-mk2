from pathlib import Path
from typing import Any, Literal

from ollama import ChatResponse
from pydantic import BaseModel, Field
from pydantic.json_schema import JsonSchemaValue

from bsllmner2.config import Config

# === Ontology search models (moved from ontology_search.py) ===


class TermAnnotation(BaseModel):
    term_uri: str = Field(..., description="The term URI")
    term_id: str = Field(..., description="The term ID, e.g., CL:0000000")  # normalized
    prop_uri: str | None = Field(
        None,
        description="The property URI, e.g., http://www.w3.org/2000/01/rdf-schema#label",
    )
    value: str = Field(..., description="The property value")


class OntologyIndex(BaseModel):
    term_id_to_labels: dict[str, list[str]] = Field(default_factory=dict)
    value_to_annotations: dict[str, list[TermAnnotation]] = Field(default_factory=dict)  # key is _normalize_key(value)


class SearchResult(TermAnnotation):
    label: str | None = None
    exact_match: bool
    text2term_score: float | None = None
    reasoning: str | None = None


class CliExtractArgs(BaseModel):
    """Command-line arguments for the bsllmner2 CLI extract mode."""

    bs_entries: Path = Field(
        ...,
        description="Path to the input JSON or JSONL file containing BioSample entries.",
        examples=["data/bs_entries.json", "data/bs_entries.jsonl"],
    )
    prompt: Path = Field(
        ...,
        description="Path to the prompt file in YAML format.",
        examples=["prompt/prompt_extract.yml"],
    )
    format: Path | None = Field(
        None,
        description="Path to the JSON schema file for the output format.",
        examples=["format/cell_line.schema.json"],
    )
    model: str = "llama3.1:70b"
    thinking: bool | None = None
    max_entries: int | None = None
    run_name: str | None = None
    resume: bool = False
    batch_size: int = Field(..., gt=0)


class SelectConfigField(BaseModel):
    ontology_file: Path | None = Field(
        None,
        description="Path to the ontology OWL file or TSV file.",
        examples=["ontology/cellosaurus.owl", "ontology/cellosaurus.tsv"],
    )
    prompt_description: str | None = Field(
        None,
        description="Description to be included in the prompt for select mode.",
    )
    ontology_filter: dict[str, str] | None = Field(
        None,
        description="Filter criteria for ontology terms, e.g., {'hasDbXref': 'NCBI_TaxID:9606'}.",
        examples=[{"hasDbXref": "NCBI_TaxID:9606"}],
    )
    value_type: Literal["string", "array"] = Field(
        "string",
        description="Expected value type for the selected field.",
        examples=["string", "array"],
    )


class SelectConfig(BaseModel):
    fields: dict[str, SelectConfigField] = Field(
        ...,
        description="Configuration for each field to be selected using the ontology. The key is the field name to be extracted.",
    )


class CliSelectArgs(BaseModel):
    """Command-line arguments for the bsllmner2 CLI select mode."""

    bs_entries: Path = Field(
        ...,
        description="Path to the input JSON or JSONL file containing BioSample entries.",
        examples=["data/bs_entries.json", "data/bs_entries.jsonl"],
    )
    mapping: Path | None = Field(
        None,
        description="Path to the mapping file in TSV format.",
        examples=["mapping/mapping.tsv"],
    )
    model: str = "llama3.1:70b"
    thinking: bool | None = None
    max_entries: int | None = None
    run_name: str | None = None
    resume: bool = False
    batch_size: int = Field(..., gt=0)

    select_config: Path = Field(
        ...,
        description="Path to the select configuration file in JSON format.",
        examples=["config/select_config.json"],
    )
    include_reasoning: bool = True


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


BsEntries = list[dict[str, Any]]


class MappingValue(BaseModel):
    experiment_type: str
    extraction_answer: str | None
    mapping_answer_id: str | None
    mapping_answer_label: str | None


Mapping = dict[str, MappingValue]  # key: bs_entry accession


class WfInput(BaseModel):
    bs_entries: BsEntries
    prompt: list[Prompt] = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    thinking: bool | None = None
    format: JsonSchemaValue | None = None
    config: Config
    cli_args: CliExtractArgs | CliSelectArgs | None = None


LlmOutputValue = dict[str, Any] | list[Any] | None


class LlmOutput(BaseModel):
    accession: str
    output: LlmOutputValue = None
    output_full: str | None = None
    characteristics: dict[str, Any] | None = None
    taxId: int | str | None = None
    chat_response: ChatResponse


SelectFieldResults = dict[str, SearchResult | None]


class SelectResult(BaseModel):
    accession: str
    extract_output: LlmOutputValue = None
    # field -> value -> List[SearchResult]
    search_results: dict[str, dict[str, list[SearchResult]]] = Field(default_factory=dict)
    text2term_results: dict[str, dict[str, list[SearchResult]]] = Field(default_factory=dict)
    llm_chat_response: dict[str, dict[str, ChatResponse | None]] = Field(default_factory=dict)
    results: dict[str, SelectFieldResults | str | list[str] | None] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    correct: int = 0
    total: int = 0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class RunMetadata(BaseModel):
    run_name: str
    model: str
    thinking: bool | None = None
    username: str | None = None
    start_time: str
    end_time: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    processing_time: float | None = None
    matched_entries: int | None = None
    total_entries: int | None = None
    accuracy: float | None = None
    completed_count: int | None = None


class ErrorInfo(BaseModel):
    type: str
    message: str
    traceback: str


class ErrorLog(BaseModel):
    timestamp: str
    error: ErrorInfo


class Result(BaseModel):
    input: WfInput
    output: list[LlmOutput] = []
    run_metadata: RunMetadata
    error_log: ErrorLog | None = None
