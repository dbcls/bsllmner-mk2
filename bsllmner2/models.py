from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ollama import ChatResponse
from pydantic import BaseModel, Field, computed_field

RunStatus = Literal["running", "completed", "failed"]

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


# === CLI args models ===


class CliCommonArgs(BaseModel):
    """Common command-line arguments shared between extract and select modes."""

    bs_entries: Path = Field(
        ...,
        description="Path to the input JSON or JSONL file containing BioSample entries.",
        examples=["data/bs_entries.json", "data/bs_entries.jsonl"],
    )
    model: str = "llama3.1:70b"
    thinking: bool = False
    max_entries: int | None = None
    run_name: str | None = None
    num_ctx: int = 4096
    resume: bool = False
    batch_size: int = Field(..., gt=0)


class CliExtractArgs(CliCommonArgs):
    """Command-line arguments for the bsllmner2 CLI extract mode."""

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


class CliSelectArgs(CliCommonArgs):
    """Command-line arguments for the bsllmner2 CLI select mode."""

    mapping: Path | None = Field(
        None,
        description="Path to the mapping file in TSV format.",
        examples=["mapping/mapping.tsv"],
    )
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


class EvaluationMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    total: int = 0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def correct(self) -> int:
        return self.tp + self.tn


class ErrorInfo(BaseModel):
    type: str
    message: str
    traceback: str


class ErrorLog(BaseModel):
    timestamp: datetime
    error: ErrorInfo


class LlmTimingFields(BaseModel):
    total_duration: int = 0
    load_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    prompt_eval_count: int = 0


def llm_timing_from_chat_response(resp: ChatResponse) -> LlmTimingFields:
    return LlmTimingFields(
        total_duration=getattr(resp, "total_duration", 0) or 0,
        load_duration=getattr(resp, "load_duration", 0) or 0,
        eval_count=getattr(resp, "eval_count", 0) or 0,
        eval_duration=getattr(resp, "eval_duration", 0) or 0,
        prompt_eval_count=getattr(resp, "prompt_eval_count", 0) or 0,
    )


class ExtractEntry(BaseModel):
    accession: str
    extracted: dict[str, Any] | list[Any] | None = None
    raw_output: str | None = None
    llm_timing: LlmTimingFields = Field(default_factory=LlmTimingFields)


class ResolvedValue(BaseModel):
    value: str
    term_id: str | None = None
    term_uri: str | None = None
    label: str | None = None
    exact_match: bool | None = Field(None, description="Whether the value was an exact match in the ontology")
    reasoning: str | None = None


FieldCandidates = dict[str, list[SearchResult]]


class SelectEntry(BaseModel):
    extract: ExtractEntry
    search_results: dict[str, FieldCandidates] = Field(default_factory=dict)
    text2term_results: dict[str, FieldCandidates] = Field(default_factory=dict)
    select_timings: dict[str, dict[str, LlmTimingFields]] = Field(default_factory=dict)
    results: dict[str, list[ResolvedValue]] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    run_name: str
    model: str
    thinking: bool = False
    start_time: datetime
    end_time: datetime | None = None
    status: RunStatus = "running"
    processing_time_sec: float | None = None
    total_entries: int | None = None


# === Timing models (moved from benchmark.py) ===


class LlmTimingSummary(BaseModel):
    """Timing statistics aggregated from multiple LLM calls."""

    call_count: int
    total_duration_sec: float
    # Latency is computed as total_duration minus load_duration, in seconds.
    mean_latency_sec: float
    p50_latency_sec: float
    p95_latency_sec: float
    p99_latency_sec: float
    # tokens/sec = eval_count / (eval_duration / 1e9)
    mean_tokens_per_sec: float | None
    p50_tokens_per_sec: float | None
    p95_tokens_per_sec: float | None
    # load_duration (warm-up impact analysis)
    mean_load_duration_sec: float
    max_load_duration_sec: float
    # token counts
    total_prompt_tokens: int
    total_eval_tokens: int


class StageTimings(BaseModel):
    """Wall-clock timings per stage for a single batch."""

    batch_idx: int
    batch_size: int
    ner_sec: float | None = None
    ontology_search_sec: float | None = None
    text2term_sec: float | None = None
    llm_select_sec: float | None = None
    resume_write_sec: float | None = None


class DiskIoTimings(BaseModel):
    """Timing data for disk I/O operations."""

    index_cache_load_sec: list[float] = Field(default_factory=list)
    index_cache_save_sec: list[float] = Field(default_factory=list)
    index_build_from_file_sec: list[float] = Field(default_factory=list)
    resume_write_sec: list[float] = Field(default_factory=list)


class PerformanceSummary(BaseModel):
    """Performance data embedded in result files."""

    total_input_entries: int
    completed_count: int
    total_wall_sec: float | None = None
    stage_timings: list[StageTimings] = Field(default_factory=list)
    ner_llm_timing: LlmTimingSummary | None = None
    select_llm_timing: LlmTimingSummary | None = None
    disk_io: DiskIoTimings = Field(default_factory=DiskIoTimings)


class ExtractResult(BaseModel):
    entries: list[ExtractEntry]
    run_metadata: RunMetadata
    performance: PerformanceSummary | None = None
    errors: list[ErrorLog] = []


class SelectResult(BaseModel):
    entries: list[SelectEntry]
    run_metadata: RunMetadata
    evaluation: EvaluationMetrics | None = None
    performance: PerformanceSummary | None = None
    errors: list[ErrorLog] = []
