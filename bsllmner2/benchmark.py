"""Benchmark utilities: timing, models, I/O, and logging."""

import contextlib
import json
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from ollama import ChatResponse
from pydantic import BaseModel, Field

from bsllmner2.config import LOGGER, RESULT_DIR

# === Benchmark models ===

BENCHMARK_DIR = RESULT_DIR.joinpath("benchmarks")


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


class BenchmarkSummary(BaseModel):
    """Per-run benchmark summary (saved as JSON file)."""

    run_name: str
    model: str
    thinking: bool | None = None
    total_entries: int
    completed_count: int
    total_wall_sec: float | None = None
    stage_timings: list[StageTimings] = Field(default_factory=list)
    ner_llm_timing: LlmTimingSummary | None = None
    select_llm_timing: LlmTimingSummary | None = None
    disk_io: DiskIoTimings = Field(default_factory=DiskIoTimings)
    select_accuracy: float | None = None
    select_precision: float | None = None
    select_recall: float | None = None
    select_f1: float | None = None
    select_matched_entries: int | None = None


# === Timer utilities ===


@dataclass
class StageTimer:
    """Timer for measuring elapsed wall-clock time of a pipeline stage."""

    name: str
    _start: float = field(default=0.0, init=False)
    elapsed_sec: float = 0.0

    def __post_init__(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed_sec = time.perf_counter() - self._start

        return self.elapsed_sec


@contextlib.contextmanager
def stage_timer(name: str) -> Generator[StageTimer, None, None]:
    """Context manager that measures elapsed wall-clock time."""
    timer = StageTimer(name=name)
    try:
        yield timer
    finally:
        timer.stop()


def compute_percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of *values*. Returns 0.0 for empty list."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Linear interpolation (same method as numpy default)
    rank = (p / 100) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    fraction = rank - lower

    if upper >= n:
        return sorted_values[-1]

    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def aggregate_llm_timings(responses: list[ChatResponse]) -> LlmTimingSummary:
    """Compute LlmTimingSummary from a list of ChatResponse objects.

    - latency = (total_duration - load_duration) / 1e9
    - tokens_per_sec = eval_count / (eval_duration / 1e9)
    - load_duration_sec = load_duration / 1e9
    """
    if not responses:
        return LlmTimingSummary(
            call_count=0,
            total_duration_sec=0.0,
            mean_latency_sec=0.0,
            p50_latency_sec=0.0,
            p95_latency_sec=0.0,
            p99_latency_sec=0.0,
            mean_tokens_per_sec=None,
            p50_tokens_per_sec=None,
            p95_tokens_per_sec=None,
            mean_load_duration_sec=0.0,
            max_load_duration_sec=0.0,
            total_prompt_tokens=0,
            total_eval_tokens=0,
        )

    latencies: list[float] = []
    tokens_per_sec_list: list[float] = []
    load_durations: list[float] = []
    total_duration_sum = 0.0
    total_prompt_tokens = 0
    total_eval_tokens = 0

    for resp in responses:
        td = getattr(resp, "total_duration", 0) or 0
        ld = getattr(resp, "load_duration", 0) or 0
        ec = getattr(resp, "eval_count", 0) or 0
        ed = getattr(resp, "eval_duration", 0) or 0
        pec = getattr(resp, "prompt_eval_count", 0) or 0

        total_duration_sum += td / 1e9
        latencies.append((td - ld) / 1e9)
        load_durations.append(ld / 1e9)
        total_prompt_tokens += pec
        total_eval_tokens += ec

        if ed > 0:
            tokens_per_sec_list.append(ec / (ed / 1e9))

    call_count = len(responses)
    mean_latency = sum(latencies) / call_count
    mean_load = sum(load_durations) / call_count

    mean_tps: float | None = None
    p50_tps: float | None = None
    p95_tps: float | None = None
    if tokens_per_sec_list:
        mean_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
        p50_tps = compute_percentile(tokens_per_sec_list, 50)
        p95_tps = compute_percentile(tokens_per_sec_list, 95)

    return LlmTimingSummary(
        call_count=call_count,
        total_duration_sec=total_duration_sum,
        mean_latency_sec=mean_latency,
        p50_latency_sec=compute_percentile(latencies, 50),
        p95_latency_sec=compute_percentile(latencies, 95),
        p99_latency_sec=compute_percentile(latencies, 99),
        mean_tokens_per_sec=mean_tps,
        p50_tokens_per_sec=p50_tps,
        p95_tokens_per_sec=p95_tps,
        mean_load_duration_sec=mean_load,
        max_load_duration_sec=max(load_durations),
        total_prompt_tokens=total_prompt_tokens,
        total_eval_tokens=total_eval_tokens,
    )


# === I/O ===


def dump_benchmark(summary: BenchmarkSummary, run_name: str) -> Path:
    """Write a BenchmarkSummary as JSON to the benchmarks directory."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    bench_file = BENCHMARK_DIR.joinpath(f"{run_name}_benchmark.json")
    with bench_file.open("w", encoding="utf-8") as f:
        json.dump(summary.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    return bench_file


# === Logging ===


def log_benchmark_summary(summary: BenchmarkSummary) -> None:
    """Log a human-readable benchmark summary to the logger."""
    LOGGER.info("=== Benchmark Summary ===")
    LOGGER.info("  total_wall_sec: %.2f", summary.total_wall_sec or 0.0)
    LOGGER.info("  total_entries: %d, completed: %d", summary.total_entries, summary.completed_count)
    if summary.ner_llm_timing:
        t = summary.ner_llm_timing
        LOGGER.info(
            "  NER LLM: %d calls, mean_latency=%.3fs, mean_tokens/sec=%s",
            t.call_count,
            t.mean_latency_sec,
            f"{t.mean_tokens_per_sec:.1f}" if t.mean_tokens_per_sec is not None else "N/A",
        )
    if summary.select_llm_timing:
        t = summary.select_llm_timing
        LOGGER.info(
            "  Select LLM: %d calls, mean_latency=%.3fs, mean_tokens/sec=%s",
            t.call_count,
            t.mean_latency_sec,
            f"{t.mean_tokens_per_sec:.1f}" if t.mean_tokens_per_sec is not None else "N/A",
        )
    if summary.select_accuracy is not None:
        LOGGER.info(
            "  Select: accuracy=%.2f%%, precision=%.2f%%, recall=%.2f%%, f1=%.2f%%",
            summary.select_accuracy,
            summary.select_precision or 0.0,
            summary.select_recall or 0.0,
            summary.select_f1 or 0.0,
        )
