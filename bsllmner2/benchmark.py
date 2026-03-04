"""Benchmark utilities: timing, aggregation, and logging."""

import contextlib
import time
from collections.abc import Generator
from dataclasses import dataclass, field

from ollama import ChatResponse

from bsllmner2.config import LOGGER
from bsllmner2.models import (
    EvaluationMetrics,
    LlmTimingFields,
    LlmTimingSummary,
    PerformanceSummary,
    llm_timing_from_chat_response,
)

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


# === Aggregation ===


def aggregate_from_timing_fields(fields: list[LlmTimingFields]) -> LlmTimingSummary:
    """Compute LlmTimingSummary from a list of LlmTimingFields objects."""
    if not fields:
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

    for f in fields:
        total_duration_sum += f.total_duration / 1e9
        latencies.append((f.total_duration - f.load_duration) / 1e9)
        load_durations.append(f.load_duration / 1e9)
        total_prompt_tokens += f.prompt_eval_count
        total_eval_tokens += f.eval_count

        if f.eval_duration > 0:
            tokens_per_sec_list.append(f.eval_count / (f.eval_duration / 1e9))

    call_count = len(fields)
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


def aggregate_llm_timings(responses: list[ChatResponse]) -> LlmTimingSummary:
    """Compute LlmTimingSummary from a list of ChatResponse objects."""
    timing_fields = [llm_timing_from_chat_response(r) for r in responses]
    return aggregate_from_timing_fields(timing_fields)


# === Logging ===


def log_performance_summary(
    summary: PerformanceSummary,
    evaluation: EvaluationMetrics | None = None,
) -> None:
    """Log a human-readable performance summary to the logger."""
    LOGGER.info("=== Performance Summary ===")
    LOGGER.info("  total_wall_sec: %.2f", summary.total_wall_sec or 0.0)
    LOGGER.info("  total_input_entries: %d, completed: %d", summary.total_input_entries, summary.completed_count)
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
    if evaluation is not None and evaluation.accuracy is not None:
        LOGGER.info(
            "  Select: accuracy=%.2f%%, precision=%.2f%%, recall=%.2f%%, f1=%.2f%%",
            evaluation.accuracy * 100,
            (evaluation.precision or 0.0) * 100,
            (evaluation.recall or 0.0) * 100,
            (evaluation.f1 or 0.0) * 100,
        )
