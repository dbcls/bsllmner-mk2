"""Common CLI utilities shared between extract and select modes."""

import argparse
import contextlib
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from bsllmner2.config import DEFAULT_NUM_CTX, LOGGER, RESUME_BATCH_SIZE, Config, default_config, get_config
from bsllmner2.errors import Bsllmner2Error
from bsllmner2.io import load_bs_entries
from bsllmner2.models import BsEntries, RunMetadata, RunStatus
from bsllmner2.pipeline import get_now

T = TypeVar("T")


def str_to_bool(v: str) -> bool:
    """Convert string to boolean for argparse.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid boolean string.

    """
    lower = v.strip().lower()
    if lower in ("true", "1", "yes", "on"):
        return True
    if lower in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'. Use 'true' or 'false'.")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between extract and select modes."""
    parser.add_argument(
        "--bs-entries",
        type=Path,
        required=True,
        help="Path to the input JSON or JSONL file containing BioSample entries.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:70b",
        help="LLM model to use for NER.",
    )
    parser.add_argument(
        "--thinking",
        type=str_to_bool,
        default=False,
        metavar="BOOL",
        help="Enable or disable thinking mode for the LLM (default: false). Use 'true' to enable, 'false' to disable.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=-1,
        help="Process only the first N entries. Default is -1 (all entries).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help=f"Host URL for the Ollama server (default: {default_config.ollama_host}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose logging.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run for identification purposes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last incomplete run if possible.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=RESUME_BATCH_SIZE,
        help=f"Number of entries to process in each batch (default: {RESUME_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"Context length for Ollama (default: {DEFAULT_NUM_CTX}).",
    )


def validate_common_args(parser: argparse.ArgumentParser, parsed_args: argparse.Namespace) -> None:
    """Validate common arguments and raise errors if invalid."""
    if not parsed_args.bs_entries.exists():
        parser.error(f"BioSample entries file {parsed_args.bs_entries} does not exist.")


def build_config(parsed_args: argparse.Namespace) -> Config:
    """Build application Config from parsed CLI arguments."""
    config = get_config()
    if parsed_args.ollama_host is not None:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug
    return config


def generate_run_name(model: str, run_name: str | None) -> tuple[str, datetime]:
    """Generate a run name and start timestamp.

    Returns:
        Tuple of (resolved_run_name, start_time).

    """
    start_time = get_now()
    time_str = start_time.strftime("%Y%m%d_%H%M%S")
    if run_name:
        return run_name, start_time
    return f"{model}_{time_str}", start_time


def load_and_trim_entries(bs_entries_path: Path, max_entries: int | None) -> BsEntries:
    """Load BioSample entries and optionally trim to max_entries."""
    bs_entries = load_bs_entries(bs_entries_path)
    if max_entries is not None:
        bs_entries = bs_entries[:max_entries]
    return bs_entries


def build_run_metadata(
    run_name: str,
    model: str,
    thinking: bool,
    start_time: datetime,
    end_time: datetime | None,
    status: RunStatus,
) -> RunMetadata:
    """Build RunMetadata from common parameters."""
    return RunMetadata(
        run_name=run_name,
        model=model,
        thinking=thinking,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


@dataclass
class BatchInfo:
    """Information about a batch being processed."""

    batch_idx: int
    total_batches: int
    start_idx: int
    end_idx: int
    entries: list[dict[str, Any]]


async def process_batches(
    entries: list[dict[str, Any]],
    batch_size: int,
    process_fn: Callable[[BatchInfo], Awaitable[T]],
    on_batch_complete: Callable[[int, T], None],
    log_prefix: str = "Processing",
) -> list[T]:
    """Process entries in batches.

    Args:
        entries: List of entries to process
        batch_size: Number of entries per batch
        process_fn: Async function to process each batch, receives BatchInfo
        on_batch_complete: Callback after each batch completes, receives (batch_idx, result)
        log_prefix: Prefix for log messages

    Returns:
        List of results from each batch

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not entries:
        return []

    total_batches = math.ceil(len(entries) / batch_size)
    results: list[T] = []

    LOGGER.info("%s %d entries in %d batches...", log_prefix, len(entries), total_batches)

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(entries))

        batch_info = BatchInfo(
            batch_idx=batch_idx,
            total_batches=total_batches,
            start_idx=start_idx,
            end_idx=end_idx,
            entries=entries[start_idx:end_idx],
        )

        LOGGER.info(
            "[BATCH %d/%d] entries from %d to %d",
            batch_idx + 1,
            total_batches,
            start_idx + 1,
            end_idx,
        )

        result = await process_fn(batch_info)
        results.append(result)

        on_batch_complete(batch_idx, result)

    return results


@dataclass
class _RunState:
    """Mutable state shared with the caller through ``run_with_lifecycle``."""

    end_time: datetime | None = None
    status: RunStatus = "running"


@contextlib.asynccontextmanager
async def run_with_lifecycle() -> AsyncIterator[_RunState]:
    """Shared try/except/finally lifecycle for CLI commands."""
    state = _RunState()
    try:
        yield state
        state.end_time = get_now()
        state.status = "completed"
    except Bsllmner2Error as e:
        LOGGER.error("Processing failed: %s", e)
        state.status = "failed"
        state.end_time = get_now()
    except Exception as e:
        LOGGER.error("Unexpected error during processing: %s", e, exc_info=True)
        state.status = "failed"
        state.end_time = get_now()
