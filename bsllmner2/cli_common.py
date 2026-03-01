"""Common CLI utilities shared between extract and select modes."""

import argparse
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from bsllmner2.config import LOGGER, RESUME_BATCH_SIZE, default_config

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
        "--mapping",
        type=Path,
        help="Path to the mapping file in TSV format.",
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
        metavar="BOOL",
        help="Enable or disable thinking mode for the LLM. Use 'true' to enable, 'false' to disable.",
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
        "--with-metrics",
        action="store_true",
        help="Enable collection of metrics during processing.",
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


def validate_common_args(parsed_args: argparse.Namespace) -> None:
    """Validate common arguments and raise errors if invalid."""
    if not parsed_args.bs_entries.exists():
        raise FileNotFoundError(f"BioSample entries file {parsed_args.bs_entries} does not exist.")


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
