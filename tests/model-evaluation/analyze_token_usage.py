"""Analyze token usage to determine appropriate num_ctx.

Runs ``bsllmner2_select`` once and analyzes the actual token counts from
the result JSON to recommend a ``num_ctx`` value (next power of 2 above
the observed maximum).

Run on the host machine (same as ``speed_exploration.py``).

Usage::

    python tests/model-evaluation/analyze_token_usage.py \
        --model qwen3:8b \
        --bs-entries tests/data/eval_biosample.json \
        --select-config tests/data/eval_select_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger(__name__)

APP_CONTAINER = "bsllmner-mk2-app"
DEFAULT_NUM_CTX = 4096
HEALTH_CHECK_INTERVAL_SEC = 2
HEALTH_CHECK_TIMEOUT_SEC = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOG.debug("$ %s", shlex.join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def _sanitize_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(n))


def _wait_for_health() -> None:
    """Poll Ollama health endpoint until it responds."""
    deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT_SEC
    while time.monotonic() < deadline:
        result = _run(
            [
                "docker", "exec", APP_CONTAINER,
                "curl", "-sf", "http://bsllmner-mk2-ollama:11434/",
            ],
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            LOG.info("Ollama is healthy")
            return
        time.sleep(HEALTH_CHECK_INTERVAL_SEC)
    raise TimeoutError("Ollama did not become healthy within timeout")


# ---------------------------------------------------------------------------
# Run bsllmner2_select
# ---------------------------------------------------------------------------


def _run_select(
    model: str,
    bs_entries: str,
    select_config: str,
    max_entries: int | None,
    num_ctx: int,
) -> dict[str, Any] | None:
    """Run bsllmner2_select and return the result JSON."""
    run_name = f"token_analysis_{_sanitize_model_name(model)}"
    cmd = [
        "docker", "exec", APP_CONTAINER,
        "uv", "run", "bsllmner2_select",
        "--bs-entries", bs_entries,
        "--model", model,
        "--select-config", select_config,
        "--num-ctx", str(num_ctx),
        "--no-reasoning",
        "--run-name", run_name,
        "--batch-size", "9999",
    ]
    if max_entries is not None:
        cmd.extend(["--max-entries", str(max_entries)])

    LOG.info("Running bsllmner2_select (num_ctx=%d) ...", num_ctx)
    result = _run(cmd, timeout=14400, check=False)

    if result.returncode != 0:
        LOG.error("bsllmner2_select failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return None

    find_result = _run(
        ["docker", "exec", APP_CONTAINER, "bash", "-c",
         f"ls -t bsllmner2-results/select/select_{run_name}*.json 2>/dev/null | head -1"],
        timeout=10,
        check=False,
    )
    result_path = find_result.stdout.strip()
    if not result_path:
        LOG.error("Could not find result file for run %s", run_name)
        return None

    cat_result = _run(
        ["docker", "exec", APP_CONTAINER, "cat", result_path],
        timeout=60,
    )
    try:
        return json.loads(cat_result.stdout)
    except json.JSONDecodeError:
        LOG.error("Failed to parse result JSON from %s", result_path)
        return None


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


def _extract_token_counts(result_json: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Extract per-call total token counts (prompt + output) for extract and select stages.

    Returns (extract_totals, select_totals).
    """
    extract_totals: list[int] = []
    select_totals: list[int] = []

    for entry in result_json.get("entries", []):
        # Extract stage
        llm_timing = entry.get("extract", {}).get("llm_timing", {})
        prompt_tokens = llm_timing.get("prompt_eval_count", 0)
        output_tokens = llm_timing.get("eval_count", 0)
        if prompt_tokens > 0 or output_tokens > 0:
            extract_totals.append(prompt_tokens + output_tokens)

        # Select stage
        select_timings = entry.get("select_timings", {})
        for field_timings in select_timings.values():
            if not isinstance(field_timings, dict):
                continue
            for timing in field_timings.values():
                if not isinstance(timing, dict):
                    continue
                p = timing.get("prompt_eval_count", 0)
                e = timing.get("eval_count", 0)
                if p > 0 or e > 0:
                    select_totals.append(p + e)

    return extract_totals, select_totals


def _compute_stats(values: list[int]) -> dict[str, int | float]:
    """Compute summary statistics for a list of token counts."""
    if not values:
        return {"calls": 0, "max": 0, "p99": 0, "p95": 0, "mean": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> int:
        idx = math.ceil(n * p / 100) - 1
        return sorted_vals[max(0, min(idx, n - 1))]

    return {
        "calls": n,
        "max": sorted_vals[-1],
        "p99": _percentile(99),
        "p95": _percentile(95),
        "mean": round(statistics.mean(values), 1),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_table(
    model: str,
    extract_stats: dict[str, int | float],
    select_stats: dict[str, int | float],
) -> str:
    """Format the analysis results as a readable table."""
    lines: list[str] = []
    lines.append(f"model: {model}")
    lines.append("")
    lines.append(f"{'Stage':<10} {'calls':>6} {'max':>8} {'p99':>8} {'p95':>8} {'mean':>8}")
    lines.append("-" * 50)
    for stage, stats in [("extract", extract_stats), ("select", select_stats)]:
        lines.append(
            f"{stage:<10} {stats['calls']:>6} {stats['max']:>8} "
            f"{stats['p99']:>8} {stats['p95']:>8} {stats['mean']:>8}"
        )

    overall_max = max(extract_stats["max"], select_stats["max"])
    recommended = _next_power_of_2(int(overall_max)) if overall_max > 0 else 4096
    lines.append("")
    lines.append(f"Overall max tokens: {overall_max}")
    lines.append(f"Recommended num_ctx: {recommended}  (next power of 2 above max)")
    return "\n".join(lines)


def _write_result_json(
    model: str,
    extract_stats: dict[str, int | float],
    select_stats: dict[str, int | float],
    output_dir: Path,
) -> Path:
    """Write analysis result as JSON."""
    overall_max = max(extract_stats["max"], select_stats["max"])
    recommended = _next_power_of_2(int(overall_max)) if overall_max > 0 else 4096

    result = {
        "model": model,
        "extract": extract_stats,
        "select": select_stats,
        "overall_max_tokens": overall_max,
        "recommended_num_ctx": recommended,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"token_usage_{_sanitize_model_name(model)}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    LOG.info("Wrote result: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze token usage to determine appropriate num_ctx.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to analyze.")
    parser.add_argument(
        "--bs-entries", type=str, required=True,
        help="Path to BioSample entries JSON (container path).",
    )
    parser.add_argument(
        "--select-config", type=str, required=True,
        help="Path to select config JSON (container path).",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=DEFAULT_NUM_CTX,
        help=f"Context length for Ollama (default: {DEFAULT_NUM_CTX}).",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Limit number of entries to process (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/model-evaluation/results"),
        help="Output directory for results (default: tests/model-evaluation/results).",
    )
    args = parser.parse_args()

    _wait_for_health()

    result_json = _run_select(
        model=args.model,
        bs_entries=args.bs_entries,
        select_config=args.select_config,
        max_entries=args.max_entries,
        num_ctx=args.num_ctx,
    )
    if result_json is None:
        LOG.error("Failed to run bsllmner2_select. Cannot analyze token usage.")
        return

    extract_totals, select_totals = _extract_token_counts(result_json)
    extract_stats = _compute_stats(extract_totals)
    select_stats = _compute_stats(select_totals)

    table = _format_table(args.model, extract_stats, select_stats)
    print()
    print(table)
    print()

    _write_result_json(args.model, extract_stats, select_stats, args.output_dir)


if __name__ == "__main__":
    main()
