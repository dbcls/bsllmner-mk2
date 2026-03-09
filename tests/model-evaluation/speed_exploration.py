"""Speed exploration: ternary search for optimal OLLAMA_NUM_PARALLEL per model.

Run on the host machine (not inside docker). Controls Ollama restarts via
``docker compose`` and runs ``bsllmner2_select`` inside the app container via
``docker exec``.
"""

from __future__ import annotations

import argparse
import json
import logging
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
BS_ENTRIES_PATH = "tests/data/eval_biosample.json"
SELECT_CONFIG_PATH = "tests/data/eval_select_config.json"
MAPPING_PATH = "tests/data/eval_gold_standard.tsv"
RESULTS_DIR_IN_CONTAINER = "results/speed_exploration"
HEALTH_CHECK_INTERVAL_SEC = 2
HEALTH_CHECK_TIMEOUT_SEC = 120
WARMUP_TIMEOUT_SEC = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOG.debug("$ %s", shlex.join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def _sanitize_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def restart_ollama(num_parallel: int, project_dir: str, model: str) -> None:
    """Restart the Ollama container with a new OLLAMA_NUM_PARALLEL value."""
    LOG.info("Restarting Ollama with OLLAMA_NUM_PARALLEL=%d ...", num_parallel)
    env_override = f"OLLAMA_NUM_PARALLEL={num_parallel}"
    _run(
        ["docker", "compose", "up", "-d", "ollama"],
        timeout=60,
    )
    # We need to set the env var via docker compose. The compose file uses a
    # hard-coded value, so we stop, update, and restart.
    _run(["docker", "compose", "stop", "ollama"], timeout=60)
    # Re-create with env override
    _run(
        ["env", env_override, "docker", "compose", "up", "-d", "ollama"],
        timeout=60,
    )
    _wait_for_health(project_dir)
    _warmup(project_dir, model)


def _wait_for_health(project_dir: str) -> None:
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


def _warmup(project_dir: str, model: str) -> None:
    """Send a dummy request to warm up the Ollama server with the target model."""
    LOG.info("Sending warm-up request for model %s ...", model)
    result = _run(
        [
            "docker", "exec", APP_CONTAINER,
            "curl", "-sf",
            "-X", "POST",
            "http://bsllmner-mk2-ollama:11434/api/generate",
            "-d", json.dumps({"model": model, "prompt": "hello", "stream": False}),
        ],
        timeout=WARMUP_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        LOG.warning("Warm-up request failed (non-fatal): %s", result.stderr[:200])


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------


def create_subset_file(
    project_dir: str,
    subset_size: int,
    seed: int,
) -> str:
    """Create a subset of eval_biosample.json inside the container. Returns container path."""
    # Read the full file via docker exec
    result = _run(
        ["docker", "exec", APP_CONTAINER, "python", "-c", f"""
import json, random, pathlib
data = json.loads(pathlib.Path("{BS_ENTRIES_PATH}").read_text())
random.seed({seed})
subset = random.sample(data, min({subset_size}, len(data)))
out = pathlib.Path("{RESULTS_DIR_IN_CONTAINER}/subset_{subset_size}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(subset, ensure_ascii=False))
print(str(out))
"""],
        timeout=30,
    )
    path = result.stdout.strip()
    LOG.info("Created subset file: %s (%d entries)", path, subset_size)
    return path


# ---------------------------------------------------------------------------
# Run bsllmner2_select
# ---------------------------------------------------------------------------


def run_select(
    model: str,
    num_ctx: int,
    subset_file: str,
    project_dir: str,
    run_name: str,
) -> dict[str, Any] | None:
    """Run bsllmner2_select inside the app container and return the result JSON."""
    cmd = [
        "docker", "exec", APP_CONTAINER,
        "uv", "run", "bsllmner2_select",
        "--bs-entries", subset_file,
        "--model", model,
        "--select-config", SELECT_CONFIG_PATH,
        "--mapping", MAPPING_PATH,
        "--num-ctx", str(num_ctx),
        "--no-reasoning",
        "--run-name", run_name,
        "--batch-size", "9999",
    ]
    result = _run(cmd, timeout=4 * 3600, check=False)

    if result.returncode != 0:
        LOG.error("bsllmner2_select failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return None

    # Find and read the result file
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
        timeout=30,
    )
    try:
        return json.loads(cat_result.stdout)
    except json.JSONDecodeError:
        LOG.error("Failed to parse result JSON from %s", result_path)
        return None


def extract_throughput(result_json: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract NER and Select mean_tokens_per_sec from a result JSON.

    Returns (ner_tps, select_tps).
    """
    perf = result_json.get("performance")
    if perf is None:
        return None, None
    ner_timing = perf.get("ner_llm_timing")
    select_timing = perf.get("select_llm_timing")
    ner_tps = ner_timing.get("mean_tokens_per_sec") if ner_timing else None
    select_tps = select_timing.get("mean_tokens_per_sec") if select_timing else None
    return ner_tps, select_tps


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(
    model: str,
    num_parallel: int,
    num_ctx: int,
    subset_file: str,
    project_dir: str,
    runs: int,
    current_num_parallel: list[int],
) -> dict[str, Any]:
    """Restart Ollama (if needed), run N times, return median throughput info."""
    if not current_num_parallel or current_num_parallel[0] != num_parallel:
        restart_ollama(num_parallel, project_dir, model)
        current_num_parallel.clear()
        current_num_parallel.append(num_parallel)

    ner_tps_list: list[float] = []
    select_tps_list: list[float] = []

    for run_idx in range(runs):
        run_name = f"speed_{_sanitize_model_name(model)}_{num_ctx}_{num_parallel}_r{run_idx}"
        LOG.info(
            "  Run %d/%d: model=%s num_parallel=%d num_ctx=%d",
            run_idx + 1, runs, model, num_parallel, num_ctx,
        )
        result_json = run_select(model, num_ctx, subset_file, project_dir, run_name)
        if result_json is None:
            # Retry once
            LOG.warning("  Retrying run %d ...", run_idx + 1)
            result_json = run_select(model, num_ctx, subset_file, project_dir, run_name + "_retry")
        if result_json is None:
            LOG.warning("  Run %d failed after retry, treating as throughput=0", run_idx + 1)
            ner_tps_list.append(0.0)
            select_tps_list.append(0.0)
            continue

        ner_tps, select_tps = extract_throughput(result_json)
        ner_tps_list.append(ner_tps if ner_tps is not None else 0.0)
        select_tps_list.append(select_tps if select_tps is not None else 0.0)

    median_ner_tps = statistics.median(ner_tps_list)
    median_select_tps = statistics.median(select_tps_list)
    total_throughput = num_parallel * median_select_tps

    LOG.info(
        "  Result: num_parallel=%d, median_select_tps=%.1f, total_throughput=%.1f",
        num_parallel, median_select_tps, total_throughput,
    )

    return {
        "num_parallel": num_parallel,
        "num_ctx": num_ctx,
        "ner_tps_runs": ner_tps_list,
        "select_tps_runs": select_tps_list,
        "median_ner_tps": median_ner_tps,
        "median_select_tps": median_select_tps,
        "total_throughput": total_throughput,
    }


# ---------------------------------------------------------------------------
# Ternary search
# ---------------------------------------------------------------------------


def ternary_search(
    model: str,
    num_ctx: int,
    subset_file: str,
    project_dir: str,
    runs: int,
    search_lo: int,
    search_hi: int,
) -> dict[str, Any]:
    """Find the NUM_PARALLEL that maximizes total throughput via ternary search."""
    LOG.info("=== Ternary search: model=%s, num_ctx=%d, range=[%d, %d] ===", model, num_ctx, search_lo, search_hi)

    lo = search_lo
    hi = search_hi
    search_history: list[dict[str, Any]] = []
    cache: dict[int, dict[str, Any]] = {}
    current_num_parallel: list[int] = []  # mutable tracker

    def _measure_cached(n: int) -> dict[str, Any]:
        if n in cache:
            return cache[n]
        result = measure(model, n, num_ctx, subset_file, project_dir, runs, current_num_parallel)
        cache[n] = result
        search_history.append(result)
        return result

    while hi - lo >= 3:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        LOG.info("  lo=%d, hi=%d, m1=%d, m2=%d", lo, hi, m1, m2)

        r1 = _measure_cached(m1)
        r2 = _measure_cached(m2)

        if r1["total_throughput"] < r2["total_throughput"]:
            lo = m1 + 1
        else:
            hi = m2 - 1

    # Measure remaining candidates
    best_result: dict[str, Any] | None = None
    for n in range(lo, hi + 1):
        result = _measure_cached(n)
        if best_result is None or result["total_throughput"] > best_result["total_throughput"]:
            best_result = result

    assert best_result is not None

    return {
        "model": model,
        "num_ctx": num_ctx,
        "best_num_parallel": best_result["num_parallel"],
        "best_total_throughput": best_result["total_throughput"],
        "best_median_select_tps": best_result["median_select_tps"],
        "best_median_ner_tps": best_result["median_ner_tps"],
        "search_history": search_history,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validation(
    model: str,
    num_parallel: int,
    num_ctx: int,
    project_dir: str,
    runs: int,
) -> dict[str, Any]:
    """Run full 600 entries with optimal parameters."""
    LOG.info("=== Validation: model=%s, num_parallel=%d, num_ctx=%d ===", model, num_parallel, num_ctx)

    current_num_parallel: list[int] = []
    if not current_num_parallel or current_num_parallel[0] != num_parallel:
        restart_ollama(num_parallel, project_dir, model)
        current_num_parallel.clear()
        current_num_parallel.append(num_parallel)

    select_tps_list: list[float] = []
    ner_tps_list: list[float] = []
    wall_sec_list: list[float] = []
    collected_metrics: list[dict[str, Any]] = []

    for run_idx in range(runs):
        run_name = f"validate_{_sanitize_model_name(model)}_{num_ctx}_{num_parallel}_r{run_idx}"
        LOG.info("  Validation run %d/%d", run_idx + 1, runs)

        result_json = run_select(model, num_ctx, BS_ENTRIES_PATH, project_dir, run_name)
        if result_json is None:
            LOG.warning("  Validation run %d failed", run_idx + 1)
            continue

        ner_tps, select_tps = extract_throughput(result_json)
        perf = result_json.get("performance", {})
        wall_sec = perf.get("total_wall_sec")

        if select_tps is not None:
            select_tps_list.append(select_tps)
        if ner_tps is not None:
            ner_tps_list.append(ner_tps)
        if wall_sec is not None:
            wall_sec_list.append(wall_sec)

        collected_metrics.append(result_json.get("evaluation", {}))

    def _median_iqr(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        med = statistics.median(values)
        if len(values) >= 4:
            q = statistics.quantiles(values, n=4)
            iqr = q[2] - q[0]
        else:
            iqr = None
        return med, iqr

    median_select_tps, iqr_select_tps = _median_iqr(select_tps_list)
    median_ner_tps, iqr_ner_tps = _median_iqr(ner_tps_list)
    median_wall_sec, iqr_wall_sec = _median_iqr(wall_sec_list)

    # Accuracy metrics (deterministic, take from first successful run)
    first_metrics = collected_metrics[0] if collected_metrics else {}

    total_throughput: float | None = None
    if median_select_tps is not None:
        total_throughput = num_parallel * median_select_tps

    return {
        "model": model,
        "num_parallel": num_parallel,
        "num_ctx": num_ctx,
        "total_throughput": total_throughput,
        "median_select_tps": median_select_tps,
        "iqr_select_tps": iqr_select_tps,
        "median_ner_tps": median_ner_tps,
        "iqr_ner_tps": iqr_ner_tps,
        "median_wall_sec": median_wall_sec,
        "iqr_wall_sec": iqr_wall_sec,
        "f1": first_metrics.get("f1"),
        "precision": first_metrics.get("precision"),
        "recall": first_metrics.get("recall"),
        "accuracy": first_metrics.get("accuracy"),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_exploration_result(result: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = _sanitize_model_name(result["model"])
    path = output_dir / f"exploration_{model_safe}_{result['num_ctx']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    LOG.info("Wrote exploration result: %s", path)
    return path


def write_validation_result(result: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = _sanitize_model_name(result["model"])
    path = output_dir / f"validation_{model_safe}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    LOG.info("Wrote validation result: %s", path)
    return path


def write_summary_tsv(exploration_results: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.tsv"
    header = [
        "model", "num_ctx", "best_num_parallel",
        "best_total_throughput", "best_median_select_tps", "best_median_ner_tps",
    ]
    lines = ["\t".join(header)]
    for r in exploration_results:
        lines.append("\t".join(str(r.get(h, "")) for h in header))
    path.write_text("\n".join(lines) + "\n")
    LOG.info("Wrote summary: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speed exploration: ternary search for optimal OLLAMA_NUM_PARALLEL per model.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to explore.")

    parser.add_argument(
        "--num-ctx", type=int, nargs="+", default=[4096],
        help="Context length(s) to explore (default: 4096).",
    )
    parser.add_argument(
        "--project-dir", type=str, default=".",
        help="Project directory (default: current directory).",
    )
    parser.add_argument(
        "--subset-size", type=int, default=200,
        help="Number of entries for exploration subset (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subset sampling (default: 42).",
    )
    parser.add_argument(
        "--runs-per-point", type=int, default=3,
        help="Number of runs per measurement point (default: 3).",
    )
    parser.add_argument(
        "--search-lo", type=int, default=4,
        help="Lower bound for NUM_PARALLEL search (default: 4).",
    )
    parser.add_argument(
        "--search-hi", type=int, default=64,
        help="Upper bound for NUM_PARALLEL search (default: 64).",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the validation phase (full 600 entries).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/model-evaluation/results"),
        help="Output directory for results (default: tests/model-evaluation/results).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)

    model: str = args.model
    num_ctx_values: list[int] = args.num_ctx
    output_dir: Path = args.output_dir
    runs: int = args.runs_per_point

    LOG.info("Model: %s", model)
    LOG.info("num_ctx values: %s", num_ctx_values)

    # Create subset file for exploration
    subset_file = create_subset_file(args.project_dir, args.subset_size, args.seed)

    exploration_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for num_ctx in num_ctx_values:
        exploration_result = ternary_search(
            model=model,
            num_ctx=num_ctx,
            subset_file=subset_file,
            project_dir=args.project_dir,
            runs=runs,
            search_lo=args.search_lo,
            search_hi=args.search_hi,
        )
        write_exploration_result(exploration_result, output_dir)
        exploration_results.append(exploration_result)

        if best is None or exploration_result["best_total_throughput"] > best["best_total_throughput"]:
            best = exploration_result

    write_summary_tsv(exploration_results, output_dir)

    # Validation with the best (NUM_PARALLEL, num_ctx)
    if not args.skip_validation and best is not None:
        validation_result = run_validation(
            model=model,
            num_parallel=best["best_num_parallel"],
            num_ctx=best["num_ctx"],
            project_dir=args.project_dir,
            runs=runs,
        )
        write_validation_result(validation_result, output_dir)

    LOG.info("Done.")


if __name__ == "__main__":
    main()
