"""Cross-model comparison: unified summary table and Pareto plot.

Reads ``validation_*.json`` files produced by ``speed_exploration.py``
and generates:

1. ``validation_summary.tsv`` — all models side-by-side (speed + accuracy)
2. ``pareto_plot.png`` — scatter plot of total throughput vs F1 with Pareto frontier

Usage::

    # Generate summary + plot from validation results
    python tests/model-evaluation/model_comparison.py \
        --results-dir tests/model-evaluation/results

    # TSV only (no matplotlib required)
    python tests/model-evaluation/model_comparison.py \
        --results-dir tests/model-evaluation/results \
        --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load validation results
# ---------------------------------------------------------------------------


def load_validation_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all validation_*.json files from the results directory."""
    files = sorted(results_dir.glob("validation_*.json"))
    if not files:
        LOG.warning("No validation_*.json files found in %s", results_dir)
        return []

    results: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            results.append(data)
            LOG.info("Loaded %s (model=%s)", f.name, data.get("model", "?"))
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning("Failed to load %s: %s", f, e)

    return results


# ---------------------------------------------------------------------------
# Validation summary TSV
# ---------------------------------------------------------------------------

SUMMARY_HEADER = [
    "model", "num_parallel", "num_ctx", "total_throughput",
    "median_select_tps", "median_ner_tps", "median_wall_sec",
    "f1", "precision", "recall", "accuracy",
]


def write_validation_summary_tsv(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a unified validation summary TSV with speed + accuracy for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_summary.tsv"
    lines = ["\t".join(SUMMARY_HEADER)]
    for r in results:
        row: list[str] = []
        for h in SUMMARY_HEADER:
            v = r.get(h)
            if v is None:
                row.append("")
            elif isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")
    LOG.info("Wrote validation summary: %s", path)
    return path


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------


def compute_pareto_frontier(
    points: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Compute Pareto frontier (maximize both x and y).

    Returns points on the frontier sorted by x.
    """
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float, str]] = []
    max_y = float("-inf")
    for x, y, label in reversed(sorted_pts):
        if y >= max_y:
            frontier.append((x, y, label))
            max_y = y
    frontier.reverse()
    return frontier


def generate_pareto_plot(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create scatter plot with Pareto frontier (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.error("matplotlib is required for plotting. Install with: uv pip install matplotlib")
        return

    points: list[tuple[float, float, str]] = []
    for r in results:
        throughput = r.get("total_throughput")
        f1 = r.get("f1")
        model = r.get("model", "?")
        if throughput is not None and f1 is not None:
            points.append((float(throughput), float(f1), str(model)))

    if not points:
        LOG.warning("No valid data points (need both total_throughput and f1).")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(xs, ys, s=80, zorder=5, color="#2563eb", edgecolors="white", linewidth=0.8)

    for x, y, label in points:
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            color="#374151",
        )

    frontier = compute_pareto_frontier(points)
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, "--", color="#dc2626", linewidth=1.5, alpha=0.7, label="Pareto frontier")
        ax.scatter(
            fx, fy, s=120, facecolors="none", edgecolors="#dc2626",
            linewidth=1.5, zorder=6, label="Pareto optimal",
        )

    ax.set_xlabel("Total Throughput (NUM_PARALLEL x tokens/sec)", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Model Comparison: Throughput vs Accuracy", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved Pareto plot: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model comparison: generate summary table and Pareto plot from validation results.",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("tests/model-evaluation/results"),
        help="Directory containing validation_*.json files (default: tests/model-evaluation/results).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as --results-dir).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip Pareto plot generation (no matplotlib dependency).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir if args.output_dir is not None else args.results_dir

    results = load_validation_results(args.results_dir)
    if not results:
        LOG.error("No validation results found. Run speed_exploration.py first.")
        return

    write_validation_summary_tsv(results, output_dir)

    if not args.no_plot:
        plot_path = output_dir / "pareto_plot.png"
        generate_pareto_plot(results, plot_path)

    LOG.info("Done. %d models compared.", len(results))


if __name__ == "__main__":
    main()
