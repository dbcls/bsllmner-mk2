"""Inspect a SelectResult JSON produced by bsllmner2_select.

Subcommands:
  summary <file> [--top-nf N] [--json]
      Run-wide overview: mapping rate per field, NOT_FOUND top values, LLM
      timing, evaluation metrics.
  show    <file> (--accession ID | --unmapped-only [--limit N]) [--json]
      Entry-level view: extracted values, resolved labels, source (exact /
      llm / text2term), unmapped values.
  find    <file> --field F --value V [--json]
      Cross-run lookup: all entries that extracted ``V`` for field ``F``,
      split into mapped / unmapped.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from bsllmner2.models import (
    LlmTimingSummary,
    ResolvedValue,
    SelectEntry,
    SelectResult,
)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_select_result(path: Path) -> SelectResult:
    """Parse a SelectResult JSON from disk.

    Older runs serialized ``run_metadata.thinking`` as ``null``; the current
    model requires a bool. Coerce in-place so this debug tool stays usable
    against historical result files.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    rm = raw.get("run_metadata")
    if isinstance(rm, dict) and rm.get("thinking") is None:
        rm["thinking"] = False
    return TypeAdapter(SelectResult).validate_python(raw)


# --------------------------------------------------------------------------
# Extracted value normalization & source inference
# --------------------------------------------------------------------------


def normalize_extracted_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x) != ""]
    s = str(raw)
    return [s] if s else []


def extracted_map(entry: SelectEntry) -> dict[str, list[str]]:
    """Return a dict {field: [extracted values]} from one entry.

    ``extracted`` is pydantic-typed as ``dict | list | None``; only dict
    payloads carry per-field extractions used by select mode.
    """
    out: dict[str, list[str]] = {}
    extracted = entry.extract.extracted
    if not isinstance(extracted, dict):
        return out
    for field, raw in extracted.items():
        values = normalize_extracted_values(raw)
        if values:
            out[field] = values
    return out


def resolved_index(entry: SelectEntry, field: str) -> dict[str, ResolvedValue]:
    """Index resolved values for a field by their source ``value`` string."""
    return {rv.value: rv for rv in entry.results.get(field, [])}


def infer_source(entry: SelectEntry, field: str, resolved: ResolvedValue) -> str:
    """Classify how the resolved term was produced.

    ``exact`` overrides everything else even if the LLM was also consulted,
    because the pipeline preferentially commits exact matches.
    """
    if resolved.exact_match:
        return "exact"
    if resolved.value in entry.select_timings.get(field, {}):
        return "llm"
    return "text2term"


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


INDENT = "  "


def format_wall_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s ({seconds:.1f}s)"
    if m:
        return f"{m}m{s:02d}s ({seconds:.1f}s)"
    return f"{s}s ({seconds:.1f}s)"


def format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return f"{numerator / denominator * 100:.1f}%"


def resolved_label(resolved: ResolvedValue) -> str:
    label = resolved.label or resolved.value
    term_id = resolved.term_id or "-"
    return f"{label} ({term_id})"


def emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def emit_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def collect_field_names(result: SelectResult) -> list[str]:
    """All field names referenced by any entry, preserving insertion order."""
    seen: dict[str, None] = {}
    for entry in result.entries:
        ext = entry.extract.extracted
        if isinstance(ext, dict):
            for field in ext:
                seen.setdefault(field, None)
        for field in entry.results:
            seen.setdefault(field, None)
    return list(seen.keys())


def compute_mapping_stats(
    result: SelectResult,
    fields: list[str],
) -> dict[str, dict[str, int]]:
    """Per-field {extracted, mapped} counts aggregated across all entries.

    "extracted" is the number of string values extracted from BioSample
    characteristics (value-level, not entry-level). "mapped" is the subset
    of those values that produced a resolved term in ``results``.
    """
    stats: dict[str, dict[str, int]] = {f: {"extracted": 0, "mapped": 0} for f in fields}
    for entry in result.entries:
        em = extracted_map(entry)
        for field, values in em.items():
            if field not in stats:
                continue
            stats[field]["extracted"] += len(values)
            resolved = resolved_index(entry, field)
            for value in values:
                if value in resolved:
                    stats[field]["mapped"] += 1
    return stats


def compute_not_found_top(
    result: SelectResult,
    fields: list[str],
    top_n: int,
) -> dict[str, list[tuple[str, int]]]:
    """Per-field Counter.most_common of extracted values that never resolved."""
    counters: dict[str, Counter[str]] = {f: Counter() for f in fields}
    for entry in result.entries:
        em = extracted_map(entry)
        for field, values in em.items():
            if field not in counters:
                continue
            resolved = resolved_index(entry, field)
            for value in values:
                if value not in resolved:
                    counters[field][value] += 1
    return {f: counters[f].most_common(top_n) for f in fields}


def llm_timing_summary_dict(ts: LlmTimingSummary | None) -> dict[str, Any] | None:
    if ts is None:
        return None
    return {
        "call_count": ts.call_count,
        "mean_latency_sec": ts.mean_latency_sec,
        "mean_tokens_per_sec": ts.mean_tokens_per_sec,
        "total_prompt_tokens": ts.total_prompt_tokens,
        "total_eval_tokens": ts.total_eval_tokens,
    }


def dispatch_summary(result: SelectResult, top_nf: int, json_out: bool) -> None:
    fields = collect_field_names(result)
    mapping = compute_mapping_stats(result, fields)
    not_found = compute_not_found_top(result, fields, top_nf)

    rm = result.run_metadata
    perf = result.performance
    wall = perf.total_wall_sec if perf is not None else rm.processing_time_sec

    run_info = {
        "name": rm.run_name,
        "model": rm.model,
        "thinking": rm.thinking,
        "status": rm.status,
        "wall_sec": wall,
        "total_entries": len(result.entries),
        "errors": len(result.errors),
    }

    ner = perf.ner_llm_timing if perf is not None else None
    sel = perf.select_llm_timing if perf is not None else None
    evaluation = result.evaluation.model_dump() if result.evaluation is not None else None

    if json_out:
        emit_json(
            {
                "run": run_info,
                "mapping_rate": {
                    f: {
                        "extracted": mapping[f]["extracted"],
                        "mapped": mapping[f]["mapped"],
                        "rate": (mapping[f]["mapped"] / mapping[f]["extracted"] if mapping[f]["extracted"] else None),
                    }
                    for f in fields
                },
                "not_found_top": {f: [{"value": v, "count": c} for v, c in not_found[f]] for f in fields},
                "llm_timing": {
                    "ner": llm_timing_summary_dict(ner),
                    "select": llm_timing_summary_dict(sel),
                },
                "evaluation": evaluation,
            }
        )
        return

    lines: list[str] = []
    lines.append("Run")
    lines.append(f"{INDENT}name:     {run_info['name']}")
    lines.append(f"{INDENT}model:    {run_info['model']}")
    thinking_repr = "true" if rm.thinking else "false" if rm.thinking is not None else "null"
    lines.append(f"{INDENT}thinking: {thinking_repr}")
    lines.append(f"{INDENT}status:   {run_info['status']}")
    lines.append(f"{INDENT}wall:     {format_wall_time(wall)}")
    lines.append(
        f"{INDENT}entries:  {len(result.entries)}"
        + (f"/{rm.total_entries}" if rm.total_entries is not None else "")
        + f"  errors: {len(result.errors)}",
    )

    lines.append("")
    lines.append("Mapping rate")
    field_width = max((len(f) for f in fields), default=5)
    header = f"{INDENT}{'field'.ljust(field_width)}  Extracted   Mapped    Rate"
    lines.append(header)
    for f in fields:
        ex = mapping[f]["extracted"]
        mp = mapping[f]["mapped"]
        lines.append(
            f"{INDENT}{f.ljust(field_width)}  {ex:9d}  {mp:7d}  {format_percent(mp, ex):>6}",
        )

    lines.append("")
    lines.append(f"NOT_FOUND top {top_nf} per field")
    for f in fields:
        items = not_found[f]
        if not items:
            lines.append(f"{INDENT}{f}: (none)")
            continue
        lines.append(f"{INDENT}{f}:")
        for value, count in items:
            lines.append(f"{INDENT}{INDENT}{value} ({count})")

    lines.append("")
    lines.append("LLM timing")
    for label, ts in (("NER", ner), ("Select", sel)):
        if ts is None:
            lines.append(f"{INDENT}{label}: (not recorded)")
            continue
        tps = f"tok/s {ts.mean_tokens_per_sec:.2f}" if ts.mean_tokens_per_sec is not None else "tok/s -"
        lines.append(
            f"{INDENT}{label}: {ts.call_count} calls, mean {ts.mean_latency_sec:.1f}s, {tps}",
        )

    if evaluation is not None:
        lines.append("")
        lines.append("Evaluation")
        for key in ("accuracy", "precision", "recall", "f1"):
            val = evaluation.get(key)
            lines.append(f"{INDENT}{key:<9} {val if val is not None else '-'}")

    emit_lines(lines)


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------


def entry_detail_dict(entry: SelectEntry) -> dict[str, Any]:
    em = extracted_map(entry)
    details: dict[str, list[dict[str, Any]]] = {}
    unmapped: list[dict[str, str]] = []
    for field, values in em.items():
        resolved = resolved_index(entry, field)
        field_rows: list[dict[str, Any]] = []
        for value in values:
            rv = resolved.get(value)
            if rv is None:
                field_rows.append(
                    {
                        "value": value,
                        "mapped": False,
                        "source": None,
                        "resolved": None,
                    }
                )
                unmapped.append({"field": field, "value": value})
            else:
                field_rows.append(
                    {
                        "value": value,
                        "mapped": True,
                        "source": infer_source(entry, field, rv),
                        "resolved": {
                            "term_id": rv.term_id,
                            "label": rv.label,
                            "exact_match": rv.exact_match,
                            "reasoning": rv.reasoning,
                        },
                    }
                )
        details[field] = field_rows
    return {
        "accession": entry.extract.accession,
        "fields": details,
        "unmapped": unmapped,
    }


def render_entry_detail(detail: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"Entry {detail['accession']}")
    fields: dict[str, list[dict[str, Any]]] = detail["fields"]
    if not fields:
        lines.append(f"{INDENT}(no extracted values)")
        return lines
    for field, rows in fields.items():
        lines.append(f"{INDENT}{field}:")
        for row in rows:
            if row["mapped"]:
                rv = row["resolved"]
                label = rv["label"] or row["value"]
                term_id = rv["term_id"] or "-"
                source = row["source"]
                lines.append(
                    f"{INDENT}{INDENT}{row['value']} -> {label} ({term_id}) [{source}]",
                )
            else:
                lines.append(f"{INDENT}{INDENT}{row['value']} -> NOT_FOUND")
    return lines


def dispatch_show_accession(result: SelectResult, accession: str, json_out: bool) -> int:
    hits = [e for e in result.entries if e.extract.accession == accession]
    if not hits:
        print(f"No entry found for accession: {accession}", file=sys.stderr)
        return 1
    detail = entry_detail_dict(hits[0])
    if json_out:
        emit_json(detail)
    else:
        emit_lines(render_entry_detail(detail))
    return 0


def dispatch_show_unmapped(result: SelectResult, limit: int | None, json_out: bool) -> int:
    selected: list[dict[str, Any]] = []
    for entry in result.entries:
        detail = entry_detail_dict(entry)
        if detail["unmapped"]:
            selected.append(detail)
            if limit is not None and len(selected) >= limit:
                break

    if json_out:
        emit_json(
            {
                "count": len(selected),
                "entries": selected,
            }
        )
        return 0

    lines: list[str] = []
    lines.append(f"Entries with unmapped values: {len(selected)}")
    if limit is not None:
        lines.append(f"{INDENT}(limit: {limit})")
    lines.append("")
    if not selected:
        lines.append("(no unmapped values)")
    for detail in selected:
        lines.extend(render_entry_detail(detail))
        lines.append("")
    emit_lines(lines)
    return 0


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


def dispatch_find(
    result: SelectResult,
    field: str,
    value: str,
    json_out: bool,
) -> None:
    matched: list[dict[str, Any]] = []
    mapped_count = 0
    unmapped_count = 0

    for entry in result.entries:
        values = extracted_map(entry).get(field, [])
        if value not in values:
            continue
        resolved = resolved_index(entry, field).get(value)
        if resolved is None:
            unmapped_count += 1
            matched.append(
                {
                    "accession": entry.extract.accession,
                    "mapped": False,
                    "source": None,
                    "resolved": None,
                }
            )
        else:
            mapped_count += 1
            matched.append(
                {
                    "accession": entry.extract.accession,
                    "mapped": True,
                    "source": infer_source(entry, field, resolved),
                    "resolved": {
                        "term_id": resolved.term_id,
                        "label": resolved.label,
                        "exact_match": resolved.exact_match,
                        "reasoning": resolved.reasoning,
                    },
                }
            )

    if json_out:
        emit_json(
            {
                "field": field,
                "value": value,
                "total": len(matched),
                "mapped": mapped_count,
                "unmapped": unmapped_count,
                "entries": matched,
            }
        )
        return

    lines: list[str] = []
    lines.append(f"Matches for {field} = {value!r}")
    lines.append(f"{INDENT}total:    {len(matched)}")
    lines.append(f"{INDENT}mapped:   {mapped_count}")
    lines.append(f"{INDENT}unmapped: {unmapped_count}")
    if not matched:
        emit_lines(lines)
        return

    mapped_rows = [m for m in matched if m["mapped"]]
    unmapped_rows = [m for m in matched if not m["mapped"]]

    if mapped_rows:
        lines.append("")
        lines.append(f"Mapped ({len(mapped_rows)}):")
        for row in mapped_rows:
            rv = row["resolved"]
            label = rv["label"] or value
            term_id = rv["term_id"] or "-"
            lines.append(
                f"{INDENT}{row['accession']} -> {label} ({term_id}) [{row['source']}]",
            )

    if unmapped_rows:
        lines.append("")
        lines.append(f"Unmapped ({len(unmapped_rows)}):")
        lines.extend(f"{INDENT}{row['accession']}" for row in unmapped_rows)

    emit_lines(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_select_result.py",
        description="Inspect a SelectResult JSON produced by bsllmner2_select.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="Run-wide overview")
    p_summary.add_argument("file", type=Path)
    p_summary.add_argument(
        "--top-nf",
        type=int,
        default=10,
        help="Number of NOT_FOUND values to show per field (default: 10).",
    )
    p_summary.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")

    p_show = sub.add_parser("show", help="Entry-level detail")
    p_show.add_argument("file", type=Path)
    g = p_show.add_mutually_exclusive_group(required=True)
    g.add_argument("--accession", help="Show a specific BioSample accession.")
    g.add_argument(
        "--unmapped-only",
        action="store_true",
        help="Show only entries that contain at least one unmapped value.",
    )
    p_show.add_argument("--limit", type=int, default=None, help="Cap entries shown with --unmapped-only.")
    p_show.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")

    p_find = sub.add_parser("find", help="Locate entries by (field, value)")
    p_find.add_argument("file", type=Path)
    p_find.add_argument("--field", required=True)
    p_find.add_argument("--value", required=True)
    p_find.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = load_select_result(args.file)

    if args.command == "summary":
        dispatch_summary(result, top_nf=args.top_nf, json_out=args.json)
        return 0
    if args.command == "show":
        if args.accession is not None:
            return dispatch_show_accession(result, args.accession, json_out=args.json)
        return dispatch_show_unmapped(result, limit=args.limit, json_out=args.json)
    if args.command == "find":
        dispatch_find(result, args.field, args.value, json_out=args.json)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
