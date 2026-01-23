#!/usr/bin/env python3
"""
Extract で抽出されたが Select でマッピングできなかった値をリストアップする。

Usage:
    python scripts/list_unmapped.py <select_result_file> [--json]

Options:
    --json  JSON 形式で出力（デフォルトはテキスト形式）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import TypeAdapter

from bsllmner2.schema import SelectResult


def normalize_values(v: Any) -> List[str]:
    """Extract した値を文字列リストに正規化する。"""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def find_unmapped(select_results: List[SelectResult]) -> List[Dict[str, Any]]:
    """
    Select 結果から、extract で抽出されたがマッピングできなかった値を抽出する。

    Returns:
        List of dicts with keys: accession, field, value
    """
    unmapped: List[Dict[str, Any]] = []

    for sr in select_results:
        if not isinstance(sr.extract_output, dict):
            continue

        for field, raw_value in sr.extract_output.items():
            values = normalize_values(raw_value)
            for value in values:
                result = sr.results.get(field, {}).get(value)
                if result is None:
                    unmapped.append({
                        "accession": sr.accession,
                        "field": field,
                        "value": value,
                    })

    return unmapped


def print_text(unmapped: List[Dict[str, Any]]) -> None:
    """テキスト形式で出力する。"""
    if not unmapped:
        print("All extracted values were successfully mapped.")
        return

    print(f"Found {len(unmapped)} unmapped value(s):\n")

    # accession でグループ化して表示
    by_accession: Dict[str, List[Dict[str, Any]]] = {}
    for item in unmapped:
        acc = item["accession"]
        by_accession.setdefault(acc, []).append(item)

    for acc, items in by_accession.items():
        print(f"=== {acc} ===")
        for item in items:
            print(f"  - {item['field']}: {item['value']}")
        print()


def print_json(unmapped: List[Dict[str, Any]]) -> None:
    """JSON 形式で出力する。"""
    print(json.dumps(unmapped, ensure_ascii=False, indent=2))


def main() -> None:
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        sys.exit(0)

    json_output = "--json" in args
    file_args = [a for a in args if not a.startswith("-")]

    if len(file_args) != 1:
        print("Error: Exactly one select result file is required.")
        print(__doc__)
        sys.exit(1)

    p = Path(file_args[0])
    if not p.exists():
        print(f"Error: File not found: {p}")
        sys.exit(1)

    data = p.read_text(encoding="utf-8")
    select_results: List[SelectResult] = TypeAdapter(
        List[SelectResult]
    ).validate_json(data)

    unmapped = find_unmapped(select_results)

    if json_output:
        print_json(unmapped)
    else:
        print_text(unmapped)


if __name__ == "__main__":
    main()
