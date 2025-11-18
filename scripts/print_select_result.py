#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import TypeAdapter

from bsllmner2.schema import SelectResult

HERE: Path = Path(__file__).parent.resolve()


def oneline(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def simplify_prop_uri(prop_uri: Optional[str]) -> str:
    if not prop_uri:
        return ""
    if "#" in prop_uri:
        return prop_uri.split("#")[-1]
    return prop_uri.rstrip("/").split("/")[-1]


def print_stage_items(title: str, items_by_field: Mapping[str, List[Any]], ordered_fields: List[str]) -> None:
    print(f"- {title}:")
    for field in ordered_fields:
        items = items_by_field.get(field) or []
        if not items:
            print(f"  - {field}: NOT FOUND")
            continue
        print(f"  - {field}:")
        for cand in items:
            d: Dict[str, Any] = cand.model_dump(exclude_none=True)
            d.pop("term_uri", None)
            d["prop_uri"] = simplify_prop_uri(d.get("prop_uri"))
            print(f"    - {oneline(d)}")


def print_text2term_with_do_nothing(sr: SelectResult, ordered_fields: List[str]) -> None:
    print("- Text2Term:")
    for field in ordered_fields:
        text2term_items = sr.text2term_results.get(field) or []
        if text2term_items:
            print(f"  - {field}:")
            for cand in text2term_items:
                d = cand.model_dump(exclude_none=True)
                d.pop("term_uri", None)
                d["prop_uri"] = simplify_prop_uri(d.get("prop_uri"))
                print(f"    - {oneline(d)}")
            continue

        # Ontology search がすでに成功していれば DO NOTHING
        prev_items = sr.search_results.get(field) or []
        if prev_items:
            print(f"  - {field}: DO NOTHING")
        else:
            print(f"  - {field}: NOT FOUND")


def print_llm_result_with_flag(sr: SelectResult, ordered_fields: List[str]) -> None:
    print("- LLM Result:")
    for field in ordered_fields:
        llm_done = sr.llm_chat_response.get(field) is not None
        if not llm_done:
            print(f"  - {field}: DO NOTHING")
            continue

        picked = sr.results.get(field)
        if picked is not None:
            d = picked.model_dump(exclude_none=True)
            d.pop("term_uri", None)
            d["prop_uri"] = simplify_prop_uri(d.get("prop_uri"))
            print(f"  - {field}:")
            print(f"    - {oneline(d)}")
        else:
            print(f"  - {field}: NOT FOUND")


def _has_term_id(cands: List[Any], term_id: str) -> bool:
    return any(getattr(c, "term_id", None) == term_id for c in cands)


def infer_source(sr: SelectResult, field: str, chosen_term_id: str) -> str:
    if sr.llm_chat_response.get(field) is not None:
        return "llm"
    if _has_term_id(sr.search_results.get(field) or [], chosen_term_id):
        return "ontology search"
    if _has_term_id(sr.text2term_results.get(field) or [], chosen_term_id):
        return "text2term"
    return "unknown"


def main() -> None:
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: print_select_result.py <select_result_file>")
        sys.exit(1)
    p = Path(args[0])
    data = p.read_text(encoding="utf-8")
    select_results: List[SelectResult] = TypeAdapter(List[SelectResult]).validate_json(data)

    for sr in select_results:
        print("")
        print(f"=== {sr.accession} ===")

        # Extracted
        print("- Extracted:")
        extracted_used: List[str] = []
        if isinstance(sr.extract_output, dict):
            for k, v in sr.extract_output.items():
                print(f"  - {k}: {v}")
                if v is not None:
                    extracted_used.append(str(k))

        # None は以降の処理から除外
        print_stage_items("Ontology Search", sr.search_results, extracted_used)
        print_text2term_with_do_nothing(sr, extracted_used)
        print_llm_result_with_flag(sr, extracted_used)

        # 最終結果
        print("- 最終結果:")
        for field in extracted_used:
            picked = sr.results.get(field)
            if picked is None:
                print(f"  - {field}: NOT FOUND")
                continue
            label = picked.label or picked.value
            term_id = picked.term_id
            source = infer_source(sr, field, term_id)
            print(f"  - {field}: {label} ({term_id}, by {source})")


if __name__ == "__main__":
    main()
