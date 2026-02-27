import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from bsllmner2.schema import SelectResult

HERE: Path = Path(__file__).parent.resolve()


def oneline(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def simplify_prop_uri(prop_uri: str | None) -> str:
    if not prop_uri:
        return ""
    if "#" in prop_uri:
        return prop_uri.split("#")[-1]
    return prop_uri.rstrip("/").split("/")[-1]


def dump_candidate(cand: Any) -> str:
    d: dict[str, Any] = cand.model_dump(exclude_none=True)
    d.pop("term_uri", None)
    d["prop_uri"] = simplify_prop_uri(d.get("prop_uri"))
    return oneline(d)


def normalize_values(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def has_term_id(cands: list[Any], term_id: str) -> bool:
    return any(getattr(c, "term_id", None) == term_id for c in cands)


def infer_source(sr: SelectResult, field: str, value: str, term_id: str) -> str:
    if sr.llm_chat_response.get(field, {}).get(value) is not None:
        return "llm"
    if has_term_id(sr.search_results.get(field, {}).get(value, []), term_id):
        return "ontology search"
    if has_term_id(sr.text2term_results.get(field, {}).get(value, []), term_id):
        return "text2term"
    return "unknown"


def print_extracted(sr: SelectResult) -> dict[str, list[str]]:
    print("- Extracted:")
    extracted: dict[str, list[str]] = {}

    if not isinstance(sr.extract_output, dict):
        return extracted

    for field, raw in sr.extract_output.items():
        values = normalize_values(raw)
        if not values:
            continue
        extracted[field] = values
        print(f"  - {field}:")
        for v in values:
            print(f"    - {v}")

    return extracted


def print_stage(
    title: str,
    stage: Mapping[str, Mapping[str, list[Any]]],
    extracted: Mapping[str, list[str]],
) -> None:
    print(f"- {title}:")
    for field, values in extracted.items():
        print(f"  - {field}:")
        for value in values:
            cands = stage.get(field, {}).get(value, [])
            if not cands:
                print(f"    - value: {value}: NOT FOUND")
                continue
            print(f"    - value: {value}:")
            for cand in cands:
                print(f"      - {dump_candidate(cand)}")


def print_text2term(
    sr: SelectResult,
    extracted: Mapping[str, list[str]],
) -> None:
    print("- Text2Term:")
    for field, values in extracted.items():
        print(f"  - {field}:")
        for value in values:
            t2t_items = sr.text2term_results.get(field, {}).get(value, [])
            if t2t_items:
                print(f"    - value: {value}:")
                for cand in t2t_items:
                    print(f"      - {dump_candidate(cand)}")
                continue

            # Ontology search 成功済みなら DO NOTHING
            prev = sr.search_results.get(field, {}).get(value, [])
            if prev:
                print(f"    - value: {value}: DO NOTHING")
            else:
                print(f"    - value: {value}: NOT FOUND")


def print_llm(
    sr: SelectResult,
    extracted: Mapping[str, list[str]],
) -> None:
    print("- LLM Result:")
    for field, values in extracted.items():
        print(f"  - {field}:")
        for value in values:
            llm_done = sr.llm_chat_response.get(field, {}).get(value) is not None
            if not llm_done:
                print(f"    - value: {value}: DO NOTHING")
                continue

            picked = sr.results.get(field, {}).get(value)
            if picked is not None:
                print(f"    - value: {value}:")
                print(f"      - {dump_candidate(picked)}")
            else:
                print(f"    - value: {value}: NOT FOUND")


def print_final(
    sr: SelectResult,
    extracted: Mapping[str, list[str]],
) -> None:
    print("- 最終結果:")
    for field, values in extracted.items():
        print(f"  - {field}:")
        for value in values:
            picked = sr.results.get(field, {}).get(value)
            if picked is None:
                print(f"    - {value}: NOT FOUND")
                continue

            label = picked.label or picked.value
            term_id = picked.term_id
            source = infer_source(sr, field, value, term_id)
            print(f"    - {value} → {label} ({term_id}, by {source})")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: print_select_result.py <select_result_file>")
        sys.exit(1)

    p = Path(sys.argv[1])
    data = p.read_text(encoding="utf-8")
    select_results: list[SelectResult] = TypeAdapter(list[SelectResult]).validate_json(data)

    for sr in select_results:
        print()
        print(f"=== {sr.accession} ===")

        extracted = print_extracted(sr)
        if not extracted:
            continue

        print_stage("Ontology Search", sr.search_results, extracted)
        print_text2term(sr, extracted)
        print_llm(sr, extracted)
        print_final(sr, extracted)


if __name__ == "__main__":
    main()
