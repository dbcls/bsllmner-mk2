"""Reselect mode: repeat Stage 2 and Stage 3 for the unresolved values of an earlier select run.

The source run's extracted values are taken as they are. For each entry and each
selected field, the values that carry no term in ``results[field]`` go through
:func:`bsllmner2.select.select` again, restricted to those fields and values, and
the outcome is merged back into a copy of the source entry. Nothing else in the
entry changes.
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ollama import ChatResponse

from bsllmner2.llm import LlmBackend
from bsllmner2.models import (
    BsEntries,
    ErrorLog,
    ExtractEntry,
    OntologyIndex,
    SelectConfig,
    SelectEntry,
)
from bsllmner2.select import SearchMemo, SelectStageTimings, select

FieldValues = dict[str, list[str]]


def fields_with_ontology(select_config: SelectConfig, requested: Iterable[str] | None) -> list[str]:
    """Return the fields to reselect, in select-config order.

    Without ``requested``, every field that has an ``ontology_file`` is taken. A
    requested field that the config does not know, or that has no
    ``ontology_file``, is an error: there is nothing to search for it.
    """
    with_ontology = [name for name, cfg in select_config.fields.items() if cfg.ontology_file is not None]
    if requested is None:
        return with_ontology
    wanted = list(dict.fromkeys(requested))
    unknown = [f for f in wanted if f not in select_config.fields]
    if unknown:
        raise ValueError(f"fields not in the select config: {unknown}")
    without = [f for f in wanted if f not in with_ontology]
    if without:
        raise ValueError(f"fields without an ontology_file: {without}")
    return [f for f in with_ontology if f in wanted]


def restrict_select_config(select_config: SelectConfig, fields: Iterable[str]) -> SelectConfig:
    """Return a select config that holds only ``fields``, in the original order."""
    keep = set(fields)
    return SelectConfig(fields={name: cfg for name, cfg in select_config.fields.items() if name in keep})


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def unresolved_values(entry: SelectEntry, fields: Iterable[str]) -> FieldValues:
    """Per field, the extracted values that carry no term in ``results[field]``.

    Values keep the order of the extracted list and appear once each. Fields
    without unresolved values are absent from the result.
    """
    extracted = entry.extract.extracted
    if extracted is None:
        return {}
    out: FieldValues = {}
    for field in fields:
        resolved = {rv.value for rv in entry.results.get(field, [])}
        pending = [v for v in dict.fromkeys(_string_values(extracted.get(field))) if v not in resolved]
        if pending:
            out[field] = pending
    return out


def extract_entry_for_reselect(entry: SelectEntry, targets: FieldValues) -> ExtractEntry:
    """Build an ExtractEntry that holds only the target values, shaped as the source extracted them.

    A field extracted as a single string stays a string, a list stays a list, so
    the restricted entry reads like the source for the fields it keeps.
    """
    source = entry.extract.extracted or {}
    extracted: dict[str, object] = {}
    for field, values in targets.items():
        extracted[field] = values[0] if isinstance(source.get(field), str) else list(values)
    return ExtractEntry(accession=entry.extract.accession, extracted=extracted)


def merge_reselected(original: SelectEntry, reselected: SelectEntry, targets: FieldValues) -> SelectEntry:
    """Return a copy of ``original`` with the target values' candidates and results taken from ``reselected``.

    For every target ``(field, value)``: the candidate lists and the timing and
    ambiguity records are replaced by what the new pass produced (a record the new
    pass did not produce is removed), and the resolved values the new pass picked
    for that value are appended to ``results[field]``. Other values and fields are
    left as they are.
    """
    merged = original.model_copy(deep=True)
    for field, values in targets.items():
        search = merged.search_results.setdefault(field, {})
        t2t = merged.text2term_results.setdefault(field, {})
        timings = merged.select_timings.setdefault(field, {})
        ambiguous = merged.ambiguous_fields.setdefault(field, {})
        new_search = reselected.search_results.get(field, {})
        new_t2t = reselected.text2term_results.get(field, {})
        new_timings = reselected.select_timings.get(field, {})
        new_ambiguous = reselected.ambiguous_fields.get(field, {})
        new_resolved = [rv for rv in reselected.results.get(field, []) if rv.value in values]
        for value in values:
            search[value] = list(new_search.get(value, []))
            t2t[value] = list(new_t2t.get(value, []))
            if value in new_timings:
                timings[value] = new_timings[value]
            else:
                timings.pop(value, None)
            if value in new_ambiguous:
                ambiguous[value] = list(new_ambiguous[value])
            else:
                ambiguous.pop(value, None)
        if new_resolved:
            merged.results.setdefault(field, []).extend(rv.model_copy(deep=True) for rv in new_resolved)
    return merged


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def reselect_entries(
    backend: LlmBackend,
    bs_entries: BsEntries,
    model: str,
    entries: list[SelectEntry],
    targets_by_accession: dict[str, FieldValues],
    select_config: SelectConfig,
    thinking: bool = False,
    include_reasoning: bool = True,
    index_map: dict[Path, OntologyIndex] | None = None,
    text2term_cache_folder: Path | None = None,
    num_ctx: int | None = None,
    search_memo: SearchMemo | None = None,
    t2t_memo: SearchMemo | None = None,
) -> tuple[list[SelectEntry], list[ChatResponse], SelectStageTimings, list[ErrorLog]]:
    """Run Stage 2 and Stage 3 for the target values of ``entries`` and merge the outcome.

    ``select_config`` must already be restricted to the fields being reselected.
    Entries without targets are returned unchanged.
    """
    extract_outputs = [
        extract_entry_for_reselect(e, targets_by_accession[e.extract.accession])
        for e in entries
        if targets_by_accession.get(e.extract.accession)
    ]
    accessions = {x.accession for x in extract_outputs}
    batch_bs_entries = [b for b in bs_entries if b.get("accession") in accessions]

    reselected, responses, timings, errors = await select(
        backend,
        batch_bs_entries,
        model,
        extract_outputs,
        select_config,
        thinking,
        include_reasoning=include_reasoning,
        index_map=index_map,
        text2term_cache_folder=text2term_cache_folder,
        num_ctx=num_ctx,
        search_memo=search_memo,
        t2t_memo=t2t_memo,
    )
    by_accession = {r.extract.accession: r for r in reselected}
    merged: list[SelectEntry] = []
    for entry in entries:
        targets = targets_by_accession.get(entry.extract.accession)
        new = by_accession.get(entry.extract.accession)
        if not targets or new is None:
            merged.append(entry)
            continue
        merged.append(merge_reselected(entry, new, targets))
    return merged, responses, timings, errors
