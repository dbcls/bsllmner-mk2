import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set

from owlready2 import Ontology, ThingClass, get_ontology
from pydantic import BaseModel, Field

# === Build ontology index ===


class OntologyIndex(BaseModel):
    label_map: Dict[str, List[str]] = Field(default_factory=dict)  # label -> List[term_id]
    exact_syn_map: Dict[str, List[str]] = Field(default_factory=dict)  # exact_syn -> List[term_id]
    lower_syn_map: Dict[str, List[str]] = Field(default_factory=dict)  # lower_syn -> List[term_id]
    term_labels: Dict[str, Optional[str]] = Field(default_factory=dict)  # term_id -> label


def _iter_prop_values(t: ThingClass, prop_name: str) -> Iterable[str]:
    values = getattr(t, prop_name, None)
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set)):
        return (str(v) for v in values)
    return (str(values),)


def _match_additional_conditions(t: ThingClass, conditions: Dict[str, str]) -> bool:
    if not conditions:
        return True
    for key, val in conditions.items():
        found = False
        for v in _iter_prop_values(t, key):
            if v == val:
                found = True
                break
        if not found:
            return False

    return True


def _term_id_of(t: ThingClass) -> str:
    name = getattr(t, "name", None)
    if isinstance(name, str) and name:
        return name
    iri = getattr(t, "iri", None)
    if isinstance(iri, str) and iri:
        return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    return str(t)


def _label_of(t: ThingClass) -> Optional[str]:
    labels = list(_iter_prop_values(t, "label"))
    if labels:
        return labels[0]
    return None


def _append_to_map(m: Dict[str, List[str]], key: str, val: str) -> None:
    lst = m.get(key, None)
    if lst is None:
        m[key] = [val]
    else:
        if not lst or lst[-1] != val:
            lst.append(val)


def build_index(
    ontology: Ontology,
    additional_conditions: Optional[Dict[str, str]] = None,
) -> OntologyIndex:
    """
    Load the OWL file and index three types of properties: label, exact_syn, and lower_syn.
    """
    label_map: Dict[str, List[str]] = {}
    exact_syn_map: Dict[str, List[str]] = {}
    lower_syn_map: Dict[str, List[str]] = {}
    term_labels: Dict[str, Optional[str]] = {}  # term_id -> label

    conditions = additional_conditions or {}

    for cls_ in ontology.classes():
        if not _match_additional_conditions(cls_, conditions):
            continue
        term_id = _term_id_of(cls_)
        label = _label_of(cls_)
        if term_id not in term_labels:
            term_labels[term_id] = label

        for prop_name in ["label", "hasExactSynonym", "hasLowercaseSynonym"]:
            for v in _iter_prop_values(cls_, prop_name):
                if v:
                    if prop_name == "label":
                        _append_to_map(label_map, v, term_id)
                    elif prop_name == "hasExactSynonym":
                        _append_to_map(exact_syn_map, v, term_id)
                    elif prop_name == "hasLowercaseSynonym":
                        _append_to_map(lower_syn_map, v.lower(), term_id)

    return OntologyIndex(
        label_map=label_map,
        exact_syn_map=exact_syn_map,
        lower_syn_map=lower_syn_map,
        term_labels=term_labels,
    )


# === Search terms ===


_WS_SPLIT_RE = re.compile(r"[ \t\n\r]+")
_DELIMS_RE = re.compile(r"[-_+/]")


class SearchResult(BaseModel):
    query: str
    match_prop: Literal["label", "hasExactSynonym", "hasLowercaseSynonym"]
    term_id: str
    synonym: Optional[str] = None
    label: Optional[str] = None


def _split_words(text: str) -> List[str]:
    test = text.strip()
    if not test:
        return []
    return [w for w in _WS_SPLIT_RE.split(text) if w]


def generate_ngrams(text: str, n: int) -> List[str]:
    words = _split_words(text)
    if not words or n <= 0 or n > len(words):
        return []

    return [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]


def build_word_combinations(query: str, max_ngram: int = 7) -> List[str]:
    """
    Generate word combinations from the query by splitting with whitespace and delimiters.
    e.g., "TP53-mut" -> ["TP53-mut", "TP53", "mut"]
    Longest combinations first.
    """
    words = _split_words(query)
    if not words:
        return []

    max_ngram_size = min(max_ngram, len(words))
    delimited_query = _DELIMS_RE.sub(" ", query)

    result: List[str] = []
    seen: Set[str] = set()

    for n in reversed(range(1, max_ngram_size + 1)):
        ngrams = generate_ngrams(query, n)
        ngrams_delim = generate_ngrams(delimited_query, n)
        for ngram in ngrams + ngrams_delim:
            if ngram not in seen:
                result.append(ngram)
                seen.add(ngram)

    return result


def _accumulate_hits_for_query(
    query: str,
    index: OntologyIndex,
    hits: Dict[str, SearchResult],
) -> None:
    for term_id in index.label_map.get(query, ()):
        if term_id not in hits:
            hits[term_id] = SearchResult(
                query=query,
                match_prop="label",
                term_id=term_id,
                synonym=None,
                label=index.term_labels.get(term_id, None),
            )
    for term_id in index.exact_syn_map.get(query, ()):
        if term_id not in hits:
            hits[term_id] = SearchResult(
                query=query,
                match_prop="hasExactSynonym",
                term_id=term_id,
                synonym=query,
                label=index.term_labels.get(term_id, None),
            )
    for term_id in index.lower_syn_map.get(query.lower(), ()):
        if term_id not in hits:
            hits[term_id] = SearchResult(
                query=query,
                match_prop="hasLowercaseSynonym",
                term_id=term_id,
                synonym=query.lower(),
                label=index.term_labels.get(term_id, None),
            )


def search_terms(
    queries: Set[str],
    index: OntologyIndex,
    max_ngram: int = 7,
) -> Dict[str, List[SearchResult]]:  # query -> List[SearchResult]
    results: Dict[str, List[SearchResult]] = {}

    # First, search for exact match (case-sensitive)
    for query in queries:
        query_results: Dict[str, SearchResult] = {}  # term_id -> SearchResult
        _accumulate_hits_for_query(query, index, query_results)

        # Finish if exact match found
        if query_results:
            results[query] = list(query_results.values())
            continue

        # If no exact match, search with word decomposition (longest match first)
        word_combinations = build_word_combinations(query, max_ngram=max_ngram)
        for wc in word_combinations:
            _accumulate_hits_for_query(wc, index, query_results)

            # If matches found for this word count, don't search shorter combinations
            if query_results:
                break

        results[query] = list(query_results.values())

    return results


if __name__ == "__main__":
    TEST_QUERIES = {"Acute myeloid leukemia", "Homo sapiens", "TP53-mut"}
    OWL_FILE_PATH = Path("/app/ontology/cellosaurus.owl").resolve()
    ontology = get_ontology(OWL_FILE_PATH.as_uri()).load()
    index = build_index(ontology, additional_conditions={"hasDbXref": "NCBI_TaxID:9606"})
    results = search_terms(TEST_QUERIES, index)
    serializable = {
        q: [r.model_dump() for r in rs]
        for q, rs in results.items()
    }
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
