import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

import text2term
from owlready2 import Ontology, ThingClass, World
from pydantic import BaseModel, Field

logging.getLogger("text2term").setLevel(logging.DEBUG)
logging.getLogger("text2term.term_collector").setLevel(logging.DEBUG)
logging.getLogger("text2term.t2t").setLevel(logging.DEBUG)

# === Build ontology index ===

DEFAULT_PREFIX_MAP: Dict[str, str] = {
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
}


def _expand_prop_uri(prop: str) -> str:
    if "://" in prop:
        return prop
    if ":" in prop:
        prefix, local = prop.split(":", 1)
        ns = DEFAULT_PREFIX_MAP.get(prefix, None)
        if ns:
            return ns + local

    return prop


class TermAnnotation(BaseModel):
    term_uri: str = Field(..., description="The term URI")
    term_id: str = Field(..., description="The term ID, e.g., CL:0000000")  # normalized
    prop_uri: Optional[str] = Field(None, description="The property URI, e.g., http://www.w3.org/2000/01/rdf-schema#label")
    value: str = Field(..., description="The property value")


def _normalize_key(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip().casefold()


def _iter_prop_values(t: ThingClass, prop_name: str) -> Iterable[str]:
    values = getattr(t, prop_name, None)
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set)):
        return (str(v) for v in values)
    return (str(values),)


def _term_uri_of(cls_: ThingClass) -> str:
    iri = getattr(cls_, "iri", None)
    if isinstance(iri, str) and iri:
        return iri

    name = getattr(cls_, "name", None)
    if isinstance(name, str) and name:
        onto = cls_.namespace.ontology
        return f"{onto.base_iri}{name}"

    return str(cls_)


def _term_id_of(t: ThingClass) -> str:
    name = getattr(t, "name", None)
    if isinstance(name, str) and name:
        return name

    iri = getattr(t, "iri", None)
    if isinstance(iri, str) and iri:
        p = urlparse(iri)
        candidate = p.fragment or p.path.rstrip("/").split("/")[-1]
        candidate = unquote(candidate)
        if candidate:
            return candidate

    labels = list(_iter_prop_values(t, "label"))
    if labels:
        return str(labels[0])

    return str(t)


def _normalize_term_id(term_id: str) -> str:
    """
    Examples:
      'CVCL_0384'                          -> 'CVCL:0384'
      'CVCL_R965'                          -> 'CVCL:R965'
      'CVCL:0384'                          -> 'CVCL:0384' (unchanged)
      'OBO:CELLOSAURUS#CVCL_R965'          -> 'CVCL:R965'
      'http://purl.obolibrary.org/obo/CVCL_R965' -> 'CVCL:R965'
    """
    t = unicodedata.normalize("NFKC", term_id).strip()
    if not t:
        return t

    if "#" in t:
        t = t.split("#", 1)[-1].strip()

    if "://" in t:
        p = urlparse(t)
        candidate = p.fragment or p.path.rstrip("/").split("/")[-1]
        t = unquote(candidate).strip() if candidate else t

    if ":" in t:
        parts = t.split(":", 1)
        if parts[0] and not parts[0].lower().startswith(("http", "https")):
            return t

    if "_" in t:
        parts = t.split("_", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}:{parts[1]}"

    return t


def _match_additional_conditions(t: ThingClass, conditions: Dict[str, str]) -> bool:
    if not conditions:
        return True
    for key, condition_val in conditions.items():
        found = False
        for target_val in _iter_prop_values(t, key):
            if _normalize_key(target_val) == _normalize_key(condition_val):
                found = True
                break
        if not found:
            return False

    return True


def iter_term_annotations(
    ontology: Ontology,
    additional_conditions: Optional[Dict[str, str]] = None,
) -> Iterable[TermAnnotation]:
    rdfs = DEFAULT_PREFIX_MAP["rdfs"]
    rdfs_ns = ontology.get_namespace(rdfs)
    skos = DEFAULT_PREFIX_MAP["skos"]
    skos_ns = ontology.get_namespace(skos)
    oio = DEFAULT_PREFIX_MAP["oboInOwl"]
    oio_ns = ontology.get_namespace(oio)

    props = {
        rdfs + "label": rdfs_ns.label,
        skos + "prefLabel": skos_ns.prefLabel,
        oio + "hasExactSynonym": oio_ns.hasExactSynonym,
        oio + "hasRelatedSynonym": oio_ns.hasRelatedSynonym,
        oio + "hasBroadSynonym": oio_ns.hasBroadSynonym,
        oio + "hasNarrowSynonym": oio_ns.hasNarrowSynonym,
        skos + "altLabel": skos_ns.altLabel,
        skos + "hiddenLabel": skos_ns.hiddenLabel,
    }

    for cls_ in ontology.classes():
        if additional_conditions and not _match_additional_conditions(cls_, additional_conditions):
            continue

        term_uri = _term_uri_of(cls_)
        term_id = _normalize_term_id(_term_id_of(cls_))

        for prop_uri, prop in props.items():
            if prop is None:
                continue
            for value in _iter_prop_values(cls_, prop.name):
                yield TermAnnotation(
                    term_uri=term_uri,
                    term_id=term_id,
                    prop_uri=prop_uri,
                    value=value,
                )


class OntologyIndex(BaseModel):
    term_id_to_labels: Dict[str, List[str]] = Field(default_factory=dict)
    value_to_annotations: Dict[str, List[TermAnnotation]] = Field(default_factory=dict)  # key is _normalize_key(value)


LABEL_URI_PROPS = {
    DEFAULT_PREFIX_MAP["rdfs"] + "label",
    DEFAULT_PREFIX_MAP["skos"] + "prefLabel",
}


def _is_label_prop(prop_uri: Optional[str]) -> bool:
    if not prop_uri:
        return False
    return prop_uri in LABEL_URI_PROPS


def build_index(
    ontology: Ontology,
    additional_conditions: Optional[Dict[str, str]] = None,
) -> OntologyIndex:
    term_id_to_labels: Dict[str, List[str]] = {}
    term_id_label_norms: Dict[str, Set[str]] = {}
    value_to_annotations: Dict[str, List[TermAnnotation]] = {}

    for ann in iter_term_annotations(ontology, additional_conditions):
        key = _normalize_key(ann.value)
        if _is_label_prop(ann.prop_uri):
            labels = term_id_to_labels.setdefault(ann.term_id, [])
            norms = term_id_label_norms.setdefault(ann.term_id, set())
            if key not in norms:
                labels.append(ann.value)
                norms.add(key)

        value_to_annotations.setdefault(key, []).append(ann)

    return OntologyIndex(
        term_id_to_labels=term_id_to_labels,
        value_to_annotations=value_to_annotations,
    )


def build_index_from_owl(
    owl_file: Path,
    additional_conditions: Optional[Dict[str, str]] = None,
) -> OntologyIndex:
    world = World()
    ontology = world.get_ontology(owl_file.as_uri()).load()
    return build_index(ontology, additional_conditions=additional_conditions)


def _iter_rows(file_path: Path) -> Iterable[Tuple[str, str, str]]:
    delimiter = "\t" if file_path.suffix == ".tsv" else None
    with file_path.open("r", encoding="utf-8") as f:
        if delimiter is None:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.reader(f, dialect)
            except csv.Error:
                reader = csv.reader(f, "excel")
        else:
            reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if len(row) < 3:
                continue
            yield row[0], row[1], row[2]


def build_index_from_table(
    file_path: Path,
) -> OntologyIndex:
    term_id_to_labels: Dict[str, List[str]] = {}
    term_id_label_norms: Dict[str, Set[str]] = {}
    value_to_annotations: Dict[str, List[TermAnnotation]] = {}

    for term_id, prop_raw, raw_value in _iter_rows(file_path):
        prop_uri = _expand_prop_uri(prop_raw)
        ann = TermAnnotation(
            term_uri=term_id,
            term_id=_normalize_term_id(term_id),
            prop_uri=prop_uri,
            value=raw_value,
        )
        key = _normalize_key(raw_value)
        value_to_annotations.setdefault(key, []).append(ann)

        if _is_label_prop(prop_uri):
            labels = term_id_to_labels.setdefault(term_id, [])
            norms = term_id_label_norms.setdefault(term_id, set())
            if key not in norms:
                labels.append(raw_value)
                norms.add(key)

    return OntologyIndex(
        term_id_to_labels=term_id_to_labels,
        value_to_annotations=value_to_annotations,
    )


# === Search terms ===


class SearchResult(TermAnnotation):
    label: Optional[str] = None
    exact_match: bool
    text2term_score: Optional[float] = None
    reasoning: Optional[str] = None


WS_SPLIT_RE = re.compile(r"[ \t\n\r]+")
BASE_DELIMS_RE = re.compile(r"[\-_/+·,;()\[\]{}|]+")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
ALNUM_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def _split_ws(text: str) -> List[str]:
    test = text.strip()
    if not test:
        return []
    return [w for w in WS_SPLIT_RE.split(test) if w]


def _tokenize_atom(atom: str) -> List[str]:
    """
    Tokenize a single whitespace-separated atom:
      1) Split CamelCase boundary
      2) Split alphabet<->digit boundaries
      3) Split by default delimiters
    """
    if not atom:
        return []
    s = CAMEL_BOUNDARY_RE.sub(" ", atom)
    s = ALNUM_BOUNDARY_RE.sub(" ", s)

    parts: List[str] = []
    for p in [s]:
        parts.extend([x for x in BASE_DELIMS_RE.split(p) if x])

    return parts


def _collect_joiners(query: str) -> List[str]:
    """
    Return joiners to produce joined n-gram variants.
    Always include space; add delimiters that actually appear in the original query.
    """
    joiners: List[str] = [" "]
    for j in ["-", "/", "_", "+"]:
        if j in query:
            joiners.append(j)
    return joiners


def _generate_windows(tokens: List[str], max_ngram: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) windows for n-grams, from longer to shorter."""
    nmax = min(max_ngram, len(tokens))
    for n in range(nmax, 0, -1):  # longest first
        for i in range(0, len(tokens) - n + 1):
            yield (i, i + n)


def build_word_combinations(
    query: str,
    max_ngram: int = 7,
) -> List[str]:
    """
    Build robust word combinations for search from a free-text query.

    Features:
      - NFKC normalization
      - CamelCase and alpha<->digit boundary splitting
      - Delimiter splitting (hyphen, slash, underscore, etc.)
      - Generate joined n-gram variants using joiners that appear in the query (e.g., "-", "/", "_", "+")
      - Always include the normalized original query as the first element
      - Return combinations in descending n-gram length; deduplicate with case-insensitive keys

    Args:
      query: Raw query string.
      max_ngram: Maximum n-gram size.

    Returns:
      A list of combinations (longest first), with duplicates removed.
    """
    if not query or not query.strip() or max_ngram <= 0:
        return []

    q_norm = _normalize_key(query)

    atoms = _split_ws(q_norm)
    tokens: List[str] = []
    for a in atoms:
        tokens.extend(_tokenize_atom(a))

    if not tokens:
        return [q_norm]

    joiners = _collect_joiners(q_norm)

    results: List[str] = []
    seen: Set[str] = set()

    # Always keep the normalized original query as the first element
    if q_norm not in seen:
        results.append(q_norm)
        seen.add(q_norm)

    for i, j in _generate_windows(tokens, max_ngram):
        window = tokens[i:j]

        # Always create a space-joined form
        joined = " ".join(window)
        if not joined in seen:
            results.append(joined)
            seen.add(joined)

        # Also create variants joined by any observed joiners (other than space)
        for jn in joiners:
            if jn == " ":
                continue
            variant = jn.join(window)
            if not variant in seen:
                results.append(variant)
                seen.add(variant)

    return results


def search_terms(
    index: OntologyIndex,
    queries: Iterable[str],
    max_ngram: int = 7,
) -> Dict[str, List[SearchResult]]:
    if not queries:
        return {}

    results: Dict[str, List[SearchResult]] = {}

    for query in queries:
        combinations = build_word_combinations(query, max_ngram=max_ngram)
        if not combinations:
            continue

        seen: Set[Tuple[str, Optional[str], str]] = set()  # (term_id, prop_uri, value)

        for comb in combinations:
            anns = index.value_to_annotations.get(comb, [])
            if not anns:
                continue

            for ann in anns:
                id_prop_val = (ann.term_id, ann.prop_uri, ann.value)
                if id_prop_val in seen:
                    continue
                seen.add(id_prop_val)

                label_list = index.term_id_to_labels.get(ann.term_id, [])
                label = label_list[0] if label_list else None

                result = SearchResult(
                    term_uri=ann.term_uri,
                    term_id=ann.term_id,
                    prop_uri=ann.prop_uri,
                    value=ann.value,
                    label=label,
                    exact_match=(_normalize_key(query) == _normalize_key(ann.value))
                )
                results.setdefault(query, []).append(result)

    return results


# === text2term mapper ===


def search_terms_with_text2term(
    queries: Iterable[str],
    owl_file: Path,
) -> Dict[str, List[SearchResult]]:
    df = text2term.map_terms(
        source_terms=list(queries),
        target_ontology=str(owl_file),
    )
    required = ["Source Term", "Mapped Term Label", "Mapped Term IRI", "Mapped Term CURIE", "Mapping Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from text2term output: {missing}. "
                         f"Have: {list(df.columns)}")

    index = build_index_from_owl(owl_file)

    results: Dict[str, List[SearchResult]] = {q: [] for q in queries}

    for query, group in df.groupby("Source Term"):
        seen: Set[Tuple[str, str]] = set()
        for _, row in group.iterrows():
            raw_value = row["Mapped Term Label"]
            value_key = _normalize_key(raw_value)
            term_uri = row["Mapped Term IRI"]
            raw_term_id = row["Mapped Term CURIE"]
            term_id = _normalize_term_id(raw_term_id)
            text2term_score = row["Mapping Score"]

            id_val = (term_id, value_key)
            if id_val in seen:
                continue
            seen.add(id_val)

            # Find the property URI from the index (owl file)
            prop_uri: Optional[str] = None
            candidates: List[TermAnnotation] = [
                ann for ann in index.value_to_annotations.get(value_key, [])  # pylint: disable=E1101
                if ann.term_id == term_id
            ]
            if len(candidates) == 1:
                prop_uri = candidates[0].prop_uri
            elif len(candidates) > 1:
                # Prefer label property if multiple candidates exist
                label_candidates = [c for c in candidates if _is_label_prop(c.prop_uri)]
                if label_candidates:
                    prop_uri = label_candidates[0].prop_uri
                else:
                    prop_uri = candidates[0].prop_uri  # fallback to the first one

            # Find the label from the index (owl file)
            label_list = index.term_id_to_labels.get(term_id, [])  # pylint: disable=E1101
            label = label_list[0] if label_list else None

            result = SearchResult(
                term_uri=term_uri,
                term_id=term_id,
                prop_uri=prop_uri,
                value=raw_value,
                label=label,
                exact_match=_normalize_key(query) == value_key,
                text2term_score=text2term_score,
            )
            results[query].append(result)

    return results


if __name__ == "__main__":
    # TEST_QUERIES = {"HeLa", "MCF-7", "A549"}
    # OWL_FILE_PATH = Path("/app/ontology/cellosaurus.owl").resolve()
    # TEST_QUERIES = {"GSK1210151A"}
    # OWL_FILE_PATH = Path("/app/ontology/chebi.owl").resolve()
    TEST_QUERIES = {"NEAT1", "SOX11", "DNMT3b", "SERPINE2", "PAF1"}
    OWL_FILE_PATH = Path("/app/ontology/ncbi_gene_human.owl").resolve()
    index = build_index_from_owl(
        OWL_FILE_PATH,
        # additional_conditions={"hasDbXref": "NCBI_TaxID:9606"}
    )
    results = search_terms(index, TEST_QUERIES)
    # results = search_terms_with_text2term(TEST_QUERIES, OWL_FILE_PATH)
    serializable = {
        q: [r.model_dump() for r in rs]
        for q, rs in results.items()
    }
    import json
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
