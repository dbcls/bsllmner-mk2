r"""Preprocess cellosaurus.obo into a per-species OBO with augmented annotations.

For each cell-line term retained under the selected NCBI Taxonomy ID this script:
  - Keeps NCBI_TaxID xref lines so downstream OWL tooling still sees the species.
  - Preserves the pre-existing `comment: Disease: <NCIt/ORDO label>` and
    `comment: derived_from: <CVCL label>` annotations.
  - Synthesizes a one-line `def: "..." []` from Category / Sex / Species of
    origin / Disease / Derived from so ROBOT convert emits an IAO_0000115
    textual definition in the OWL. Missing attributes are skipped (no
    placeholders).

Output is `ontology/cellosaurus_<suffix>.mod.obo` (suffix: human, mouse, ...).
Convert to OWL with:

    docker run -v $PWD/ontology:/work -w /work --rm obolibrary/robot \
        robot convert -i cellosaurus_human.mod.obo -o cellosaurus_human.owl \
        --format owl
"""

import argparse
import re
import sys
from pathlib import Path
from typing import TextIO

HERE = Path(__file__).parent.resolve()
ONTOLOGY_DIR = HERE.parent.joinpath("ontology")
INPUT_FILE = ONTOLOGY_DIR.joinpath("cellosaurus.obo")

TAXID_TO_SUFFIX = {
    "9606": "human",
    "10090": "mouse",
}

# subsetdef keys that denote Sex. Everything else declared in the OBO header is
# treated as a Category token (the subsetdef namespace is flat in Cellosaurus).
SEX_SUBSETS = frozenset(
    {
        "Female",
        "Male",
        "Mixed_sex",
        "Sex_ambiguous",
        "Sex_unspecified",
    }
)

SUBSETDEF_RE = re.compile(r'^subsetdef:\s*(\S+)\s+"([^"]+)"')
SUBSET_RE = re.compile(r"^subset:\s*(\S+)")
NAME_RE = re.compile(r"^name:")
DISEASE_XREF_RE = re.compile(r"^xref: (?:NCIt:C\d+|ORDO:Orphanet_\d+) ! (.+)$")
TAXID_XREF_RE = re.compile(r"^xref: NCBI_TaxID:(\d+)\s+!\s+(.+)$")
DERIVED_FROM_RE = re.compile(r"^relationship: derived_from CVCL_[A-Za-z0-9]+ ! (.+)$")
REMOVE_PREFIXES = ("comment:", "xref:", "relationship:", "creation_date:")


def output_path_for(taxid: str) -> Path:
    suffix = TAXID_TO_SUFFIX.get(taxid, taxid)
    return ONTOLOGY_DIR.joinpath(f"cellosaurus_{suffix}.mod.obo")


def _strip_species_annotation(species: str) -> str:
    """'Homo sapiens (Human)' -> 'Homo sapiens'."""
    base = species.split("(", maxsplit=1)[0].strip()
    return base or species.strip()


def build_subset_description_map(input_path: Path) -> dict[str, str]:
    """Parse subsetdef lines from the OBO header into a key -> description map."""
    mapping: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("[Term]", "[Typedef]")):
                break
            m = SUBSETDEF_RE.match(line.rstrip("\n"))
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def compose_definition(
    *,
    disease: str | None,
    category: str | None,
    sex: str | None,
    species: str | None,
    derived_from: str | None,
) -> str:
    """Build a single-sentence textual definition from the selected attributes.

    Missing attributes are skipped. When all are absent the fallback is the bare
    string ``"cell line"``.
    """
    lead: list[str] = []
    if sex in {"Female", "Male"}:
        lead.append(sex)
    if disease:
        lead.append(disease)
    lead.append("cell line")
    if derived_from:
        lead.append(f"derived from {derived_from}")
    main = " ".join(lead)

    tail: list[str] = []
    if category:
        tail.append(f"Category: {category}")
    if sex and sex not in {"Female", "Male"}:
        tail.append(f"Sex: {sex}")
    if species:
        tail.append(f"Species of origin: {species}")

    if tail:
        return main + "; " + "; ".join(tail)
    return main


def _escape_def(text: str) -> str:
    """Escape a string for OBO ``def:`` quoted literal syntax."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _extract_attributes(
    lines: list[str],
    subset_desc: dict[str, str],
) -> tuple[list[str], list[str], list[str], str | None, list[str], set[str]]:
    """Scan a [Term] block once, returning the attribute materials for composing def:.

    Returns (diseases, derived_from_names, category_values, sex_value, species_names, taxids).
    """
    diseases: list[str] = []
    derived_from_names: list[str] = []
    category_values: list[str] = []
    sex_value: str | None = None
    species_names: list[str] = []
    taxids: set[str] = set()

    for line in lines:
        stripped = line.rstrip("\n")

        m = DISEASE_XREF_RE.match(stripped)
        if m:
            diseases.append(m.group(1))
            continue

        m = DERIVED_FROM_RE.match(stripped)
        if m:
            derived_from_names.append(m.group(1))
            continue

        m = TAXID_XREF_RE.match(stripped)
        if m:
            species_names.append(_strip_species_annotation(m.group(2)))
            taxids.add(m.group(1))
            continue

        m = SUBSET_RE.match(stripped)
        if m:
            key = m.group(1)
            desc = subset_desc.get(key, key)
            if key in SEX_SUBSETS:
                sex_value = desc
            else:
                category_values.append(desc)

    return diseases, derived_from_names, category_values, sex_value, species_names, taxids


def process_term_block(
    lines: list[str],
    taxid_to_keep: str,
    subset_desc: dict[str, str],
) -> list[str] | None:
    """Rewrite one [Term] block.

    Returns the rewritten lines. If the term's NCBI_TaxID xref does not match
    ``taxid_to_keep``, returns None so the caller can drop the block entirely.
    """
    diseases, derived_from_names, category_values, sex_value, species_names, taxids = _extract_attributes(
        lines,
        subset_desc,
    )

    if taxid_to_keep not in taxids:
        return None

    definition = compose_definition(
        disease=diseases[0] if diseases else None,
        category=category_values[0] if category_values else None,
        sex=sex_value,
        species=species_names[0] if species_names else None,
        derived_from=derived_from_names[0] if derived_from_names else None,
    )

    out: list[str] = []
    trailing_comments: list[str] = []
    def_emitted = False

    for line in lines:
        stripped = line.rstrip("\n")

        if not def_emitted and NAME_RE.match(stripped):
            out.append(line)
            out.append(f'def: "{_escape_def(definition)}" []\n')
            def_emitted = True
            continue

        m = DISEASE_XREF_RE.match(stripped)
        if m:
            trailing_comments.append(f"comment: Disease: {m.group(1)}\n")

        m = DERIVED_FROM_RE.match(stripped)
        if m:
            trailing_comments.append(f"comment: derived_from: {m.group(1)}\n")

        if TAXID_XREF_RE.match(stripped):
            out.append(line)
            continue

        if stripped.startswith(REMOVE_PREFIXES):
            continue

        out.append(line)

    out.extend(trailing_comments)
    return out


def _flush_block(
    block: list[str],
    fout: TextIO,
    taxid: str,
    subset_desc: dict[str, str],
    counts: dict[str, int],
) -> None:
    if not block:
        return
    if block[0].startswith("[Term]"):
        processed = process_term_block(block, taxid, subset_desc)
        if processed is None:
            counts["term_dropped"] += 1
            return
        for bl in processed:
            fout.write(bl)
            counts["lines_written"] += 1
        counts["term_kept"] += 1
        return
    # Typedef or unknown stanza: pass through untouched
    for bl in block:
        fout.write(bl)
        counts["lines_written"] += 1


def preprocess(input_path: Path, output_path: Path, taxid: str) -> None:
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        print("Run download_ontology_files.py first.", file=sys.stderr)
        sys.exit(1)

    subset_desc = build_subset_description_map(input_path)
    counts: dict[str, int] = {"lines_written": 0, "term_kept": 0, "term_dropped": 0}

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        block: list[str] = []
        in_body = False

        for line in fin:
            if line.startswith(("[Term]", "[Typedef]")):
                # Flush whatever was buffered (header or previous block)
                if not in_body:
                    # Header accumulated in block: pass through verbatim
                    for bl in block:
                        fout.write(bl)
                        counts["lines_written"] += 1
                    in_body = True
                else:
                    _flush_block(block, fout, taxid, subset_desc, counts)
                block = [line]
                continue

            block.append(line)

        # Final flush
        if not in_body:
            for bl in block:
                fout.write(bl)
                counts["lines_written"] += 1
        else:
            _flush_block(block, fout, taxid, subset_desc, counts)

    print(
        f"Preprocessed {input_path.name}: {counts['lines_written']} lines written, "
        f"{counts['term_kept']} terms kept, {counts['term_dropped']} terms dropped "
        f"(taxid={taxid}).",
    )
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess cellosaurus.obo into a per-species OBO with augmented annotations.",
    )
    parser.add_argument(
        "--taxid",
        required=True,
        help="NCBI Taxonomy ID to extract (e.g. 9606 = human, 10090 = mouse).",
    )
    args = parser.parse_args()
    taxid = str(args.taxid).strip()
    output_path = output_path_for(taxid)
    preprocess(INPUT_FILE, output_path, taxid)


if __name__ == "__main__":
    main()
