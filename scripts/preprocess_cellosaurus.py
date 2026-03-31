# Preprocess cellosaurus.obo to retain disease and derived_from information
# as rdfs:comment before converting to OWL.
#
# Usage:
#   python scripts/preprocess_cellosaurus.py
#   # Then convert with robot:
#   docker run -v $PWD/ontology:/work -w /work --rm obolibrary/robot \
#       robot convert -i cellosaurus.mod.obo -o cellosaurus.mod.owl --format owl

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ONTOLOGY_DIR = HERE.parent.joinpath("ontology")
INPUT_FILE = ONTOLOGY_DIR.joinpath("cellosaurus.obo")
OUTPUT_FILE = ONTOLOGY_DIR.joinpath("cellosaurus.mod.obo")

# xref patterns for disease ontologies (NCIt and ORDO)
DISEASE_XREF_RE = re.compile(r"^xref: (?:NCIt:C\d+|ORDO:Orphanet_\d+) ! (.+)$")
# xref pattern for NCBI TaxID (to keep)
TAXID_XREF_RE = re.compile(r"^xref: NCBI_TaxID:")
# relationship: derived_from pattern
DERIVED_FROM_RE = re.compile(r"^relationship: derived_from CVCL_[A-Za-z0-9]+ ! (.+)$")
# Lines to remove entirely
REMOVE_PREFIXES = ("comment:", "xref:", "relationship:", "creation_date:")


def preprocess(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Error: {input_path} not found.", file=sys.stderr)
        print("Run download_ontology_files.py first.", file=sys.stderr)
        sys.exit(1)

    pending_comments: list[str] = []
    written = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")

            # Extract disease name from NCIt/ORDO xref before filtering
            disease_match = DISEASE_XREF_RE.match(stripped)
            if disease_match:
                pending_comments.append(f'comment: "Disease: {disease_match.group(1)}"')

            # Extract derived_from cell line name before filtering
            derived_match = DERIVED_FROM_RE.match(stripped)
            if derived_match:
                pending_comments.append(f'comment: "derived_from: {derived_match.group(1)}"')

            # Keep NCBI_TaxID xrefs
            if TAXID_XREF_RE.match(stripped):
                fout.write(line)
                written += 1
                continue

            # Remove comment, xref, relationship, creation_date lines
            if stripped.startswith(REMOVE_PREFIXES):
                skipped += 1
                continue

            # Write the original line
            fout.write(line)
            written += 1

            # Emit accumulated comments after the last removed line
            # (flush when we hit a non-removed line after accumulating)
            if pending_comments:
                for comment_line in pending_comments:
                    fout.write(comment_line + "\n")
                    written += 1
                pending_comments.clear()

    print(f"Preprocessed {input_path.name}: {written} lines written, {skipped} lines removed.")
    print(f"Output: {output_path}")


def main() -> None:
    preprocess(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
