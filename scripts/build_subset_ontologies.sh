#!/usr/bin/env bash
# Generate subset OWL files for CL / UBERON / ChEBI / MONDO using
# sh-ikeda/ontology-constructor-for-bsllmner SPARQL templates + ROBOT
# (Docker image obolibrary/robot). Outputs land under ontology/ and are
# loadable by owlready2.

set -euo pipefail

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/build_subset_ontologies.sh [--force]

Generates subset OWL files for CL / UBERON / ChEBI / MONDO under ontology/.

Prerequisites:
  - Upstream OWLs present under ontology/: cl.owl, efo.owl, uberon.owl,
    mondo.owl, chebi.owl (run scripts/download_ontology_files.py first).
  - docker with access to obolibrary/robot:latest.
  - git (for sh-ikeda clone/pull under work/).

Options:
  --force    Regenerate output even if it already exists.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
ONTOLOGY_DIR="${REPO_ROOT}/ontology"
WORK_DIR="${REPO_ROOT}/work"
SH_IKEDA_DIR="${WORK_DIR}/ontology-constructor-for-bsllmner"

missing=()
for f in cl.owl efo.owl uberon.owl mondo.owl chebi.owl; do
  [[ -f "${ONTOLOGY_DIR}/${f}" ]] || missing+=("${f}")
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing upstream OWL file(s) under ${ONTOLOGY_DIR}: ${missing[*]}" >&2
  echo "Run 'uv run python scripts/download_ontology_files.py' first." >&2
  exit 3
fi

mkdir -p "${WORK_DIR}"
if [[ ! -d "${SH_IKEDA_DIR}/.git" ]]; then
  echo "Cloning sh-ikeda/ontology-constructor-for-bsllmner into ${SH_IKEDA_DIR}"
  git clone --depth 1 https://github.com/sh-ikeda/ontology-constructor-for-bsllmner "${SH_IKEDA_DIR}"
else
  echo "Updating ${SH_IKEDA_DIR}"
  git -C "${SH_IKEDA_DIR}" pull --ff-only
fi

robot() {
  local heap="$1"; shift
  docker run --rm \
    -e ROBOT_JAVA_ARGS="${heap}" \
    -v "${ONTOLOGY_DIR}:/work" \
    -v "${SH_IKEDA_DIR}:/queries" \
    -v "${HERE}:/scripts" \
    -w /work \
    obolibrary/robot:latest \
    robot "$@"
}

add_class_type() {
  # Apply scripts/add_class_type.rq to add rdf:type owl:Class triples so
  # owlready2 can recognize subjects as classes.
  local infile="$1"
  local outfile="$2"
  robot "-Xmx8g" query --input "${infile}" --update /scripts/add_class_type.rq --output "${outfile}"
}

skip_if_exists() {
  local out="$1"
  if (( FORCE == 0 )) && [[ -f "${ONTOLOGY_DIR}/${out}" ]]; then
    echo "Skip (exists): ${out}"
    return 1
  fi
  return 0
}

cleanup() {
  for f in "$@"; do
    rm -f "${ONTOLOGY_DIR}/${f}"
  done
}

build_cl_variant() {
  # $1: variant ("human" or "mouse"), $2: output subset owl name
  local variant="$1"
  local out="$2"
  local cl_ttl="_cl_${variant}.ttl"
  local merged_ttl="_cl_${variant}_with_efo.ttl"
  local merged_owl="_cl_${variant}_with_efo.owl"

  robot "-Xmx8g" query --input cl.owl --query "/queries/cl/cl_construct_${variant}.rq" "${cl_ttl}"
  if [[ ! -f "${ONTOLOGY_DIR}/_efo_cell.ttl" ]]; then
    robot "-Xmx8g" query --input efo.owl --query /queries/cl/efo_construct.rq _efo_cell.ttl
  fi
  cat "${ONTOLOGY_DIR}/${cl_ttl}" "${ONTOLOGY_DIR}/_efo_cell.ttl" > "${ONTOLOGY_DIR}/${merged_ttl}"
  robot "-Xmx8g" convert --input "${merged_ttl}" --format owl --output "${merged_owl}"
  add_class_type "${merged_owl}" "${out}"
  cleanup "${cl_ttl}" "${merged_ttl}" "${merged_owl}"
}

build_simple_subset() {
  # $1: source OWL (e.g. uberon.owl), $2: query path, $3: output subset owl name
  local src="$1"
  local query="$2"
  local out="$3"
  local stem="_${out%.owl}"
  local tmp_ttl="${stem}.ttl"
  local tmp_owl="${stem}_pre.owl"

  robot "-Xmx8g" query --input "${src}" --query "${query}" "${tmp_ttl}"
  robot "-Xmx8g" convert --input "${tmp_ttl}" --format owl --output "${tmp_owl}"
  add_class_type "${tmp_owl}" "${out}"
  cleanup "${tmp_ttl}" "${tmp_owl}"
}

build_chebi_subset() {
  robot "-Xmx24g" query --input chebi.owl --update /queries/chebi/chebi_update.rq --output _chebi_role.owl
  robot "-Xmx24g" query --input _chebi_role.owl --query /queries/chebi/chebi_construct.rq _chebi_mod.ttl
  robot "-Xmx8g" convert --input _chebi_mod.ttl --format owl --output _chebi_pre.owl
  add_class_type _chebi_pre.owl chebi_subset.owl
  cleanup _chebi_role.owl _chebi_mod.ttl _chebi_pre.owl
}

if skip_if_exists cl_human_subset.owl; then
  build_cl_variant human cl_human_subset.owl
fi

if skip_if_exists cl_mouse_subset.owl; then
  build_cl_variant mouse cl_mouse_subset.owl
fi
cleanup _efo_cell.ttl

if skip_if_exists uberon_human_subset.owl; then
  build_simple_subset uberon.owl /queries/uberon/uberon_construct_human.rq uberon_human_subset.owl
fi

if skip_if_exists uberon_mouse_subset.owl; then
  build_simple_subset uberon.owl /queries/uberon/uberon_construct_mouse.rq uberon_mouse_subset.owl
fi

if skip_if_exists mondo_human_subset.owl; then
  build_simple_subset mondo.owl /queries/mondo/mondo_construct_human.rq mondo_human_subset.owl
fi

if skip_if_exists chebi_subset.owl; then
  build_chebi_subset
fi

echo ""
echo "Done. Subset OWLs under ${ONTOLOGY_DIR}:"
ls -lh "${ONTOLOGY_DIR}"/*_subset.owl 2>/dev/null || true
