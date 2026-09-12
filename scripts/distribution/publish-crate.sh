#!/usr/bin/env bash
# Publish one bsllmner-mk2 RO-Crate to s3://biosampleplus/releases/<release-id>/
# and register it in index.json. Run it from a host that can both read the crate
# and reach the internet -- on shared HPC sites those are not always the same
# machine. See README.md in this directory.
#
#   bash publish-crate.sh <crate-dir> <release-id> [--skip-verify] [--force]
#
# Uploads the crate payload byte-for-byte: RO-Crate checksums and the digests in
# ro-crate-metadata.json refer to the uncompressed bytes at their exact relative
# paths, so nothing may be recompressed or moved. Bulk downloaders get the
# separate <release-id>.tar.gz sibling instead.
set -euo pipefail

BUCKET=biosampleplus
REGION=ap-northeast-1
AWS="${AWS_BIN:-aws}"

CRATE_DIR="${1:-}"; RELEASE_ID="${2:-}"; shift 2 || true
SKIP_VERIFY=false; FORCE=false
for a in "$@"; do
  case "$a" in
    --skip-verify) SKIP_VERIFY=true ;;
    --force) FORCE=true ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [ -z "$CRATE_DIR" ] || [ -z "$RELEASE_ID" ] || [ ! -d "$CRATE_DIR" ]; then
  echo "usage: bash publish-crate.sh <crate-dir> <release-id> [--skip-verify] [--force]" >&2
  exit 1
fi
CRATE_DIR="$(cd "$CRATE_DIR" && pwd)"
BASE="$(basename "$CRATE_DIR")"
PARENT="$(dirname "$CRATE_DIR")"
DEST="s3://$BUCKET/releases/$RELEASE_ID/"
TARBALL_KEY="releases/$RELEASE_ID.tar.gz"

[ -f "$CRATE_DIR/ro-crate-metadata.json" ] || { echo "error: no ro-crate-metadata.json in $CRATE_DIR" >&2; exit 1; }

echo "==> release  $RELEASE_ID"
echo "    crate    $CRATE_DIR"
echo "    dest     $DEST"

# ---------------------------------------------------------------- immutability
if [ "$FORCE" != true ]; then
  if [ -n "$($AWS s3 ls "$DEST" --region "$REGION" 2>/dev/null | head -1)" ]; then
    echo "error: releases/$RELEASE_ID/ already exists. Published releases are" >&2
    echo "       immutable; mint a new release ID, or pass --force to replace a" >&2
    echo "       genuinely broken upload." >&2
    exit 1
  fi
fi

# -------------------------------------------------------------------- verify
if [ "$SKIP_VERIFY" != true ] && [ -f "$CRATE_DIR/provenance/checksums.sha256" ]; then
  echo "==> verifying payload against provenance/checksums.sha256"
  ( cd "$CRATE_DIR" && sha256sum -c --quiet provenance/checksums.sha256 ) \
    || { echo "error: checksum verification FAILED — refusing to publish" >&2; exit 1; }
  echo "    all files match"
fi

# -------------------------------------------------------------------- tarball
WORK="${WORK_DIR:-$PARENT}"
TARBALL="$WORK/$RELEASE_ID.tar.gz"
if [ -f "$TARBALL" ]; then
  echo "==> reusing existing $TARBALL"
else
  echo "==> building $TARBALL (pigz -9, $(nproc) cores)"
  tar -I "pigz -9 -p $(nproc)" -cf "$TARBALL" \
      -C "$PARENT" --transform "s,^$BASE,$RELEASE_ID," "$BASE"
fi
echo "    $(du -h "$TARBALL" | cut -f1)"

# --------------------------------------------------------------------- upload
# One pass per extension so objects carry a useful Content-Type. The crate is
# uploaded verbatim; --no-guess-mime-type keeps the CLI from inventing a
# Content-Encoding that would make clients silently transform the bytes.
sync_ext() {  # <glob> <content-type>
  echo "    $1 -> $2"
  $AWS s3 sync "$CRATE_DIR/" "$DEST" --region "$REGION" \
    --exclude "*" --include "$1" \
    --no-guess-mime-type --content-type "$2" \
    --cache-control "public, max-age=86400" \
    --only-show-errors
}
echo "==> uploading crate payload"
sync_ext "*.json"   "application/json"
sync_ext "*.jsonl"  "application/x-ndjson"
sync_ext "*.owl"    "application/rdf+xml"
sync_ext "*.tsv"    "text/tab-separated-values; charset=utf-8"
sync_ext "*.log"    "text/plain; charset=utf-8"
sync_ext "*.sha256" "text/plain; charset=utf-8"
sync_ext "*.md"     "text/markdown; charset=utf-8"
sync_ext "*.yml"    "application/yaml"
sync_ext "*.py"     "text/x-python; charset=utf-8"

echo "==> catching any remaining file types"
$AWS s3 sync "$CRATE_DIR/" "$DEST" --region "$REGION" \
  --cache-control "public, max-age=86400" --only-show-errors

echo "==> uploading $TARBALL_KEY"
$AWS s3 cp "$TARBALL" "s3://$BUCKET/$TARBALL_KEY" --region "$REGION" \
  --content-type "application/gzip" --cache-control "public, max-age=86400" \
  --only-show-errors

# ------------------------------------------------------------------ index.json
echo "==> updating index.json"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
$AWS s3 cp "s3://$BUCKET/index.json" "$TMP/index.json" --region "$REGION" --quiet

CRATE_BYTES=$(du -sb "$CRATE_DIR" | cut -f1)
CRATE_FILES=$(find "$CRATE_DIR" -type f | wc -l)
TAR_BYTES=$(stat -c %s "$TARBALL")

python3 - "$TMP/index.json" "$CRATE_DIR/ro-crate-metadata.json" \
         "$RELEASE_ID" "$CRATE_BYTES" "$CRATE_FILES" "$TAR_BYTES" "$TARBALL_KEY" <<'PY'
import datetime, json, sys
idx_p, meta_p, rid, cbytes, cfiles, tbytes, tkey = sys.argv[1:8]
idx = json.load(open(idx_p))
g = json.load(open(meta_p))["@graph"]
root = next((e for e in g if e.get("@id") == "./"), {})

def ids(v):
    v = v if isinstance(v, list) else [v]
    return [x.get("@id") if isinstance(x, dict) else x for x in v]

names = {e["@id"]: e.get("name") for e in g if "@id" in e}
entry = {
    "release_id": rid,
    "prefix": f"releases/{rid}/",
    "tarball": tkey,
    "name": root.get("name"),
    "description": root.get("description"),
    "date_published": root.get("datePublished"),
    "license": (root.get("license") or {}).get("@id"),
    "authors": [names.get(a) or a for a in ids(root.get("author", []))],
    "ro_crate_profile": "https://w3id.org/ro/crate/1.1",
    "run_count": sum(1 for e in g if e.get("@type") == "CreateAction"),
    "file_count": int(cfiles),
    "total_bytes": int(cbytes),
    "tarball_bytes": int(tbytes),
}
rels = [r for r in idx.get("releases", []) if r.get("release_id") != rid]
rels.append(entry)
rels.sort(key=lambda r: r.get("release_id", ""))
idx["releases"] = rels
idx.pop("runs", None)
idx["updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
json.dump(idx, open(idx_p, "w"), indent=2, ensure_ascii=False)
open(idx_p, "a").write("\n")
print(f"    index.json now lists {len(rels)} release(s)")
PY

$AWS s3 cp "$TMP/index.json" "s3://$BUCKET/index.json" --region "$REGION" \
  --content-type "application/json" --cache-control "public, max-age=300" --quiet

echo "==> done"
echo "    https://$BUCKET.s3.$REGION.amazonaws.com/releases/$RELEASE_ID/ro-crate-metadata.json"
echo "    https://$BUCKET.s3.$REGION.amazonaws.com/$TARBALL_KEY"
