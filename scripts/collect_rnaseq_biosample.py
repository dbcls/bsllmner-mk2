"""Collect human RNA-Seq BioSample entries from DDBJ Search API.

Uses cursor-based pagination (staging) to iterate all SRA experiments
matching 'RNA-Seq' + 'Homo sapiens', filters by LIBRARY_STRATEGY,
extracts BioSample IDs from dbXrefs, then bulk-fetches BioSample entries.

Usage:
    python3 scripts/collect_rnaseq_biosample.py \
        --date-from 2025-01-01 --date-to 2025-03-31 \
        --output-dir /tmp/rnaseq-biosample
"""

import argparse
import asyncio
import functools
import json
from pathlib import Path
from typing import Any

import httpx

print = functools.partial(print, flush=True)  # noqa: A001

# Staging has cursor support; production does not (yet).
STAGING_API = "https://ddbj-staging.nig.ac.jp/search/api"
PROD_API = "https://ddbj.nig.ac.jp/search/api"

PER_PAGE = 100
BULK_BATCH_SIZE = 1000
BULK_MAX_RETRIES = 3
BULK_RETRY_DELAY_SEC = 5.0


# === Step 1: Collect BioSample IDs via SRA experiment search + cursor ===


def _extract_library_strategy(item: dict[str, Any]) -> str:
    """Extract LIBRARY_STRATEGY from an SRA experiment's properties."""
    props = item.get("properties", {})
    experiment = props.get("EXPERIMENT_SET", {}).get("EXPERIMENT", {})
    descriptor = experiment.get("DESIGN", {}).get("LIBRARY_DESCRIPTOR", {})
    return descriptor.get("LIBRARY_STRATEGY", "")


def _extract_biosample_ids(item: dict[str, Any]) -> list[str]:
    """Extract BioSample identifiers from an entry's dbXrefs."""
    return [xref["identifier"] for xref in item.get("dbXrefs", []) if xref.get("type") == "biosample"]


async def collect_biosample_ids(
    client: httpx.AsyncClient,
    date_from: str,
    date_to: str,
) -> list[str]:
    """Search SRA experiments and collect linked BioSample IDs.

    Uses cursor pagination on staging API. Filters results locally
    by LIBRARY_STRATEGY == 'RNA-Seq'.
    """
    biosample_ids: dict[str, None] = {}  # ordered set
    total_checked = 0
    total_matched = 0
    cursor: str | None = None
    page_num = 0

    while True:
        if cursor is not None:
            # Cursor mode: only cursor + perPage allowed
            params: dict[str, Any] = {
                "perPage": PER_PAGE,
                "cursor": cursor,
            }
        else:
            # First request: include all search params
            params = {
                "perPage": PER_PAGE,
                "includeProperties": "true",
                "sort": "datePublished:asc",
                "keywords": '"RNA-Seq",Homo sapiens',
                "datePublishedFrom": date_from,
                "datePublishedTo": date_to,
            }

        response = await client.get(
            f"{STAGING_API}/entries/sra-experiment/",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for item in data["items"]:
            total_checked += 1
            strategy = _extract_library_strategy(item)
            if strategy != "RNA-Seq":
                continue
            total_matched += 1
            for bs_id in _extract_biosample_ids(item):
                biosample_ids[bs_id] = None

        pagination = data["pagination"]
        page_num += 1

        if page_num == 1:
            print(f"Total search results: {pagination['total']}")

        if page_num % 50 == 0:
            print(
                f"  Page {page_num}: checked {total_checked}, "
                f"matched {total_matched}, "
                f"unique BioSample IDs: {len(biosample_ids)}"
            )

        if not pagination.get("hasNext") or not pagination.get("nextCursor"):
            break
        cursor = pagination["nextCursor"]

    print(
        f"Done scanning. Checked {total_checked} experiments, "
        f"{total_matched} matched RNA-Seq, "
        f"{len(biosample_ids)} unique BioSample IDs."
    )
    return list(biosample_ids)


# === Step 2: Bulk-fetch BioSample entries (includeDbXrefs=false) ===


async def bulk_fetch_biosample(
    client: httpx.AsyncClient,
    biosample_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch BioSample entries via Bulk API with dbXrefs disabled."""
    all_entries: list[dict[str, Any]] = []
    all_not_found: list[str] = []

    total_batches = (len(biosample_ids) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE

    for i in range(0, len(biosample_ids), BULK_BATCH_SIZE):
        chunk = biosample_ids[i : i + BULK_BATCH_SIZE]
        batch_num = i // BULK_BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(chunk)} IDs)...")

        data: dict[str, Any] = {}
        for attempt in range(1, BULK_MAX_RETRIES + 1):
            try:
                response = await client.post(
                    f"{PROD_API}/entries/biosample/bulk",
                    json={"ids": chunk},
                    params={"format": "json", "includeDbXrefs": "false"},
                )
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                if attempt == BULK_MAX_RETRIES:
                    raise RuntimeError(
                        f"Bulk API failed after {BULK_MAX_RETRIES} attempts",
                    ) from e
                print(f"  Retry {attempt}/{BULK_MAX_RETRIES}: {e}")
                await asyncio.sleep(BULK_RETRY_DELAY_SEC * attempt)

        for entry in data.get("entries", []):
            props = entry.get("properties", {})
            accession = props.get("accession") or entry.get("identifier", "")
            if accession and "accession" not in props:
                props["accession"] = accession
            all_entries.append(props)

        all_not_found.extend(data.get("notFound", []))

    return all_entries, all_not_found


# === Main ===


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect human RNA-Seq BioSample entries from DDBJ Search API.",
    )
    parser.add_argument(
        "--date-from",
        required=True,
        help="Start date for datePublished filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to",
        required=True,
        help="End date for datePublished filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rnaseq-biosample"),
        help="Output directory (default: /tmp/rnaseq-biosample).",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    date_label = f"{args.date_from}_{args.date_to}"

    print(f"=== Collecting human RNA-Seq BioSamples: {args.date_from} to {args.date_to} ===")

    # Step 1: Collect BioSample IDs (skip if file already exists)
    ids_path = output_dir.joinpath(f"biosample_ids_{date_label}.json")
    if ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            biosample_ids = json.load(f)
        print(f"\n[Step 1] Loaded {len(biosample_ids)} BioSample IDs from {ids_path} (cached)")
    else:
        print("\n[Step 1] Scanning SRA experiments (staging API with cursor)...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            biosample_ids = await collect_biosample_ids(
                client,
                date_from=args.date_from,
                date_to=args.date_to,
            )
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(biosample_ids, f, indent=2)
        print(f"Saved {len(biosample_ids)} BioSample IDs to {ids_path}")

    # Step 2: Bulk-fetch BioSample entries (includeDbXrefs=false for speed)
    print(f"\n[Step 2] Bulk-fetching {len(biosample_ids)} BioSample entries...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        entries, not_found = await bulk_fetch_biosample(client, biosample_ids)

    if not_found:
        print(f"  {len(not_found)} entries not found (first 10): {not_found[:10]}")

    # Save entries as JSONL
    entries_path = output_dir.joinpath(f"bs_entries_{date_label}.jsonl")
    with entries_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} BioSample entries to {entries_path}")

    print("\n=== Done ===")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
