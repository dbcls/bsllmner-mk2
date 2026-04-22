import argparse
import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from bsllmner2.config import LOGGER

HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE.parent.joinpath("chip-atlas-data")


# === Parse experimentList.tab from ChIP-Atlas ===


# ref.: https://github.com/inutano/chip-atlas/wiki#data-for-each-srx
# All ChIP-seq, ATAC-seq, DNase-seq, and Bisulfite-seq experiments recorded in ChIP-Atlas are described in experimentList.tab (Download, Table schema)
CHIP_ATLAS_EXPERIMENT_LIST_URL = "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab"
CHIP_ATLAS_EXPERIMENT_LIST_PATH = DATA_DIR.joinpath("experimentList.tab")
CHIP_ATLAS_EXPERIMENT_LIST_JSON_PATH = DATA_DIR.joinpath("experimentList.json")


class ChipAtlasExperiment(BaseModel):
    """Metadata for each experiment in ChIP-Atlas.

    Ref.: https://github.com/inutano/chip-atlas/wiki#tables-summarizing-metadata-and-files.
    """

    srx: str = Field(
        ...,
        description="Experimental ID (SRX, ERX, DRX)",
        examples=["SRX000001"],
    )
    genome_assembly: str | None = Field(
        None,
        description="Reference genome assembly",
        examples=["hg19"],
    )
    track_type_class: str | None = Field(
        None,
        description="Class of track type",
        examples=["TFs"],
    )
    track_type: str | None = Field(
        None,
        description="Specific track type",
        examples=["GATA2"],
    )
    cell_type_class: str | None = Field(
        None,
        description="Class of cell type",
        examples=["Blood"],
    )
    cell_type: str | None = Field(
        None,
        description="Specific cell type",
        examples=["K-562"],
    )
    cell_type_description: str | None = Field(
        None,
        description="Detailed description of the cell type",
        examples=["Primary Tissue=Blood|Tissue Diagnosis=Leukemia Chronic Myelogenous"],
    )
    processing_logs: str | None = Field(
        None,
        description=(
            "Processing logs of sequencing. "
            "For ChIP/ATAC/DNase-seq: '# of reads, percent mapped, percent duplicates, # of peaks [Q < 1E-05]'. "
            "For Bisulfite-seq: '# of reads, percent mapped, * coverage, # of hyper MR'."
        ),
        examples=["30180878,82.3,42.1,6691"],
    )
    title: str | None = Field(
        None,
        description="Experiment title",
        examples=["GSM722415: GATA2 K562bmp r1 110325 3"],
    )

    meta_fields: dict[str, str | None] = Field(
        default_factory=dict,
        description="Additional metadata fields as key-value pairs",
        examples=[{"antibody": "GATA2", "treatment": None}],
    )

    biosample_id: str | None = Field(
        None,
        description=(
            "BioSample ID associated with this experiment. "
            "Note: According to a full check of SRA_Accessions.tab, SRX and BioSample "
            "have a one-to-one relationship."
        ),
        examples=["SAMEA104646858"],
    )


async def download_chip_atlas_experiment_list(
    url: str = CHIP_ATLAS_EXPERIMENT_LIST_URL,
    path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH,
    force: bool = False,
) -> None:
    if path.exists() and not force:
        LOGGER.info("%s already exists. Skipping download.", path.name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading %s from %s...", path.name, url)

    async with httpx.AsyncClient(timeout=60.0) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                file.write(chunk)


def iterate_tsv(path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n").split("\t")


def parse_meta_field(element: str) -> tuple[str, str] | None:
    """Parse a metadata field in the format "key=value".

    Note: sometimes the value is "NA".
    """
    if "=" not in element:
        return None

    key, value = element.split("=", 1)
    return key, value


def normalize_field_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return None
    return value


def parse_experiment_list(
    path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH,
    genome_assembly: str | None = None,
) -> list[ChipAtlasExperiment]:
    LOGGER.info("Parsing %s...", path)

    experiments = []
    for line in iterate_tsv(path):
        fields = [normalize_field_value(line[j]) if j < len(line) else None for j in range(9)]
        srx = fields[0]
        if srx is None:
            continue

        if genome_assembly is not None and fields[1] != genome_assembly:
            continue

        meta_fields = {}
        if len(line) >= 10:
            for element in line[9:]:
                parsed = parse_meta_field(element)
                if parsed:
                    key, value = parsed
                    meta_fields[key] = normalize_field_value(value)

        experiments.append(
            ChipAtlasExperiment(
                srx=srx,
                genome_assembly=fields[1],
                track_type_class=fields[2],
                track_type=fields[3],
                cell_type_class=fields[4],
                cell_type=fields[5],
                cell_type_description=fields[6],
                processing_logs=fields[7],
                title=fields[8],
                meta_fields=meta_fields,
                biosample_id=None,  # to be filled later
            ),
        )

    return experiments


# === Parse SRA.Accessions.tab from ncbi.nlm.nih.gov ===

SRA_ACCESSIONS_FILE_URL = "https://ftp.ncbi.nlm.nih.gov/sra/reports/Metadata/SRA_Accessions.tab"
SRA_ACCESSIONS_FILE_PATH = DATA_DIR.joinpath("SRA_Accessions.tab")
BioAccessionType = Literal["bioproject", "biosample"]
SraEntityType = Literal["STUDY", "SAMPLE", "EXPERIMENT", "RUN"]
SRX_TO_BIOSAMPLE_PATH = DATA_DIR.joinpath("srx_to_biosample.json")


# ref.: https://github.com/linsalrob/SRA_Metadata/blob/master/README.md
ID_INDEX = 0
TYPE_INDEX = 6
BIOSAMPLE_INDEX = 17
BIOPROJECT_INDEX = 18
TYPE_FILTERS: dict[BioAccessionType, list[SraEntityType]] = {
    "bioproject": ["STUDY", "EXPERIMENT", "RUN"],
    "biosample": ["SAMPLE", "EXPERIMENT", "RUN"],
}


async def download_sra_accessions_file(
    path: Path = SRA_ACCESSIONS_FILE_PATH,
    url: str = SRA_ACCESSIONS_FILE_URL,
    force: bool = False,
) -> None:
    if path.exists() and not force:
        LOGGER.info("%s already exists. Skipping download.", path.name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading %s from %s...", path.name, url)

    async with httpx.AsyncClient(timeout=60.0) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                file.write(chunk)


def parse_sra_accessions_file(
    from_type: SraEntityType,
    to_type: BioAccessionType,
    path: Path = SRA_ACCESSIONS_FILE_PATH,
) -> dict[str, list[str]]:
    LOGGER.info("Parsing %s...", path)

    if from_type not in TYPE_FILTERS[to_type]:
        raise ValueError(f"Invalid from_type '{from_type}' for to_type '{to_type}'")

    id_to_relation_ids: dict[str, set[str]] = {}

    line_count = 0

    for line in iterate_tsv(path):
        line_count += 1
        if line_count == 1:
            # Skip header line
            continue

        if len(line) <= max(ID_INDEX, TYPE_INDEX, BIOSAMPLE_INDEX, BIOPROJECT_INDEX):
            continue

        type_value = line[TYPE_INDEX]
        if type_value != from_type:
            continue

        from_id = line[ID_INDEX]
        to_id = line[BIOPROJECT_INDEX] if to_type == "bioproject" else line[BIOSAMPLE_INDEX]
        if to_id == "-":
            continue

        if from_id not in id_to_relation_ids:
            id_to_relation_ids[from_id] = set()
        id_to_relation_ids[from_id].add(to_id)

    return {k: sorted(v) for k, v in id_to_relation_ids.items()}


# === Prepare BP Entries and Mapping (for bsllmner2) ===

DDBJ_SEARCH_BASE_URL = "https://ddbj.nig.ac.jp/search/api/entries/biosample"
DDBJ_SEARCH_BULK_URL = f"{DDBJ_SEARCH_BASE_URL}/bulk"
BULK_BATCH_SIZE = 1000
BULK_MAX_RETRIES = 3
BULK_RETRY_DELAY_SEC = 5.0
BS_ENTRIES_FILE_PATH = DATA_DIR.joinpath("bs_entries.jsonl")


async def download_bs_entry(
    accession: str,
    client: httpx.AsyncClient | None = None,
) -> Any | None:
    url = f"{DDBJ_SEARCH_BASE_URL}/{accession}.json"
    if client is None:
        async with httpx.AsyncClient(timeout=30.0) as c:
            return await _fetch_single_entry(c, url, accession)
    return await _fetch_single_entry(client, url, accession)


async def _fetch_single_entry(
    client: httpx.AsyncClient,
    url: str,
    accession: str,
) -> Any | None:
    response = await client.get(url)
    if response.status_code == 200:
        return response.json()
    if response.status_code == 404:
        LOGGER.info("BioSample entry not found for %s", accession)
        return None
    response.raise_for_status()
    return None


async def download_bs_entries_bulk(
    accessions: list[str],
    client: httpx.AsyncClient,
) -> tuple[dict[str, Any], list[str]]:
    """Fetch multiple BioSample entries via the Bulk API.

    Returns (accession -> properties dict, list of not-found accessions).
    """
    data: dict[str, Any] = {}
    for attempt in range(1, BULK_MAX_RETRIES + 1):
        try:
            response = await client.post(
                DDBJ_SEARCH_BULK_URL,
                json={"ids": accessions},
                params={"format": "json"},
            )
            response.raise_for_status()
            data = response.json()
            break
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if attempt == BULK_MAX_RETRIES:
                raise RuntimeError(
                    f"Bulk API request failed after {BULK_MAX_RETRIES} attempts",
                ) from e
            LOGGER.warning(
                "Bulk API request failed (attempt %d/%d): %s. Retrying...",
                attempt,
                BULK_MAX_RETRIES,
                e,
            )
            await asyncio.sleep(BULK_RETRY_DELAY_SEC * attempt)

    entries_map: dict[str, Any] = {}
    for entry in data.get("entries", []):
        props = entry.get("properties", {})
        accession = props.get("accession") or entry.get("identifier", "")
        if accession and "accession" not in props:
            props["accession"] = accession
        if accession:
            entries_map[accession] = props

    not_found: list[str] = data.get("notFound", [])
    return entries_map, not_found


def get_bs_entry_cache_path(accession: str) -> Path:
    m = re.search(r"(\d+)$", accession)
    if not m:
        return DATA_DIR.joinpath("bs_entries", f"{accession}.json")

    num = m.group(1)
    if len(num) < 3:
        prefix = num.zfill(3)
    else:
        prefix = num[:3]

    return DATA_DIR.joinpath("bs_entries", prefix, f"{accession}.json")


async def fetch_and_cache_bs_entries(
    biosample_ids: list[str],
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fetch BioSample entries using Bulk API with caching."""
    cached_entries: dict[str, Any] = {}
    uncached_ids: list[str] = []

    for accession in biosample_ids:
        cache_path = get_bs_entry_cache_path(accession)
        if cache_path.exists() and not force:
            with cache_path.open("r", encoding="utf-8") as f:
                cached_entries[accession] = json.load(f)
        else:
            uncached_ids.append(accession)

    LOGGER.info(
        "BioSample entries: %d cached, %d to fetch",
        len(cached_entries),
        len(uncached_ids),
    )

    fetched_entries: dict[str, Any] = {}
    not_found_all: list[str] = []

    if uncached_ids:
        total_batches = (len(uncached_ids) + BULK_BATCH_SIZE - 1) // BULK_BATCH_SIZE
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(uncached_ids), BULK_BATCH_SIZE):
                chunk = uncached_ids[i : i + BULK_BATCH_SIZE]
                LOGGER.info(
                    "Fetching batch %d/%d (%d entries)...",
                    i // BULK_BATCH_SIZE + 1,
                    total_batches,
                    len(chunk),
                )
                entries_map, not_found = await download_bs_entries_bulk(chunk, client)
                not_found_all.extend(not_found)

                for accession, entry_props in entries_map.items():
                    cache_path = get_bs_entry_cache_path(accession)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(entry_props, f, indent=2)
                    fetched_entries[accession] = entry_props

    if not_found_all:
        LOGGER.info(
            "%d BioSample entries not found (first 10): %s",
            len(not_found_all),
            not_found_all[:10],
        )

    result: list[dict[str, Any]] = []
    for accession in biosample_ids:
        entry = cached_entries.get(accession) or fetched_entries.get(accession)
        if entry is not None:
            result.append(entry)

    return result


# === Main ===


class Args(BaseModel):
    force: bool = Field(
        False,
        description="Force re-download of files even if they already exist.",
    )
    genome_assembly: str | None = Field(
        None,
        description="If given, filter experiments by genome_assembly (e.g. 'hg38').",
    )


def parse_args(raw_args: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description="Prepare ChIP-Atlas caches")

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist.",
    )
    parser.add_argument(
        "--genome-assembly",
        type=str,
        default=None,
        help="If given, filter experiments by genome_assembly (e.g. 'hg38').",
    )

    ns = parser.parse_args(raw_args)
    return Args(
        force=ns.force,
        genome_assembly=ns.genome_assembly,
    )


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    LOGGER.info("Preparing ChIP-Atlas BioSample entries...")
    args = parse_args()
    LOGGER.info("Arguments: %s", args.model_dump())

    # 1. download experimentList.tab & parse
    await download_chip_atlas_experiment_list(
        url=CHIP_ATLAS_EXPERIMENT_LIST_URL,
        path=CHIP_ATLAS_EXPERIMENT_LIST_PATH,
        force=args.force,
    )

    experiments = parse_experiment_list(
        path=CHIP_ATLAS_EXPERIMENT_LIST_PATH,
        genome_assembly=args.genome_assembly,
    )

    # 2. Download SRA_Accessions.tab & map SRX to BioSample
    await download_sra_accessions_file(
        path=SRA_ACCESSIONS_FILE_PATH,
        force=args.force,
    )
    srx_to_biosample = parse_sra_accessions_file(
        from_type="EXPERIMENT",
        to_type="biosample",
        path=SRA_ACCESSIONS_FILE_PATH,
    )

    for ex in experiments:
        if ex.srx in srx_to_biosample:
            biosample_ids = srx_to_biosample[ex.srx]
            if len(biosample_ids) > 1:
                LOGGER.info(
                    "Warning: Multiple BioSample IDs found for %s: %s",
                    ex.srx,
                    biosample_ids,
                )
            ex.biosample_id = biosample_ids[0]

    # 3. Save experimentList.json and srx_to_biosample.json
    with CHIP_ATLAS_EXPERIMENT_LIST_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump([ex.model_dump() for ex in experiments], f, indent=2)
    LOGGER.info("Saved experimentsList.json to %s", CHIP_ATLAS_EXPERIMENT_LIST_JSON_PATH)
    with SRX_TO_BIOSAMPLE_PATH.open("w", encoding="utf-8") as f:
        json.dump(srx_to_biosample, f, indent=2)
    LOGGER.info("Saved srx_to_biosample.json to %s", SRX_TO_BIOSAMPLE_PATH)

    # 4. Collect unique BioSample IDs (preserving order)
    unique_bs_ids: list[str] = []
    seen: set[str] = set()
    for ex in experiments:
        if ex.biosample_id is not None and ex.biosample_id not in seen:
            unique_bs_ids.append(ex.biosample_id)
            seen.add(ex.biosample_id)

    # 5. Fetch via Bulk API with caching
    bs_entries = await fetch_and_cache_bs_entries(
        biosample_ids=unique_bs_ids,
        force=args.force,
    )

    # 6. Write bs_entries.jsonl
    with BS_ENTRIES_FILE_PATH.open("w", encoding="utf-8") as f:
        for entry in bs_entries:
            f.write(json.dumps(entry) + "\n")
    LOGGER.info("Saved BioSample entries to %s", BS_ENTRIES_FILE_PATH)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
