import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

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
    """
    Metadata for each experiment in ChIP-Atlas.
    ref.: https://github.com/inutano/chip-atlas/wiki#tables-summarizing-metadata-and-files
    """
    srx: str = Field(
        ...,
        description="Experimental ID (SRX, ERX, DRX)",
        examples=["SRX000001"],
    )
    genome_assembly: Optional[str] = Field(
        None,
        description="Reference genome assembly",
        examples=["hg19"],
    )
    track_type_class: Optional[str] = Field(
        None,
        description="Class of track type",
        examples=["TFs"],
    )
    track_type: Optional[str] = Field(
        None,
        description="Specific track type",
        examples=["GATA2"],
    )
    cell_type_class: Optional[str] = Field(
        None,
        description="Class of cell type",
        examples=["Blood"],
    )
    cell_type: Optional[str] = Field(
        None,
        description="Specific cell type",
        examples=["K-562"],
    )
    cell_type_description: Optional[str] = Field(
        None,
        description="Detailed description of the cell type",
        examples=[
            "Primary Tissue=Blood|Tissue Diagnosis=Leukemia Chronic Myelogenous"
        ],
    )
    processing_logs: Optional[str] = Field(
        None,
        description=(
            "Processing logs of sequencing. "
            "For ChIP/ATAC/DNase-seq: '# of reads, percent mapped, percent duplicates, # of peaks [Q < 1E-05]'. "
            "For Bisulfite-seq: '# of reads, percent mapped, * coverage, # of hyper MR'."
        ),
        examples=["30180878,82.3,42.1,6691"],
    )
    title: Optional[str] = Field(
        None,
        description="Experiment title",
        examples=["GSM722415: GATA2 K562bmp r1 110325 3"],
    )

    meta_fields: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Additional metadata fields as key-value pairs",
        examples=[{"antibody": "GATA2", "treatment": None}],
    )

    biosample_id: Optional[str] = Field(
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
    force: bool = False
) -> None:
    if path.exists() and not force:
        LOGGER.info("%s already exists. Skipping download.", path.name)
        return None

    LOGGER.info("Downloading %s from %s...", path.name, url)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with path.open("wb") as file:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    file.write(chunk)


def iterate_tsv(path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n").split("\t")


def parse_meta_field(element: str) -> Optional[tuple[str, str]]:
    """
    Parse a metadata field in the format "key=value".
    note: sometimes the value is "NA".
    """
    if "=" not in element:
        return None

    key, value = element.split("=", 1)
    return key, value


def normalize_field_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return None
    return value


def parse_experiment_list(
    path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH,
    genome_assembly: Optional[str] = None,
) -> List[ChipAtlasExperiment]:
    LOGGER.info("Parsing %s...", path)

    experiments = []
    for i, line in enumerate(iterate_tsv(path)):
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

        experiments.append(ChipAtlasExperiment(
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
        ))

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
TYPE_FILTERS: Dict[BioAccessionType, List[SraEntityType]] = {
    "bioproject": ["STUDY", "EXPERIMENT", "RUN"],
    "biosample": ["SAMPLE", "EXPERIMENT", "RUN"],
}


async def download_sra_accessions_file(
    path: Path = SRA_ACCESSIONS_FILE_PATH,
    url: str = SRA_ACCESSIONS_FILE_URL,
    force: bool = False
) -> None:
    if path.exists() and not force:
        LOGGER.info("%s already exists. Skipping download.", path.name)
        return None

    LOGGER.info("Downloading %s from %s...", path.name, url)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with path.open("wb") as file:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    file.write(chunk)


def parse_sra_accessions_file(
    from_type: SraEntityType,
    to_type: BioAccessionType,
    path: Path = SRA_ACCESSIONS_FILE_PATH,
) -> Dict[str, List[str]]:
    LOGGER.info("Parsing %s...", path)

    if from_type not in TYPE_FILTERS[to_type]:
        raise ValueError(f"Invalid from_type '{from_type}' for to_type '{to_type}'")

    id_to_relation_ids: Dict[str, Set[str]] = {}

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

DDBJ_SEARCH_BASE_URL = "https://ddbj.nig.ac.jp/search/entry/biosample"
BS_ENTRIES_FILE_PATH = DATA_DIR.joinpath("bs_entries.jsonl")


async def download_bs_entry(accession: str) -> Optional[Any]:
    url = f"{DDBJ_SEARCH_BASE_URL}/{accession}.json"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            LOGGER.info("BioSample entry not found for %s", accession)
            return None
        else:
            response.raise_for_status()
            return None


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


def load_or_download_bs_entry(accession: str, force: bool = False) -> Optional[Any]:
    cache_path = get_bs_entry_cache_path(accession)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    entry = asyncio.run(download_bs_entry(accession))
    if entry is not None:
        entry_props = entry.get("properties", {})
        if "accession" not in entry_props:
            entry_props["accession"] = accession
        with cache_path.open("w", encoding="utf-8") as file:
            json.dump(entry_props, file, indent=2)

        return entry_props

    return None


# === Main ===


class Args(BaseModel):
    force: bool = Field(
        False,
        description="Force re-download of files even if they already exist.",
    )
    genome_assembly: Optional[str] = Field(
        None,
        description="If given, filter experiments by genome_assembly (e.g. 'hg38').",
    )


def parse_args(raw_args: Optional[List[str]] = None) -> Args:
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
    LOGGER.info("Preparing ChIP-Atlas BioSample entries...")
    args = parse_args()
    LOGGER.info("Arguments: %s", args.model_dump())

    # 1. download experimentList.tab & parse
    asyncio.run(
        download_chip_atlas_experiment_list(
            url=CHIP_ATLAS_EXPERIMENT_LIST_URL,
            path=CHIP_ATLAS_EXPERIMENT_LIST_PATH,
            force=args.force,
        )
    )

    experiments = parse_experiment_list(
        path=CHIP_ATLAS_EXPERIMENT_LIST_PATH,
        genome_assembly=args.genome_assembly,
    )

    # 2. Download SRA_Accessions.tab & map SRX to BioSample
    asyncio.run(
        download_sra_accessions_file(
            path=SRA_ACCESSIONS_FILE_PATH,
            force=args.force,
        )
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

    # 4. Download BioSample entries
    bs_entries = []
    for ex in experiments:
        if ex.biosample_id is not None:
            bs_entry = load_or_download_bs_entry(
                accession=ex.biosample_id,
                force=args.force,
            )
            if bs_entry is not None:
                bs_entries.append(bs_entry)
    with BS_ENTRIES_FILE_PATH.open("w", encoding="utf-8") as f:
        for entry in bs_entries:
            f.write(json.dumps(entry) + "\n")
    LOGGER.info("Saved BioSample entries to %s", BS_ENTRIES_FILE_PATH)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
