import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

import httpx
from pydantic import BaseModel, Field

HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE.parent.joinpath("chip-atlas-data")
RESULTS_DIR = DATA_DIR.joinpath("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],  # stderrに出す
)

LOGGER = logging.getLogger(__name__)


# === Parse experimentList.tab from ChIP-Atlas ===


# ref.: https://github.com/inutano/chip-atlas/wiki#data-for-each-srx
# All ChIP-seq, ATAC-seq, DNase-seq, and Bisulfite-seq experiments recorded in ChIP-Atlas are described in experimentList.tab (Download, Table schema)
CHIP_ATLAS_EXPERIMENT_LIST_URL = "https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab"
CHIP_ATLAS_FILE_NAME = "experimentList.tab"
CHIP_ATLAS_EXPERIMENT_LIST_PATH = DATA_DIR.joinpath(CHIP_ATLAS_FILE_NAME)


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


CHIP_ATLAS_KEYS = [
    "genome_assembly",
    "track_type_class",
    "track_type",
    "cell_type_class",
    "cell_type",
    "cell_type_description",
]


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
    if "=" not in element:
        return None
    key, value = element.split("=", 1)
    return key, value


def dump_meta_field_keys(
    experiments: List[ChipAtlasExperiment],
) -> None:
    field_keys: Set[str] = set()
    for experiment in experiments:
        field_keys.update(experiment.meta_fields.keys())

    with DATA_DIR.joinpath("meta_field_keys.txt").open("w", encoding="utf-8") as f:
        for key in sorted(field_keys):
            f.write(f"{key}\n")


def normalize_field_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return None
    return value


def parse_experiment_list(
    path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH,
    num_lines: Optional[int] = None
) -> List[ChipAtlasExperiment]:
    LOGGER.info("Parsing %s...", path)

    experiments = []
    for i, line in enumerate(iterate_tsv(path)):
        fields = [normalize_field_value(line[j]) if j < len(line) else None for j in range(9)]
        srx = fields[0]
        if srx is None:
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
            biosample_id=None,
        ))

        if num_lines is not None and i + 1 == num_lines:
            break

    return experiments


# === Parse SRA.Accessions.tab from ncbi.nlm.nih.gov ===


SRA_ACCESSIONS_FILE_URL = "https://ftp.ncbi.nlm.nih.gov/sra/reports/Metadata/SRA_Accessions.tab"
SRA_ACCESSIONS_FILE_NAME = "SRA_Accessions.tab"
SRA_ACCESSIONS_FILE_PATH = DATA_DIR.joinpath(SRA_ACCESSIONS_FILE_NAME)
BioAccessionType = Literal["bioproject", "biosample"]
SraEntityType = Literal["STUDY", "SAMPLE", "EXPERIMENT", "RUN"]
SRX_TO_BIOSAMPLE_PATH = DATA_DIR.joinpath("srx_to_biosample.json")


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
    num_lines: Optional[int] = None,
) -> Dict[str, List[str]]:
    LOGGER.info("Parsing %s...", path)

    if from_type not in TYPE_FILTERS[to_type]:
        raise ValueError(f"Invalid from_type '{from_type}' for to_type '{to_type}'")

    id_to_relation_ids: Dict[str, Set[str]] = {}

    line_count = 0

    for line in iterate_tsv(path):
        line_count += 1
        if line_count == 1:
            continue
        if num_lines is not None and line_count >= num_lines:
            break

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


def save_srx_to_biosample_mapping(mapping: Dict[str, List[str]]) -> None:
    with SRX_TO_BIOSAMPLE_PATH.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)


def load_srx_to_biosample_mapping() -> Dict[str, List[str]]:
    if not SRX_TO_BIOSAMPLE_PATH.exists():
        raise FileNotFoundError(f"{SRX_TO_BIOSAMPLE_PATH} does not exist.")

    with SRX_TO_BIOSAMPLE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def use_cache_srx_to_biosample_mapping(
    sra_accessions_path: Path = SRA_ACCESSIONS_FILE_PATH,
    force: bool = False
) -> Dict[str, List[str]]:
    if SRX_TO_BIOSAMPLE_PATH.exists() and not force:
        return load_srx_to_biosample_mapping()

    mapping = parse_sra_accessions_file(
        from_type="EXPERIMENT",
        to_type="biosample",
        path=sra_accessions_path,
        num_lines=None,
    )
    save_srx_to_biosample_mapping(mapping)
    return mapping


# === Prepare BP Entries and Mapping (for bsllmner2) ===

DDBJ_SEARCH_BASE_URL = "https://ddbj.nig.ac.jp/search/entry/biosample"


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


def load_or_download_bs_entry(accession: str, force: bool = False) -> Optional[Any]:
    digits = "".join([c for c in accession if c.isdigit()])
    digits = digits.zfill(6)
    prefix = digits[:3]

    base_dir = DATA_DIR.joinpath("bs_entries", prefix)
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_path = base_dir.joinpath(f"{accession}.json")

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

    return entry


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# === Main ====


class Args(BaseModel):
    chip_atlas_experiment_list: Path = Field(
        CHIP_ATLAS_EXPERIMENT_LIST_PATH,
        description="Path to the ChIP-Atlas experiment list file.",
    )
    sra_accessions_file: Path = Field(
        SRA_ACCESSIONS_FILE_PATH,
        description="Path to the SRA accessions file.",
    )
    predict_field: str = Field(
        "cell_type",
        description="The metadata field to predict (e.g., cell_type, cell_type_class)"
    )
    model: str = Field(
        "llama3.1:70b",
        description="LLM model to use for NER.",
    )
    num_lines: Optional[int] = Field(
        None,
        description="Number of lines to process from the chip-atlas experiment list for testing.",
    )
    force: bool = Field(
        False,
        description="Force re-download of files even if they already exist.",
    )
    batch_size: int = Field(
        128,
        description="Batch size for NER calls.",
    )
    resume: bool = Field(
        False,
        description="Resume from the last processed entry if output file exists.",
    )


def parse_args(raw_args: Optional[List[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description="Process ChIP-Atlas accessions")

    parser.add_argument(
        "--chip-atlas-experiment-list",
        type=Path,
        default=CHIP_ATLAS_EXPERIMENT_LIST_PATH,
        help="Path to the ChIP-Atlas experiment list file.",
    )
    parser.add_argument(
        "--sra-accessions-file",
        type=Path,
        default=SRA_ACCESSIONS_FILE_PATH,
        help="Path to the SRA accessions file.",
    )
    parser.add_argument(
        "--predict-field",
        type=str,
        default="cell_type",
        help="The metadata field to predict (e.g., cell_type, tissue, antibody)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:70b",
        help="LLM model to use for NER.",
    )
    parser.add_argument(
        "--num-lines",
        type=int,
        default=None,
        help="Number of lines to process from the chip-atlas experiment list for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for NER calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last processed entry if output file exists.",
    )

    if raw_args is None:
        raw_args = sys.argv[1:]
    parsed_args = parser.parse_args(raw_args)

    if parsed_args.predict_field not in CHIP_ATLAS_KEYS:
        raise ValueError(f"Invalid predict_field '{parsed_args.predict_field}'. Must be one of {CHIP_ATLAS_KEYS}.")

    return Args(
        chip_atlas_experiment_list=parsed_args.chip_atlas_experiment_list,
        sra_accessions_file=parsed_args.sra_accessions_file,
        predict_field=parsed_args.predict_field,
        model=parsed_args.model,
        num_lines=parsed_args.num_lines,
        force=parsed_args.force,
        batch_size=parsed_args.batch_size,
        resume=parsed_args.resume,
    )


EXPERIMENTS_CACHE_PATH = DATA_DIR.joinpath("experiments.json")


def main() -> None:
    LOGGER.info("Start chip_atlas_batch")

    args = parse_args(sys.argv[1:])
    LOGGER.info("Argument: %s", args.model_dump())

    # === Pre-processing ===

    if EXPERIMENTS_CACHE_PATH.exists() and not args.force:
        with EXPERIMENTS_CACHE_PATH.open("r", encoding="utf-8") as file:
            experiments = [ChipAtlasExperiment.model_validate(ex) for ex in json.load(file)]
    else:
        asyncio.run(download_chip_atlas_experiment_list(
            path=args.chip_atlas_experiment_list,
            force=args.force
        ))
        experiments = parse_experiment_list(
            path=args.chip_atlas_experiment_list,
            num_lines=None,
        )
        asyncio.run(download_sra_accessions_file(
            path=args.sra_accessions_file,
            force=args.force
        ))
        srx_to_biosample = use_cache_srx_to_biosample_mapping(
            sra_accessions_path=args.sra_accessions_file,
            force=args.force
        )
        for experiment in experiments:
            if experiment.srx in srx_to_biosample:
                biosample_ids = srx_to_biosample[experiment.srx]
                if len(biosample_ids) > 1:
                    LOGGER.info("Warning: Multiple BioSample IDs found for %s: %s", experiment.srx, biosample_ids)
                experiment.biosample_id = biosample_ids[0]
        with EXPERIMENTS_CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump([ex.model_dump() for ex in experiments], file, indent=2)

    # === Main processing ===

    work_items: List[Tuple[str, str, Any, Optional[str]]] = []

    for experiment in experiments:
        # もとの "hg38" フィルタは維持
        if experiment.genome_assembly != "hg38":
            continue
        if experiment.biosample_id is None:
            LOGGER.info("Skipping %s because biosample_id is None", experiment.srx)
            continue

        bs_entry = load_or_download_bs_entry(experiment.biosample_id, force=args.force)
        if bs_entry is None:
            continue

        work_items.append((
            experiment.biosample_id,
            experiment.srx,
            bs_entry,
            None
        ))
        if args.num_lines is not None and len(work_items) >= args.num_lines:
            break

    LOGGER.info("To-do items: %d", len(work_items))

    jsonl_path = RESULTS_DIR.joinpath("biosample_entries.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = 5000

    def flush_chunk(chunk: List[Tuple[str, str, Any, Optional[str]]]) -> None:
        # chunk の内容を追記して、呼び出し側で chunk を空にする
        with jsonl_path.open("a", encoding="utf-8") as f:
            for biosample_id, srx, bs_entry, _ in chunk:
                rec = {
                    "biosample_id": biosample_id,
                    "srx": srx,
                    "entry": bs_entry,
                }
                f.write(json_dumps(rec) + "\n")

    # === chunk flush 書き込み ===
    buffer: List[Tuple[str, str, Any, Optional[str]]] = []
    for item in work_items:
        buffer.append(item)
        if len(buffer) >= chunk_size:
            flush_chunk(buffer)
            buffer.clear()   # メモリ開放

    # 余りを flush
    if buffer:
        flush_chunk(buffer)
        buffer.clear()

    LOGGER.info("JSONL generated: %s", jsonl_path)


if __name__ == "__main__":
    main()
