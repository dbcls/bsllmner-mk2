import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

import httpx
from pydantic import BaseModel, Field

from bsllmner2.client.ollama import ner
from bsllmner2.config import get_config
from bsllmner2.schema import LlmOutput, Prompt

HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE.parent.joinpath("tmp-data")
DATA_DIR.mkdir(exist_ok=True, parents=True)


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


async def download_chip_atlas_experiment_list(
    url: str = CHIP_ATLAS_EXPERIMENT_LIST_URL,
    path: Path = CHIP_ATLAS_EXPERIMENT_LIST_PATH,
    force: bool = False
) -> None:
    if path.exists() and not force:
        print(f"{path.name} already exists. Skipping download.", file=sys.stderr)
        return None

    print(f"Downloading {path.name} from {url}...", file=sys.stderr)

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
    print(f"Parsing {path}...", file=sys.stderr)

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
            biosample_id=None,  # to be filled later
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
        print(f"{path.name} already exists. Skipping download.", file=sys.stderr)
        return None

    print(f"Downloading {path.name} from {url}...", file=sys.stderr)

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
    print(f"Parsing {path}...", file=sys.stderr)

    if from_type not in TYPE_FILTERS[to_type]:
        raise ValueError(f"Invalid from_type '{from_type}' for to_type '{to_type}'")

    id_to_relation_ids: Dict[str, Set[str]] = {}

    line_count = 0

    for line in iterate_tsv(path):
        line_count += 1
        if line_count == 1:
            # Skip header line
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
        return json.load(file)  # type: ignore


def use_cache_srx_to_biosample_mapping(
    sra_accessions_path: Path = SRA_ACCESSIONS_FILE_PATH,
    force: bool = False
) -> Dict[str, List[str]]:
    if SRX_TO_BIOSAMPLE_PATH.exists() and not force:
        srx_to_biosample = load_srx_to_biosample_mapping()
    else:
        srx_to_biosample = parse_sra_accessions_file(
            from_type="EXPERIMENT",
            to_type="biosample",
            path=sra_accessions_path,
            num_lines=None,
        )
        save_srx_to_biosample_mapping(srx_to_biosample)

    return srx_to_biosample


# === Prepare BP Entries and Mapping (for bsllmner2) ===

DDBJ_SEARCH_BASE_URL = "https://ddbj.nig.ac.jp/search/entry/biosample"


async def download_bs_entry(accession: str) -> Optional[Any]:
    url = f"{DDBJ_SEARCH_BASE_URL}/{accession}.json"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"BioSample entry not found for {accession}", file=sys.stderr)
            return None
        else:
            response.raise_for_status()
            return None


def load_or_download_bs_entry(accession: str, force: bool = False) -> Optional[Any]:
    cache_path = DATA_DIR.joinpath(f"bs_entries/{accession}.json")
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

    return entry


# === utils for call bsllmner2 ===


def find_actual_value(
    experiment: ChipAtlasExperiment,
    field: str,
    case_sensitive: bool = False,
) -> Optional[str]:
    if case_sensitive:
        return experiment.meta_fields.get(field, None)
    else:
        norm_field = field.casefold()
        for key, value in experiment.meta_fields.items():
            if key.casefold() == norm_field:
                return value

    return None


def generate_prompt(predict_field: str) -> List[Prompt]:
    return [
        Prompt(
            role="system",
            content="You are an expert in biological metadata curation."
        ),
        Prompt(
            role="user",
            content=f"""
I will input JSON formatted metadata of a sample for a biological experiment.
Your task is to extract relevant biological information (if present) from the input data 
and format it according to the specified schema.

---

Categories to extract:
- "{predict_field}"

---

Output rules:
- Return **only JSON**, matching the provided schema (via the `format` option).
- If the category cannot be found in the input, output null for "{predict_field}".
- Prefer exact mentions in the input; if multiple candidates exist, pick the most specific and widely recognized.
- Do not hallucinate or infer values absent from the input.

---

Here is the input metadata:
"""
        )
    ]


def generate_format_schema(predict_field: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            predict_field: {
                "type": ["string", "null"],
            }
        },
        "required": [predict_field],
        "additionalProperties": False,
    }


class Evaluation(BaseModel):
    actual: str
    expected: Optional[str]
    match: bool


class Result(BaseModel):
    srx: str
    biosample_id: str
    evaluation: Evaluation
    output: LlmOutput


def evaluate_output(
    outputs: List[LlmOutput],
    bp_mapping: Dict[str, Tuple[str, str]],  # Key: biosample_id, Value: (actual_value, srx)
    predict_field: str,
) -> List[Result]:
    results = []
    for output in outputs:
        biosample_id = output.accession
        if biosample_id not in bp_mapping:
            print(f"Warning: No actual value found for {biosample_id}", file=sys.stderr)
            continue
        actual_value, srx = bp_mapping[biosample_id]
        if output.output is None or not isinstance(output.output, dict):
            expected_value = None
        else:
            expected_value = output.output.get(predict_field, None)
        evaluation = Evaluation(
            actual=actual_value,
            expected=expected_value,
            match=(actual_value == expected_value),
        )
        results.append(Result(
            srx=srx,
            biosample_id=biosample_id,
            evaluation=evaluation,
            output=output,
        ))

    return results


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
        "cell_line",
        description="The metadata field to predict (e.g., cell_type, tissue, antibody)"
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
        default="cell_line",
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

    if raw_args is None:
        raw_args = sys.argv[1:]
    parsed_args = parser.parse_args(raw_args)

    return Args(
        chip_atlas_experiment_list=parsed_args.chip_atlas_experiment_list,
        sra_accessions_file=parsed_args.sra_accessions_file,
        predict_field=parsed_args.predict_field,
        model=parsed_args.model,
        num_lines=parsed_args.num_lines,
        force=parsed_args.force,
    )


EXPERIMENTS_CACHE_PATH = DATA_DIR.joinpath("experiments.json")


def main() -> None:
    args = parse_args(sys.argv[1:])
    print(f"Arguments: {args.model_dump_json(indent=2)}", file=sys.stderr)

    # === Pre-processing to prepare experiments with biosample_id ===

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
            num_lines=None,  # do all lines (because this is pre-processing step)
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
                    print(
                        f"Warning: Multiple BioSample IDs found for {experiment.srx}: {biosample_ids}",
                        file=sys.stderr
                    )
                experiment.biosample_id = biosample_ids[0]
        with EXPERIMENTS_CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump([ex.model_dump() for ex in experiments], file, indent=2)

    # === Main processing ===
    bp_mapping: Dict[str, Tuple[str, str]] = {}  # Key: biosample_id
    bp_entries: List[Any] = []
    for experiment in experiments:
        if experiment.biosample_id is None:
            print(f"Skipping {experiment.srx} because biosample_id is None", file=sys.stderr)
            continue
        actual_value = find_actual_value(experiment, args.predict_field, case_sensitive=False)
        if actual_value is None:  # TODO: None?
            continue
        bs_entry = load_or_download_bs_entry(experiment.biosample_id, force=args.force)
        if bs_entry is None:
            continue

        bp_mapping[experiment.biosample_id] = (actual_value, experiment.srx)
        bp_entries.append(bs_entry)

        if args.num_lines is not None and len(bp_entries) >= args.num_lines:
            break

    bsllmner2_config = get_config()
    prompt = generate_prompt(args.predict_field)
    format_schema = generate_format_schema(args.predict_field)
    ner_output = asyncio.run(ner(
        config=bsllmner2_config,
        bs_entries=bp_entries,
        prompt=prompt,
        format_=format_schema,
        model=args.model,
        thinking=False,
    ))
    results = evaluate_output(ner_output, bp_mapping, args.predict_field)
    with DATA_DIR.joinpath(f"chip_atlas_ner_results_{args.model}_{args.predict_field}.json").open("w", encoding="utf-8") as file:
        json.dump([res.model_dump() for res in results], file, indent=2)

    all_num = len(results)
    match_num = sum(1 for res in results if res.evaluation.match)
    accuracy = match_num / all_num if all_num > 0 else 0.0
    print(f"Total entries evaluated: {all_num}", file=sys.stderr)
    print(f"Number of matches: {match_num}", file=sys.stderr)
    print(f"Accuracy: {accuracy:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
