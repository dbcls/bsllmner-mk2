import shutil
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import (Any, Callable, List, Literal, NoReturn, Optional, Tuple,
                    TypeVar)

import uvicorn
import yaml
from fastapi import (APIRouter, FastAPI, File, Form, HTTPException, Query,
                     Response, UploadFile, status)

from bsllmner2.config import (MODULE_ROOT, PROMPT_EXTRACT_FILE_PATH, REPO_ROOT,
                              RESULT_DIR, get_config, set_logging_level)
from bsllmner2.schema import (API_VERSION, BsEntries, Mapping, Prompt, Result,
                              RunMetadata, ServiceInfo)
from bsllmner2.utils import (dump_result, get_now_str, list_run_metadata,
                             list_run_names, load_bs_entries, load_mapping,
                             load_result, to_result)

SMALL_TEST_DATA = {
    "bs_entries": REPO_ROOT.joinpath("tests/test-data/cell_line_example.biosample.json"),
    "mapping": REPO_ROOT.joinpath("tests/test-data/cell_line_example.mapping.tsv"),
}
LARGE_TEST_DATA = {
    "bs_entries": REPO_ROOT.joinpath("tests/zenodo-data/biosample_cellosaurus_mapping_testset.json"),
    "mapping": REPO_ROOT.joinpath("tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv"),
}


# === API Router ===

router = APIRouter()


@router.get(
    "/service-info",
    response_model=ServiceInfo,
)
async def service_info() -> ServiceInfo:
    return ServiceInfo(api_version=API_VERSION)


@router.get(
    "/default-extract-prompt",
    response_model=List[Prompt],
)
async def read_default_extract_prompt() -> List[Prompt]:
    if not PROMPT_EXTRACT_FILE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Default extract prompt file not found.",
        )

    with PROMPT_EXTRACT_FILE_PATH.open("r", encoding="utf-8") as f:
        raw_obj = yaml.safe_load(f)

    return [Prompt(**item) for item in raw_obj]

T = TypeVar("T")


async def load_upload_file(
    file: UploadFile,
    loader_func: Callable[[Path], T],
) -> T:
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.flush()
        temp_file.seek(0)
        return loader_func(Path(temp_file.name))


def _never() -> NoReturn:
    raise AssertionError("Unreachable code reached.")


async def _prepare_input_data(
    use_small_test_data: bool,
    use_large_test_data: bool,
    bs_entries: Optional[UploadFile],
    mapping: Optional[UploadFile],
) -> Tuple[BsEntries, Mapping]:
    num_sources = sum([
        use_small_test_data,
        use_large_test_data,
        bool(bs_entries and mapping),
    ])
    if num_sources != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one of small test data, large test data, or uploaded files must be specified.",
        )
    if use_small_test_data:
        return (
            load_bs_entries(SMALL_TEST_DATA["bs_entries"]),
            load_mapping(SMALL_TEST_DATA["mapping"]),
        )
    if use_large_test_data:
        return (
            load_bs_entries(LARGE_TEST_DATA["bs_entries"]),
            load_mapping(LARGE_TEST_DATA["mapping"]),
        )
    if bs_entries and mapping:
        bs_entries_data = await load_upload_file(bs_entries, load_bs_entries)
        mapping_data = await load_upload_file(mapping, load_mapping)
        return (bs_entries_data, mapping_data)

    _never()


@router.post(
    "/extract",
    response_model=Result,
)
async def extract(
    use_small_test_data: bool = Form(False),
    use_large_test_data: bool = Form(False),
    bs_entries: Optional[UploadFile] = File(None),
    mapping: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    model: str = Form(...),
    thinking: bool = Form(False),
    max_entries: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    run_name: Optional[str] = Form(None),
) -> Result:
    bs_entries_data, mapping_data = await _prepare_input_data(
        use_small_test_data,
        use_large_test_data,
        bs_entries,
        mapping,
    )
    if max_entries and (1 <= max_entries < len(bs_entries_data)):
        bs_entries_data = bs_entries_data[:max_entries]
    prompt_data = [Prompt(**item) for item in yaml.safe_load(prompt)]

    now = get_now_str()
    run_name = run_name or f"{model}_{now}"
    queue_obj = to_result(
        bs_entries=bs_entries_data,
        mapping=mapping_data,
        prompt=prompt_data,
        model=model,
        thinking=thinking,
        output=[],
        evaluation=[],
        config=get_config(),
        run_metadata=RunMetadata(
            run_name=run_name,
            model=model,
            thinking=thinking,
            username=username,
            start_time=now,
            end_time=None,
            status="running",
        ),
        args=None,  # CLI args are not used in API
        metrics=None,  # Metrics are collected in the worker
    )

    # Save the initial state of the result
    queue_file = dump_result(queue_obj, run_name)

    # pylint: disable=consider-using-with
    subprocess.Popen(
        [
            sys.executable,
            str(MODULE_ROOT.joinpath("extract_worker.py")),
            str(queue_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return queue_obj


class RunSortBy(str, Enum):
    START_TIME = "start_time"
    ACCURACY = "accuracy"
    PROCESSING_TIME = "processing_time"


class RunSortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


def make_none_safe_key(reverse: bool) -> Callable[[Any], Tuple[int, Any]]:
    def _key(val: Any) -> Tuple[int, Any]:
        if val is None:
            return (0 if reverse else 1, None)
        return (1 if reverse else 0, val)
    return _key


@router.get(
    "/extract/runs",
    response_model=List[RunMetadata],
)
async def list_extract_runs(
    response: Response,
    username: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    run_status: Literal["running", "completed", "failed"] = Query(None),
    sort_by: RunSortBy = Query(RunSortBy.START_TIME),
    sort_order: RunSortOrder = Query(RunSortOrder.DESC),
    page: int = Query(1, desciption="1-based page number"),
    page_size: int = Query(10, description="Number of results per page")
) -> List[RunMetadata]:
    metadata = list_run_metadata()

    # Filtering
    if username:
        metadata = [m for m in metadata if m.username == username]
    if model:
        metadata = [m for m in metadata if m.model == model]
    if run_status:
        metadata = [m for m in metadata if m.status == run_status]

    # Sorting
    reverse = sort_order == RunSortOrder.DESC
    key_fn = make_none_safe_key(reverse)
    metadata.sort(
        key=lambda m: key_fn(getattr(m, sort_by.value)),
        reverse=reverse,
    )

    # Pagination
    start = (page - 1) * page_size
    end = start + page_size
    response.headers["X-Total-Count"] = str(len(metadata))

    return metadata[start:end]


@router.get(
    "/extract/runs/{run_name}",
    response_model=Result,
)
async def get_extract_run(
    run_name: str,
) -> Result:
    try:
        result = load_result(RESULT_DIR.joinpath(f"{run_name}.json"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_name}' not found.",
        ) from exc
    return result


@router.get(
    "/extract/run-names",
    response_model=List[str],
)
async def list_extract_run_names() -> List[str]:
    return list_run_names()


# === main application setup ===


def create_app() -> FastAPI:
    app_config = get_config()
    set_logging_level(app_config.debug)  # Reconfigure logging level

    app = FastAPI(
        root_path=app_config.api_url_prefix,
        debug=app_config.debug,
    )

    app.include_router(router)

    return app


def run_api() -> None:
    app_config = get_config()
    set_logging_level(app_config.debug)
    uvicorn.run(
        "bsllmner2.api:create_app",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=app_config.debug,
        reload_dirs=[str(MODULE_ROOT)] if app_config.debug else None,
        factory=True,
    )


if __name__ == "__main__":
    run_api()
