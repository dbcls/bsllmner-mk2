"""Ontology index and text2term cache I/O for select mode.

Split out of :mod:`bsllmner2.select` so that filesystem-touching helpers live
separately from the async orchestration logic. The public surface
(:data:`INDEX_CACHE_DIR`, :data:`TEXT2TERM_CACHE_DIR`, :func:`build_index_map`,
:func:`build_text2term_cache`, :func:`_text2term_acronym`) is re-exported from
:mod:`bsllmner2.select` so existing tests that ``patch("bsllmner2.select.X", ...)``
continue to work.

The pickle usage below is intentional: the index cache files are produced by
this same process from trusted local OWL/TSV inputs. They are not loaded from
untrusted sources.
"""

import os
import pickle
from pathlib import Path

from bsllmner2.benchmark import stage_timer
from bsllmner2.config import LOGGER
from bsllmner2.models import DiskIoTimings, OntologyIndex, SelectConfig
from bsllmner2.ontology_search import (
    build_index_from_file,
    build_text2term_cache_for_owl,
    text2term_cache_exists,
)

INDEX_CACHE_DIR = Path(os.environ.get("BSLLMNER2_INDEX_CACHE_DIR", "ontology/index_cache"))
TEXT2TERM_CACHE_DIR = Path(os.environ.get("BSLLMNER2_TEXT2TERM_CACHE_DIR", "ontology/text2term_cache"))

# Stable cache-key suffix kept so on-disk cache file names remain consistent with past runs.
_CACHE_KEY_SUFFIX = "nofilter"


def _text2term_acronym(ontology_file: Path) -> str:
    """Stable text2term acronym that invalidates alongside the word-combination index cache."""
    return f"{ontology_file.stem}_{_CACHE_KEY_SUFFIX}"


def build_index_map(select_config: SelectConfig) -> tuple[dict[Path, OntologyIndex], DiskIoTimings]:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mapping: dict[Path, OntologyIndex] = {}
    disk_io = DiskIoTimings()

    for field_config in select_config.fields.values():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue
        if ontology_file_path in mapping:
            continue

        cache_file_path = INDEX_CACHE_DIR.joinpath(f"{ontology_file_path.name}_{_CACHE_KEY_SUFFIX}_v2.pkl")
        if cache_file_path.exists():
            try:
                with stage_timer("cache_load") as t, cache_file_path.open("rb") as f:
                    index = pickle.load(f)
                disk_io.index_cache_load_sec.append(t.elapsed_sec)
                mapping[ontology_file_path] = index
                continue
            except (OSError, EOFError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
                LOGGER.warning("Failed to load cache %s", cache_file_path, exc_info=True)

        with stage_timer("index_build") as t:
            index = build_index_from_file(ontology_file_path)
        disk_io.index_build_from_file_sec.append(t.elapsed_sec)
        mapping[ontology_file_path] = index

        try:
            with stage_timer("cache_save") as t, cache_file_path.open("wb") as f:
                pickle.dump(index, f)
            disk_io.index_cache_save_sec.append(t.elapsed_sec)
        except OSError:
            LOGGER.warning("Failed to save cache %s", cache_file_path, exc_info=True)

    return mapping, disk_io


def build_text2term_cache(select_config: SelectConfig) -> DiskIoTimings:
    """Ensure each OWL ontology is preregistered with text2term under its stable acronym.

    Called once per run before batch processing. Populates the shared cache folder so per-batch
    ``text2term.map_terms(..., use_cache=True)`` avoids OWL parsing. Non-OWL ontology files are
    skipped. Failures are logged and the acronym is simply not preregistered, in which case the
    per-batch wrapper will fall back to the uncached path (``target_ontology=<owl_path>``).
    """
    try:
        TEXT2TERM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        LOGGER.warning(
            "Failed to create text2term cache directory %s; text2term will run without cache",
            TEXT2TERM_CACHE_DIR,
            exc_info=True,
        )
        return DiskIoTimings()

    disk_io = DiskIoTimings()
    seen: set[Path] = set()

    for field_config in select_config.fields.values():
        ontology_file_path = field_config.ontology_file
        if ontology_file_path is None:
            continue
        if ontology_file_path.suffix != ".owl":
            continue
        if ontology_file_path in seen:
            continue
        seen.add(ontology_file_path)

        acronym = _text2term_acronym(ontology_file_path)

        try:
            with stage_timer("text2term_cache_load") as t:
                exists = text2term_cache_exists(acronym, TEXT2TERM_CACHE_DIR)
            disk_io.text2term_cache_load_sec.append(t.elapsed_sec)
        except (OSError, AttributeError, RuntimeError):
            LOGGER.warning(
                "text2term cache_exists failed for %s (acronym=%s); skipping preregistration",
                ontology_file_path,
                acronym,
                exc_info=True,
            )
            continue

        if exists:
            LOGGER.info("text2term cache hit for %s (acronym=%s)", ontology_file_path, acronym)
            continue

        try:
            with stage_timer("text2term_cache_build") as t:
                build_text2term_cache_for_owl(ontology_file_path, acronym, TEXT2TERM_CACHE_DIR)
            disk_io.text2term_cache_build_sec.append(t.elapsed_sec)
            LOGGER.info("text2term cache built for %s (acronym=%s)", ontology_file_path, acronym)
        except (OSError, AttributeError, RuntimeError):
            LOGGER.warning(
                "text2term cache_ontology failed for %s (acronym=%s); falling back to per-call OWL parse",
                ontology_file_path,
                acronym,
                exc_info=True,
            )

    return disk_io
