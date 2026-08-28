import logging
import logging.config
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from bsllmner2.utils import parse_bool

MODULE_ROOT = Path(__file__).parent.resolve()
FILTER_KEYS_PATH = MODULE_ROOT.joinpath("filter_keys.json")
PROMPT_EXTRACT_FILE_PATH = MODULE_ROOT.joinpath("prompts", "extract.yml")
SCHEMA_CELL_LINE_FILE_PATH = MODULE_ROOT.joinpath("format", "cell_line.schema.json")
RESULT_DIR = Path(os.environ.get("BSLLMNER2_RESULT_DIR", str(Path.cwd().joinpath("bsllmner2-results"))))
EXTRACT_RESULT_DIR = RESULT_DIR.joinpath("extract")
SELECT_RESULT_DIR = RESULT_DIR.joinpath("select")
TMP_DIR = Path(
    os.environ.get("BSLLMNER2_TMP_DIR", str(Path(tempfile.gettempdir()).joinpath(f"bsllmner2-{os.getuid()}")))
)

RESUME_BATCH_SIZE = 1024
DEFAULT_NUM_CTX = 4096
DEFAULT_OLLAMA_CONCURRENCY = 256
OLLAMA_CONCURRENCY_ENV = "BSLLMNER2_OLLAMA_CONCURRENCY"


def resolve_default_ollama_concurrency() -> int:
    """Read ``BSLLMNER2_OLLAMA_CONCURRENCY`` from the environment at call time.

    Resolving lazily (rather than at import) is required for tests that use
    ``monkeypatch.setenv`` to override the concurrency limit after import.
    """
    raw = os.environ.get(OLLAMA_CONCURRENCY_ENV)
    if raw is None:
        return DEFAULT_OLLAMA_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning(
            "Invalid %s=%r; falling back to default %d.",
            OLLAMA_CONCURRENCY_ENV,
            raw,
            DEFAULT_OLLAMA_CONCURRENCY,
        )
        return DEFAULT_OLLAMA_CONCURRENCY
    if value < 1:
        LOGGER.warning(
            "%s must be >= 1 (got %d); falling back to default %d.",
            OLLAMA_CONCURRENCY_ENV,
            value,
            DEFAULT_OLLAMA_CONCURRENCY,
        )
        return DEFAULT_OLLAMA_CONCURRENCY
    return value


# Trial flags for the Stage 2a / Stage 3 candidate-handling experiments.
# Both default to the shipped behaviour. Read lazily so tests and per-run env vars apply.
EXPOSE_EXACT_MATCH_ENV = "BSLLMNER2_EXPOSE_EXACT_MATCH"
DISABLE_AUTOPICK_ENV = "BSLLMNER2_DISABLE_AUTOPICK"


def expose_exact_match() -> bool:
    """Whether Stage 3 candidates carry their ``exact_match`` flag.

    Without it the LLM cannot tell a value that matched a registered label or synonym
    from one that only surfaced through a fuzzy score.
    """
    return parse_bool(os.environ.get(EXPOSE_EXACT_MATCH_ENV, "false"))


def disable_autopick() -> bool:
    """Whether Stage 2a skips its ``single`` verdict and defers every value to Stage 3.

    Auto-pick commits on a name match alone, so an extracted parent-line name confirmed by
    the ontology never reaches the stage that could weigh it against the sample metadata.
    """
    return parse_bool(os.environ.get(DISABLE_AUTOPICK_ENV, "false"))


class Config(BaseModel):
    """Application configuration for bsllmner2."""

    ollama_host: str = "http://localhost:11434"
    debug: bool = False


default_config = Config()
ENV_PREFIX = "BSLLMNER2_"


def _parse_bool_env(value: str | bool) -> bool:
    """Parse environment variable value as boolean (non-strict)."""
    return parse_bool(value, strict=False)


def get_config() -> Config:
    """Get the application configuration.

    This function can be extended to load configuration from environment variable.

    Returns:
        Config: The current application configuration.

    """
    debug_env = os.environ.get(f"{ENV_PREFIX}DEBUG")
    debug = _parse_bool_env(debug_env) if debug_env is not None else default_config.debug

    return Config(
        ollama_host=os.environ.get("OLLAMA_HOST", default_config.ollama_host),
        debug=debug,
    )


# === logging ===


def set_logging_config() -> None:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stderr"},
        },
        "loggers": {"bsllmner2": {"handlers": ["default"], "level": "INFO", "propagate": False}},
    }

    logging.config.dictConfig(config)


LOGGER = logging.getLogger("bsllmner2")


def set_logging_level(debug: bool = False) -> None:
    """Set the logging level for the application.

    Args:
        debug (bool): If True, set logging level to DEBUG; otherwise, set to INFO.

    """
    level = logging.DEBUG if debug else logging.INFO
    LOGGER.setLevel(level)
