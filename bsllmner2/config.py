import logging
import logging.config
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

MODULE_ROOT = Path(__file__).parent.resolve()
FILTER_KEYS_PATH = MODULE_ROOT.joinpath("filter_keys.json")
PROMPT_EXTRACT_FILE_PATH = MODULE_ROOT.joinpath("prompt", "prompt_extract.yml")
PROMPT_SELECT_FILE_PATH = MODULE_ROOT.joinpath("prompt", "prompt_select.yml")
SCHEMA_CELL_LINE_FILE_PATH = MODULE_ROOT.joinpath("format", "cell_line.schema.json")
RESULT_DIR = Path(os.environ.get("BSLLMNER2_RESULT_DIR", str(Path.cwd().joinpath("bsllmner2-results"))))
EXTRACT_RESULT_DIR = RESULT_DIR.joinpath("extract")
SELECT_RESULT_DIR = RESULT_DIR.joinpath("select")
TMP_DIR = Path(
    os.environ.get("BSLLMNER2_TMP_DIR", str(Path(tempfile.gettempdir()).joinpath(f"bsllmner2-{os.getuid()}")))
)
PROGRESS_DIR = TMP_DIR.joinpath("progress")

RESUME_BATCH_SIZE = 1024
DEFAULT_NUM_CTX = 4096


class Config(BaseModel):
    """Application configuration for bsllmner2."""

    ollama_host: str = "http://localhost:11434"
    debug: bool = False


default_config = Config()
ENV_PREFIX = "BSLLMNER2_"


def _parse_bool_env(value: str | bool) -> bool:
    """Parse environment variable value as boolean."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "on")


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
