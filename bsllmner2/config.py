import logging
import logging.config
import os
from pathlib import Path

from pydantic import BaseModel

MODULE_ROOT = Path(__file__).parent
REPO_ROOT = MODULE_ROOT.parent
FILTER_KEY_VAL_RULES_PATH = MODULE_ROOT.joinpath("bs", "filter_key_val_rules.json")
PROMPT_EXTRACT_FILE_PATH = MODULE_ROOT.joinpath("prompt", "prompt_extract.yml")
PROMPT_SELECT_FILE_PATH = MODULE_ROOT.joinpath("prompt", "prompt_select.yml")
RESULT_DIR = REPO_ROOT.joinpath("bsllmner2-results")


class Config(BaseModel):
    """
    Application configuration for bsllmner2.
    """
    ollama_host: str = "http://localhost:11434"
    debug: bool = False


default_config = Config()
ENV_PREFIX = "BSLLMNER2_"


def get_config() -> Config:
    """
    Get the application configuration.
    This function can be extended to load configuration from environment variable.

    Returns:
        Config: The current application configuration.
    """
    return Config(
        ollama_host=os.environ.get("OLLAMA_HOST", default_config.ollama_host),
        debug=bool(os.environ.get(f"{ENV_PREFIX}DEBUG", default_config.debug)),
    )


# === logging ===

def set_logging_config() -> None:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr"
            }
        },
        "loggers": {
            "bsllmner2": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            }
        }
    }

    logging.config.dictConfig(config)


set_logging_config()
LOGGER = logging.getLogger("bsllmner2")


def set_logging_level(debug: bool = False) -> None:
    """
    Set the logging level for the application.
    Args:
        debug (bool): If True, set logging level to DEBUG; otherwise, set to INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    LOGGER.setLevel(level)
