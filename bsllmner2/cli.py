import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ollama import ChatResponse
from pydantic import BaseModel

from bsllmner2.client.ollama import OLLAMA_MODELS, Output, ner
from bsllmner2.config import (LOGGER, PROMPT_EXTRACT_FILE_PATH, RESULT_DIR,
                              Config, default_config, get_config,
                              set_logging_level)
from bsllmner2.prompt import load_prompt_file
from bsllmner2.utils import load_bs_entries


class Args(BaseModel):
    """
    Command-line arguments for the bsllmner2 CLI.
    """
    bs_entries: Path
    prompt: Path
    model: str = OLLAMA_MODELS[0]
    max_entries: Optional[int] = None


def parse_args(args: List[str]) -> Tuple[Config, Args]:
    """
    Parse command-line arguments for the bsllmner2 CLI.

    Returns:
        Args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Named Entity Recognition (NER) of biological terms in BioSample records using LLMs, developed as bsllmner-mk2.",
    )
    parser.add_argument(
        "--bs-entries",
        type=Path,
        required=True,
        help="Path to the input JSON or JSONL file containing BioSample entries.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROMPT_EXTRACT_FILE_PATH,
        help="""\
            Path to the prompt file in YAML format.
            Default is 'prompt/prompt_extract.yml' relative to the project root.
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=OLLAMA_MODELS,
        default=OLLAMA_MODELS[0],
        help=f"LLM model to use for NER. Available models: {', '.join(OLLAMA_MODELS)}"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=-1,
        help="""\
            Process only the first N entries from the input file.
            Default is -1, which means all entries will be processed.
        """,
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help=f"Host URL for the Ollama server (default: {default_config.ollama_host}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose logging.",
    )

    parsed_args = parser.parse_args(args)

    # Priority: CLI argument > Environment variable > Default config (from config.py)
    config = get_config()
    if not parsed_args.bs_entries.exists():
        raise FileNotFoundError(f"BioSample entries file {parsed_args.bs_entries} does not exist.")
    if not parsed_args.prompt.exists():
        raise FileNotFoundError(f"Prompt file {parsed_args.prompt} does not exist.")
    if parsed_args.ollama_host:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug

    return config, Args(
        bs_entries=parsed_args.bs_entries.resolve(),
        prompt=parsed_args.prompt.resolve(),
        model=parsed_args.model,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None
    )


def dump_results(config: Config, args: Args, results: List[Tuple[ChatResponse, Output]]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y%m%d_%H%M%S")
    results_file = RESULT_DIR.joinpath(f"results_{args.model}_{now}.json")
    outputs_only = [output.output for _, output in results]
    with results_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "args": args.model_dump(mode="json"),
                "outputs_only": outputs_only,
                "results": [{
                    "chat_response": response.model_dump(),
                    "output": output.model_dump()
                } for response, output in results]
            },
            indent=2, ensure_ascii=False,
        ))

    return results_file


def run_cli() -> None:
    """
    Run the CLI for bsllmner2.
    """
    LOGGER.info("Starting bsllmner2 CLI...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.debug("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.debug("Args:\n%s", args.model_dump_json(indent=2))

    bs_entries = load_bs_entries(args.bs_entries)
    if args.max_entries is not None:
        bs_entries = bs_entries[:args.max_entries]
    prompts = load_prompt_file(args.prompt)
    results = ner(config, bs_entries, prompts, args.model)
    results_file = dump_results(config, args, results)
    LOGGER.info("Processing complete. Results saved to %s", results_file)


if __name__ == "__main__":
    run_cli()
