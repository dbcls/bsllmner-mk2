import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

from bsllmner2.client.ollama import ner
from bsllmner2.config import (LOGGER, PROMPT_EXTRACT_FILE_PATH, Config,
                              default_config, get_config, set_logging_level)
from bsllmner2.metrics import LiveMetricsCollector
from bsllmner2.schema import CliExtractArgs, RunMetadata
from bsllmner2.utils import (dump_result, evaluate_output, get_now_str,
                             load_bs_entries, load_mapping, load_prompt_file,
                             to_result)


def parse_args(args: List[str]) -> Tuple[Config, CliExtractArgs]:
    """
    Parse command-line arguments for the bsllmner2 CLI extract mode.

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
        "--mapping",
        type=Path,
        required=True,
        help="Path to the mapping file in TSV format.",
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
        default="llama3.1:70b",
        help="LLM model to use for NER.",
    )
    parser.add_argument(
        "--thinking",
        choices=["true", "false"],
        help="Enable or disable thinking mode for the LLM. Use 'true' to enable thinking, 'false' to disable it.",
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
        "--with-metrics",
        action="store_true",
        help="Enable collection of metrics during processing.",
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
    if not parsed_args.mapping.exists():
        raise FileNotFoundError(f"Mapping file {parsed_args.mapping} does not exist.")
    if not parsed_args.prompt.exists():
        raise FileNotFoundError(f"Prompt file {parsed_args.prompt} does not exist.")
    if parsed_args.ollama_host:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug

    return config, CliExtractArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        mapping=parsed_args.mapping.resolve(),
        prompt=parsed_args.prompt.resolve(),
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        with_metrics=parsed_args.with_metrics,
    )


async def run_cli_extract_async() -> None:
    """
    Run the CLI for bsllmner2 extract mode.
    """
    LOGGER.info("Starting bsllmner2 CLI extract mode...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.debug("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.debug("Args:\n%s", args.model_dump_json(indent=2))

    bs_entries = load_bs_entries(args.bs_entries)
    if args.max_entries is not None:
        bs_entries = bs_entries[:args.max_entries]
    mapping = load_mapping(args.mapping)
    prompt = load_prompt_file(args.prompt)

    if args.with_metrics:
        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()
    try:
        start_time = get_now_str()
        output = await ner(config, bs_entries, prompt, args.model, args.thinking)
        end_time = get_now_str()
    finally:
        if args.with_metrics:
            metrics_collector.stop()
    metrics = metrics_collector.get_records() if args.with_metrics else None

    evaluation = evaluate_output(output, mapping)
    run_name = f"extract_{args.model}_{start_time}"
    run_metadata = RunMetadata(
        run_name=run_name,
        username=None,
        model=args.model,
        thinking=args.thinking,
        start_time=start_time,
        end_time=end_time,
        status="completed"
    )
    result = to_result(
        bs_entries=bs_entries,
        mapping=mapping,
        prompt=prompt,
        model=args.model,
        thinking=args.thinking,
        output=output,
        evaluation=evaluation,
        config=config,
        args=args,
        metrics=metrics,
        run_metadata=run_metadata,
    )

    result_file = dump_result(result, run_name)
    LOGGER.info("Processing complete. Result saved to %s", result_file)


def run_cli_extract() -> None:
    """
    Run the CLI for bsllmner2 extract mode in an event loop.
    """
    asyncio.run(run_cli_extract_async())


if __name__ == "__main__":
    run_cli_extract()
