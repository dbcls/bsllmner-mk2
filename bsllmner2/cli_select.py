import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

from bsllmner2.client.ollama import ner, select
from bsllmner2.config import (LOGGER, PROMPT_EXTRACT_FILE_PATH, Config,
                              default_config, get_config, set_logging_level)
from bsllmner2.metrics import LiveMetricsCollector
from bsllmner2.schema import CliSelectArgs, Result, RunMetadata
from bsllmner2.utils import (dump_result, evaluate_output, get_now_str,
                             load_bs_entries, load_format_schema, load_mapping,
                             load_prompt_file, load_select_config, to_result)


def parse_args(args: List[str]) -> Tuple[Config, CliSelectArgs]:
    """
    Parse command-line arguments for the bsllmner2 CLI select mode.

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
        "--format",
        default=None,
        help="""\
            Path to the JSON schema file for the output format.
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

    # Select mode specific arguments
    parser.add_argument(
        "--select-config",
        type=Path,
        required=True,
        help="Path to the select configuration file in JSON format.",
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
    if parsed_args.format is not None:
        parsed_args.format = Path(parsed_args.format)
        if not parsed_args.format.exists():
            raise FileNotFoundError(f"Format schema file {parsed_args.format} does not exist.")
    if parsed_args.ollama_host:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug

    # Select mode specific checks
    if not parsed_args.select_config.exists():
        raise FileNotFoundError(f"Select configuration file {parsed_args.select_config} does not exist.")

    return config, CliSelectArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        mapping=parsed_args.mapping.resolve(),
        prompt=parsed_args.prompt.resolve(),
        format=parsed_args.format.resolve() if parsed_args.format is not None else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        with_metrics=parsed_args.with_metrics,
        select_config=parsed_args.select_config.resolve()
    )


async def run_cli_select_async() -> None:
    """
    Run the CLI for bsllmner2 select mode.
    """
    LOGGER.info("Starting bsllmner2 CLI select mode...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.debug("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.debug("Args:\n%s", args.model_dump_json(indent=2))

    bs_entries = load_bs_entries(args.bs_entries)
    if args.max_entries is not None:
        bs_entries = bs_entries[:args.max_entries]
    mapping = load_mapping(args.mapping)
    prompt = load_prompt_file(args.prompt)
    format_ = load_format_schema(args.format) if args.format else None

    # for Select mode
    select_config = load_select_config(args.select_config)
    LOGGER.debug("Select Config:\n%s", select_config.model_dump_json(indent=2))

    if args.with_metrics:
        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()
    try:
        start_time = get_now_str()
        # Debug
        file = Path("/app/bsllmner2-results/select_llama3.1:70b_20251008_084112.json")
        with file.open("r") as f:
            extract_result = Result.model_validate_json(f.read())
        extract_outputs = extract_result.output
        # extract_outputs = await ner(config, bs_entries, prompt, format_, args.model, args.thinking)
        await select(config, bs_entries, args.model, extract_outputs, select_config, args.thinking)
        end_time = get_now_str()
    finally:
        if args.with_metrics:
            metrics_collector.stop()
    metrics = metrics_collector.get_records() if args.with_metrics else None

    evaluation = evaluate_output(extract_outputs, mapping)
    run_name = f"select_{args.model}_{start_time}"
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
        output=extract_outputs,
        evaluation=evaluation,
        config=config,
        run_metadata=run_metadata,
        format_=format_,
        thinking=args.thinking,
        args=args,
        metrics=metrics,
    )

    result_file = dump_result(result, run_name)
    LOGGER.info("Processing complete. Result saved to %s", result_file)


def run_cli_select() -> None:
    """
    Run the CLI for bsllmner2 select mode in an event loop.
    """
    asyncio.run(run_cli_select_async())


if __name__ == "__main__":
    run_cli_select()
