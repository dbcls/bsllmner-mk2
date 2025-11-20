import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import List, Set, Tuple

from bsllmner2.client.ollama import ner, select
from bsllmner2.config import (LOGGER, RESUME_BATCH_SIZE, Config,
                              default_config, get_config, set_logging_level)
from bsllmner2.metrics import LiveMetricsCollector
from bsllmner2.schema import (CliSelectArgs, LlmOutput, RunMetadata,
                              SelectResult)
from bsllmner2.utils import (build_extract_prompt_for_select,
                             build_extract_schema_for_select,
                             dump_extract_result, dump_extract_resume_file,
                             dump_select_result, dump_select_resume_file,
                             evaluate_output, get_now_str, load_bs_entries,
                             load_extract_resume_file, load_mapping,
                             load_select_config, load_select_resume_file,
                             remove_resume_files, to_result)


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
        help="Path to the mapping file in TSV format.",
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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run for identification purposes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last incomplete run if possible.",
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
    if parsed_args.ollama_host:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug

    # Select mode specific checks
    if not parsed_args.select_config.exists():
        raise FileNotFoundError(f"Select configuration file {parsed_args.select_config} does not exist.")

    return config, CliSelectArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        mapping=parsed_args.mapping.resolve() if parsed_args.mapping else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        with_metrics=parsed_args.with_metrics,
        run_name=parsed_args.run_name,
        resume=parsed_args.resume,
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

    # for Select mode
    select_config = load_select_config(args.select_config)
    LOGGER.debug("Select Config:\n%s", select_config.model_dump_json(indent=2))

    mapping = load_mapping(args.mapping) if args.mapping else None
    format_ = build_extract_schema_for_select(select_config)
    prompt = build_extract_prompt_for_select(select_config)

    start_time = get_now_str()
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.model}_{start_time}"

    bs_entries = load_bs_entries(args.bs_entries)
    if args.max_entries is not None:
        bs_entries = bs_entries[:args.max_entries]

    extract_outputs: List[LlmOutput] = []
    select_results: List[SelectResult] = []
    if args.resume:
        resume_extract_outputs = load_extract_resume_file(run_name)
        extract_outputs.extend(resume_extract_outputs)
        resume_select_results = load_select_resume_file(run_name)
        select_results.extend(resume_select_results)
        done_ids = Set([output.accession for output in resume_extract_outputs])
        if done_ids:
            LOGGER.info("Skipping %d already processed entries.", len(done_ids))
            bs_entries = [entry for entry in bs_entries if entry.get("accession") not in done_ids]

    if args.with_metrics:
        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()
    try:
        LOGGER.info("Processing %d BioSample entries...", len(bs_entries))
        batches = math.ceil(len(bs_entries) / RESUME_BATCH_SIZE)
        for batch_idx in range(batches):
            start_idx = batch_idx * RESUME_BATCH_SIZE
            end_idx = min(start_idx + RESUME_BATCH_SIZE, len(bs_entries))
            LOGGER.info("Processing batch %d/%d: entries %d to %d", batch_idx + 1, batches, start_idx + 1, end_idx)
            batch_entries = bs_entries[start_idx:end_idx]
            batch_extract_outputs = await ner(config, batch_entries, prompt, format_, args.model, args.thinking)
            extract_outputs.extend(batch_extract_outputs)
            batch_select_results = await select(config, batch_entries, args.model, batch_extract_outputs, select_config, args.thinking)
            select_results.extend(batch_select_results)

            # Dump intermediate results for resuming
            dump_extract_resume_file(extract_outputs, run_name)
            dump_select_resume_file(select_results, run_name)

        end_time = get_now_str()
    finally:
        if args.with_metrics:
            metrics_collector.stop()
    metrics = metrics_collector.get_records() if args.with_metrics else None

    if mapping is not None:
        evaluation = evaluate_output(extract_outputs, mapping)
    else:
        evaluation = []
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

    extract_result_file = dump_extract_result(result, run_name)
    select_result_file = dump_select_result(select_results, run_name)
    remove_resume_files(run_name)
    LOGGER.info("Processing complete. Result saved to %s and %s", extract_result_file, select_result_file)


def run_cli_select() -> None:
    """
    Run the CLI for bsllmner2 select mode in an event loop.
    """
    asyncio.run(run_cli_select_async())


if __name__ == "__main__":
    run_cli_select()
