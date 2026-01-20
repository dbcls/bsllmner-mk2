import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

from bsllmner2.cli_common import (BatchInfo, add_common_arguments,
                                  process_batches, validate_common_args)
from bsllmner2.client.ollama import build_index_map, ner, select
from bsllmner2.config import LOGGER, Config, get_config, set_logging_level
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
                             remove_resume_files, to_result,
                             validate_resume_consistency)


def parse_args(args: List[str]) -> Tuple[Config, CliSelectArgs]:
    """
    Parse command-line arguments for the bsllmner2 CLI select mode.

    Returns:
        Tuple of Config and CliSelectArgs.
    """
    parser = argparse.ArgumentParser(
        description="Named Entity Recognition (NER) and ontology mapping for BioSample records using LLMs.",
    )

    # Add common arguments
    add_common_arguments(parser)

    # Select mode specific arguments
    parser.add_argument(
        "--select-config",
        type=Path,
        required=True,
        help="Path to the select configuration file in JSON format.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_false",
        dest="include_reasoning",
        help="Disable reasoning step during selection.",
    )

    parsed_args = parser.parse_args(args)

    # Validate common arguments
    validate_common_args(parsed_args)

    # Validate select-specific arguments
    if not parsed_args.select_config.exists():
        raise FileNotFoundError(f"Select configuration file {parsed_args.select_config} does not exist.")

    # Build config (CLI args override environment/defaults)
    config = get_config()
    if parsed_args.ollama_host:
        config.ollama_host = parsed_args.ollama_host
    config.debug = parsed_args.debug

    return config, CliSelectArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        mapping=parsed_args.mapping.resolve() if parsed_args.mapping else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        with_metrics=parsed_args.with_metrics,
        run_name=parsed_args.run_name,
        resume=parsed_args.resume,
        batch_size=parsed_args.batch_size,
        select_config=parsed_args.select_config.resolve(),
        include_reasoning=parsed_args.include_reasoning,
    )


async def run_cli_select_async() -> None:
    """
    Run the CLI for bsllmner2 select mode.
    """
    from bsllmner2.errors import Bsllmner2Error

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
        resume_select_results = load_select_resume_file(run_name)

        # Validate consistency and detect orphans
        done_ids, orphan_ids = validate_resume_consistency(
            resume_extract_outputs, resume_select_results, run_name
        )

        if orphan_ids:
            LOGGER.warning(
                "Found %d orphan entries (extract completed but select not completed). "
                "These will be reprocessed: %s%s",
                len(orphan_ids),
                sorted(orphan_ids)[:5],
                "..." if len(orphan_ids) > 5 else ""
            )

        # Keep only fully completed entries
        select_results.extend(resume_select_results)
        extract_outputs = [o for o in resume_extract_outputs if o.accession in done_ids]

        if done_ids:
            LOGGER.info("Skipping %d already processed entries.", len(done_ids))
            bs_entries = [entry for entry in bs_entries if entry.get("accession") not in done_ids]

    select_index_map = build_index_map(select_config)

    metrics_collector = None
    if args.with_metrics:
        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()

    status = "completed"
    end_time = None
    try:
        async def process_select_batch(
            batch_info: BatchInfo,
        ) -> Tuple[List[LlmOutput], List[SelectResult]]:
            LOGGER.info("Extracting entities...")
            batch_extract_outputs = await ner(
                config, batch_info.entries, prompt, format_, args.model, args.thinking
            )
            LOGGER.info("Selecting entities...")
            batch_select_results = await select(
                config,
                batch_info.entries,
                args.model,
                batch_extract_outputs,
                select_config,
                args.thinking,
                include_reasoning=args.include_reasoning,
                index_map=select_index_map,
            )
            return batch_extract_outputs, batch_select_results

        def on_select_batch_complete(
            batch_idx: int,
            batch_result: Tuple[List[LlmOutput], List[SelectResult]],
        ) -> None:
            batch_extract_outputs, batch_select_results = batch_result
            extract_outputs.extend(batch_extract_outputs)
            select_results.extend(batch_select_results)
            # Dump both files atomically to ensure consistency
            dump_extract_resume_file(extract_outputs, run_name)
            dump_select_resume_file(select_results, run_name)

        await process_batches(
            entries=bs_entries,
            batch_size=args.batch_size,
            process_fn=process_select_batch,
            on_batch_complete=on_select_batch_complete,
            log_prefix="Processing",
        )

        end_time = get_now_str()
    except Bsllmner2Error as e:
        LOGGER.error("Processing failed: %s", e)
        status = "failed"
        end_time = get_now_str()
        raise
    except Exception as e:
        LOGGER.error("Unexpected error during processing: %s", e, exc_info=True)
        status = "failed"
        end_time = get_now_str()
        raise
    finally:
        if metrics_collector is not None:
            metrics_collector.stop()

    metrics = metrics_collector.get_records() if metrics_collector else None

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
        status=status
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
