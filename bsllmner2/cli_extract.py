import argparse
import asyncio
import sys
import time
from pathlib import Path

from ollama import ChatResponse

from bsllmner2.benchmark import (
    aggregate_llm_timings,
    log_performance_summary,
    stage_timer,
)
from bsllmner2.cli_common import (
    BatchInfo,
    add_common_arguments,
    build_config,
    build_run_metadata,
    generate_run_name,
    load_and_trim_entries,
    process_batches,
    run_with_lifecycle,
    validate_common_args,
)
from bsllmner2.config import (
    LOGGER,
    PROGRESS_DIR,
    PROMPT_EXTRACT_FILE_PATH,
    Config,
    set_logging_config,
    set_logging_level,
)
from bsllmner2.io import (
    dump_extract_result,
    dump_extract_resume_file,
    load_extract_resume_file,
    load_format_schema,
    load_prompt_file,
    remove_resume_files,
    validate_extract_resume_file,
)
from bsllmner2.llm import OllamaBackend, ner
from bsllmner2.models import CliExtractArgs, ExtractEntry, PerformanceSummary, StageTimings
from bsllmner2.pipeline import build_extract_result, populate_run_metadata


def parse_args(args: list[str]) -> tuple[Config, CliExtractArgs]:
    """Parse command-line arguments for the bsllmner2 CLI extract mode.

    Returns:
        Tuple of Config and CliExtractArgs.

    """
    parser = argparse.ArgumentParser(
        description="Named Entity Recognition (NER) of biological terms in BioSample records using LLMs.",
    )

    # Add common arguments
    add_common_arguments(parser)

    # Extract mode specific arguments
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROMPT_EXTRACT_FILE_PATH,
        help="Path to the prompt file in YAML format.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Path to the JSON schema file for the output format.",
    )

    parsed_args = parser.parse_args(args)

    # Validate common arguments
    validate_common_args(parser, parsed_args)

    # Validate extract-specific arguments
    if not parsed_args.prompt.exists():
        parser.error(f"Prompt file {parsed_args.prompt} does not exist.")
    if parsed_args.format is not None:
        parsed_args.format = Path(parsed_args.format)
        if not parsed_args.format.exists():
            parser.error(f"Format schema file {parsed_args.format} does not exist.")

    config = build_config(parsed_args)

    return config, CliExtractArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        prompt=parsed_args.prompt.resolve(),
        format=parsed_args.format.resolve() if parsed_args.format is not None else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        run_name=parsed_args.run_name,
        num_ctx=parsed_args.num_ctx,
        resume=parsed_args.resume,
        batch_size=parsed_args.batch_size,
    )


async def run_cli_extract_async() -> None:
    """Run the CLI for bsllmner2 extract mode."""
    LOGGER.info("Starting bsllmner2 CLI extract mode...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.info("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.info("Args:\n%s", args.model_dump_json(indent=2))

    backend = OllamaBackend(config.ollama_host)

    prompt = load_prompt_file(args.prompt)
    format_ = load_format_schema(args.format) if args.format else None

    run_name, start_time = generate_run_name(args.model, args.run_name)
    bs_entries = load_and_trim_entries(args.bs_entries, args.max_entries)
    total_entries = len(bs_entries)

    extract_outputs: list[ExtractEntry] = []
    all_chat_responses: list[ChatResponse] = []
    stage_timings_list: list[StageTimings] = []

    if args.resume:
        resume_extract_outputs = load_extract_resume_file(run_name)
        done_ids = validate_extract_resume_file(resume_extract_outputs, run_name)
        extract_outputs.extend(resume_extract_outputs)
        if done_ids:
            LOGGER.info("Skipping %d already processed entries.", len(done_ids))
            bs_entries = [entry for entry in bs_entries if entry.get("accession") not in done_ids]

    wall_start = time.perf_counter()

    async with run_with_lifecycle() as run_state:

        async def process_extract_batch(
            batch_info: BatchInfo,
        ) -> tuple[list[ExtractEntry], list[ChatResponse], float]:
            with stage_timer("ner") as t_ner:
                batch_outputs, batch_chat_responses = await ner(
                    backend, batch_info.entries, prompt, format_, args.model, args.thinking,
                    num_ctx=args.num_ctx,
                )
            if len(batch_outputs) < len(batch_info.entries):
                LOGGER.error(
                    "Batch returned %d outputs for %d entries (%d lost)",
                    len(batch_outputs),
                    len(batch_info.entries),
                    len(batch_info.entries) - len(batch_outputs),
                )

            return batch_outputs, batch_chat_responses, t_ner.elapsed_sec

        def on_extract_batch_complete(
            batch_idx: int,
            batch_result: tuple[list[ExtractEntry], list[ChatResponse], float],
        ) -> None:
            batch_outputs, batch_chat_responses, ner_sec = batch_result
            with stage_timer("resume_write") as t_resume:
                extract_outputs.extend(batch_outputs)
                all_chat_responses.extend(batch_chat_responses)
                dump_extract_resume_file(extract_outputs, run_name)
            stage_timings_list.append(
                StageTimings(
                    batch_idx=batch_idx,
                    batch_size=len(batch_outputs),
                    ner_sec=ner_sec,
                    resume_write_sec=t_resume.elapsed_sec,
                ),
            )

        await process_batches(
            entries=bs_entries,
            batch_size=args.batch_size,
            process_fn=process_extract_batch,
            on_batch_complete=on_extract_batch_complete,
            log_prefix="Processing",
        )

    total_wall_sec = time.perf_counter() - wall_start
    status = run_state.status
    end_time = run_state.end_time

    run_metadata = build_run_metadata(run_name, args.model, args.thinking, start_time, end_time, status)
    run_metadata = populate_run_metadata(run_metadata, extract_outputs)

    # Build performance summary
    ner_timing = aggregate_llm_timings(all_chat_responses)
    performance = PerformanceSummary(
        total_input_entries=total_entries,
        completed_count=len(extract_outputs),
        total_wall_sec=total_wall_sec,
        stage_timings=stage_timings_list,
        ner_llm_timing=ner_timing if ner_timing.call_count > 0 else None,
    )

    result = build_extract_result(
        entries=extract_outputs,
        run_metadata=run_metadata,
        performance=performance,
    )

    extract_result_file = dump_extract_result(result, run_name)
    if status == "completed":
        remove_resume_files(run_name)
    LOGGER.info("Processing %s. Result saved to %s", status, extract_result_file)
    log_performance_summary(performance)


def run_cli_extract() -> None:
    """Run the CLI for bsllmner2 extract mode in an event loop."""
    set_logging_config()
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_cli_extract_async())


if __name__ == "__main__":
    run_cli_extract()
