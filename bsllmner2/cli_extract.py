import argparse
import asyncio
import sys
from pathlib import Path

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
    load_mapping,
    load_prompt_file,
    remove_resume_files,
    validate_extract_resume_file,
)
from bsllmner2.llm import OllamaBackend, ner
from bsllmner2.metrics import LiveMetricsCollector
from bsllmner2.models import CliExtractArgs, LlmOutput
from bsllmner2.pipeline import evaluate_output, to_result


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
        mapping=parsed_args.mapping.resolve() if parsed_args.mapping else None,
        prompt=parsed_args.prompt.resolve(),
        format=parsed_args.format.resolve() if parsed_args.format is not None else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        with_metrics=parsed_args.with_metrics,
        run_name=parsed_args.run_name,
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

    mapping = load_mapping(args.mapping) if args.mapping else None
    prompt = load_prompt_file(args.prompt)
    format_ = load_format_schema(args.format) if args.format else None

    run_name, start_time = generate_run_name(args.model, args.run_name)
    bs_entries = load_and_trim_entries(args.bs_entries, args.max_entries)
    all_bs_entries = bs_entries

    extract_outputs: list[LlmOutput] = []
    if args.resume:
        resume_extract_outputs = load_extract_resume_file(run_name)
        done_ids = validate_extract_resume_file(resume_extract_outputs, run_name)
        extract_outputs.extend(resume_extract_outputs)
        if done_ids:
            LOGGER.info("Skipping %d already processed entries.", len(done_ids))
            bs_entries = [entry for entry in bs_entries if entry.get("accession") not in done_ids]

    metrics_collector = None
    if args.with_metrics:
        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()

    async with run_with_lifecycle(metrics_collector) as run_state:

        async def process_extract_batch(batch_info: BatchInfo) -> list[LlmOutput]:
            batch_outputs = await ner(backend, batch_info.entries, prompt, format_, args.model, args.thinking)
            if len(batch_outputs) < len(batch_info.entries):
                LOGGER.error(
                    "Batch returned %d outputs for %d entries (%d lost)",
                    len(batch_outputs),
                    len(batch_info.entries),
                    len(batch_info.entries) - len(batch_outputs),
                )

            return batch_outputs

        def on_extract_batch_complete(_batch_idx: int, batch_outputs: list[LlmOutput]) -> None:
            extract_outputs.extend(batch_outputs)
            dump_extract_resume_file(extract_outputs, run_name)

        await process_batches(
            entries=bs_entries,
            batch_size=args.batch_size,
            process_fn=process_extract_batch,
            on_batch_complete=on_extract_batch_complete,
            log_prefix="Processing",
        )

    status = run_state.status
    end_time = run_state.end_time
    metrics = metrics_collector.get_records() if metrics_collector else None

    if mapping is not None:
        evaluation = evaluate_output(extract_outputs, mapping)
    else:
        evaluation = []
    run_metadata = build_run_metadata(run_name, args.model, args.thinking, start_time, end_time, status)
    result = to_result(
        bs_entries=all_bs_entries,
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
    if status == "completed":
        remove_resume_files(run_name)
    LOGGER.info("Processing %s. Result saved to %s", status, extract_result_file)


def run_cli_extract() -> None:
    """Run the CLI for bsllmner2 extract mode in an event loop."""
    set_logging_config()
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_cli_extract_async())


if __name__ == "__main__":
    run_cli_extract()
