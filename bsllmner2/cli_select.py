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
from bsllmner2.config import LOGGER, PROGRESS_DIR, Config, set_logging_config, set_logging_level
from bsllmner2.io import (
    dump_extract_resume_file,
    dump_select_result,
    dump_select_resume_file,
    load_extract_resume_file,
    load_mapping,
    load_select_config,
    load_select_resume_file,
    remove_resume_files,
    validate_resume_consistency,
)
from bsllmner2.llm import OllamaBackend, ner
from bsllmner2.models import CliSelectArgs, ExtractEntry, PerformanceSummary, SelectEntry, SelectResult, StageTimings
from bsllmner2.pipeline import (
    build_extract_prompt_for_select,
    build_extract_schema_for_select,
    evaluate_select_output,
    populate_run_metadata,
)
from bsllmner2.select import (
    TEXT2TERM_CACHE_DIR,
    SearchMemo,
    SelectStageTimings,
    build_index_map,
    build_text2term_cache,
    select,
)


def parse_args(args: list[str]) -> tuple[Config, CliSelectArgs]:
    """Parse command-line arguments for the bsllmner2 CLI select mode.

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
        "--mapping",
        type=Path,
        help="Path to the mapping file in TSV format (for evaluation).",
    )
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
    validate_common_args(parser, parsed_args)

    # Validate select-specific arguments
    if not parsed_args.select_config.exists():
        parser.error(f"Select configuration file {parsed_args.select_config} does not exist.")

    config = build_config(parsed_args)

    return config, CliSelectArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        mapping=parsed_args.mapping.resolve() if parsed_args.mapping else None,
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries if parsed_args.max_entries >= 0 else None,
        run_name=parsed_args.run_name,
        num_ctx=parsed_args.num_ctx,
        resume=parsed_args.resume,
        batch_size=parsed_args.batch_size,
        select_config=parsed_args.select_config.resolve(),
        include_reasoning=parsed_args.include_reasoning,
    )


async def run_cli_select_async() -> None:
    """Run the CLI for bsllmner2 select mode."""
    LOGGER.info("Starting bsllmner2 CLI select mode...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.info("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.info("Args:\n%s", args.model_dump_json(indent=2))

    backend = OllamaBackend(config.ollama_host)

    # for Select mode
    select_config = load_select_config(args.select_config)
    LOGGER.info("Select Config:\n%s", select_config.model_dump_json(indent=2))

    mapping = load_mapping(args.mapping) if args.mapping else None
    format_ = build_extract_schema_for_select(select_config)
    prompt = build_extract_prompt_for_select(select_config)

    run_name, start_time = generate_run_name(args.model, args.run_name)
    bs_entries = load_and_trim_entries(args.bs_entries, args.max_entries)
    all_bs_entries = bs_entries

    extract_outputs: list[ExtractEntry] = []
    select_results: list[SelectEntry] = []
    all_ner_chat_responses: list[ChatResponse] = []
    all_select_chat_responses: list[ChatResponse] = []
    stage_timings_list: list[StageTimings] = []
    orphan_ids: set[str] = set()
    if args.resume:
        resume_extract_outputs = load_extract_resume_file(run_name)
        resume_select_results = load_select_resume_file(run_name)

        # Validate consistency and detect orphans
        done_ids, orphan_ids = validate_resume_consistency(resume_extract_outputs, resume_select_results, run_name)

        if orphan_ids:
            LOGGER.warning(
                "Found %d orphan entries (extract completed but select not completed). "
                "Select will be re-run for these: %s%s",
                len(orphan_ids),
                sorted(orphan_ids)[:5],
                "..." if len(orphan_ids) > 5 else "",
            )

        # Keep all extract outputs (including orphans)
        extract_outputs.extend(resume_extract_outputs)
        select_results.extend(resume_select_results)

        skip_ids = done_ids | orphan_ids
        if done_ids:
            LOGGER.info("Skipping %d already processed entries.", len(done_ids))
        bs_entries = [entry for entry in bs_entries if entry.get("accession") not in skip_ids]

    select_index_map, disk_io_timings = build_index_map(select_config)
    text2term_disk_io = build_text2term_cache(select_config)
    disk_io_timings.text2term_cache_build_sec.extend(text2term_disk_io.text2term_cache_build_sec)
    disk_io_timings.text2term_cache_load_sec.extend(text2term_disk_io.text2term_cache_load_sec)

    wall_start = time.perf_counter()

    # Run-scoped memoization so identical (field, value) queries resolved in
    # an earlier batch are reused instead of re-running ontology / text2term
    # lookups. Single-threaded asyncio means no locking is needed.
    search_memo: SearchMemo = {}
    t2t_memo: SearchMemo = {}

    async with run_with_lifecycle() as run_state:
        # Re-run select for orphan entries (extract already completed)
        if args.resume and orphan_ids:
            orphan_entries = [e for e in all_bs_entries if e.get("accession") in orphan_ids]
            orphan_extract = [o for o in extract_outputs if o.accession in orphan_ids]
            LOGGER.info("Running select for %d orphan entries...", len(orphan_ids))
            orphan_select, orphan_select_responses, _ = await select(
                backend,
                orphan_entries,
                args.model,
                orphan_extract,
                select_config,
                args.thinking,
                include_reasoning=args.include_reasoning,
                index_map=select_index_map,
                text2term_cache_folder=TEXT2TERM_CACHE_DIR,
                num_ctx=args.num_ctx,
                search_memo=search_memo,
                t2t_memo=t2t_memo,
            )
            select_results.extend(orphan_select)
            all_select_chat_responses.extend(orphan_select_responses)
            dump_extract_resume_file(extract_outputs, run_name)
            dump_select_resume_file(select_results, run_name)

        async def process_select_batch(
            batch_info: BatchInfo,
        ) -> tuple[
            list[ExtractEntry], list[SelectEntry], list[ChatResponse], list[ChatResponse], float, SelectStageTimings
        ]:
            LOGGER.info("Extracting entities...")
            with stage_timer("ner") as t_ner:
                batch_extract_outputs, batch_ner_responses = await ner(
                    backend,
                    batch_info.entries,
                    prompt,
                    format_,
                    args.model,
                    args.thinking,
                    num_ctx=args.num_ctx,
                )
            if len(batch_extract_outputs) < len(batch_info.entries):
                LOGGER.error(
                    "Batch returned %d extract outputs for %d entries (%d lost)",
                    len(batch_extract_outputs),
                    len(batch_info.entries),
                    len(batch_info.entries) - len(batch_extract_outputs),
                )
            LOGGER.info("Selecting entities...")
            batch_select_results, batch_select_responses, select_timings = await select(
                backend,
                batch_info.entries,
                args.model,
                batch_extract_outputs,
                select_config,
                args.thinking,
                include_reasoning=args.include_reasoning,
                index_map=select_index_map,
                text2term_cache_folder=TEXT2TERM_CACHE_DIR,
                num_ctx=args.num_ctx,
                search_memo=search_memo,
                t2t_memo=t2t_memo,
            )

            return (
                batch_extract_outputs,
                batch_select_results,
                batch_ner_responses,
                batch_select_responses,
                t_ner.elapsed_sec,
                select_timings,
            )

        def on_select_batch_complete(
            batch_idx: int,
            batch_result: tuple[
                list[ExtractEntry], list[SelectEntry], list[ChatResponse], list[ChatResponse], float, SelectStageTimings
            ],
        ) -> None:
            (
                batch_extract_outputs,
                batch_select_results,
                batch_ner_responses,
                batch_select_responses,
                ner_sec,
                select_timings,
            ) = batch_result
            with stage_timer("resume_write") as t_resume:
                extract_outputs.extend(batch_extract_outputs)
                select_results.extend(batch_select_results)
                all_ner_chat_responses.extend(batch_ner_responses)
                all_select_chat_responses.extend(batch_select_responses)
                # Dump both files atomically to ensure consistency
                dump_extract_resume_file(extract_outputs, run_name)
                dump_select_resume_file(select_results, run_name)
            stage_timings_list.append(
                StageTimings(
                    batch_idx=batch_idx,
                    batch_size=len(batch_extract_outputs),
                    ner_sec=ner_sec,
                    ontology_search_sec=select_timings["ontology_search_sec"],
                    text2term_sec=select_timings["text2term_sec"],
                    llm_select_sec=select_timings["llm_select_sec"],
                    resume_write_sec=t_resume.elapsed_sec,
                ),
            )

        await process_batches(
            entries=bs_entries,
            batch_size=args.batch_size,
            process_fn=process_select_batch,
            on_batch_complete=on_select_batch_complete,
            log_prefix="Processing",
        )

    total_wall_sec = time.perf_counter() - wall_start
    status = run_state.status
    end_time = run_state.end_time

    select_metrics = evaluate_select_output(select_results, mapping) if mapping is not None else None

    run_metadata = build_run_metadata(run_name, args.model, args.thinking, start_time, end_time, status)
    run_metadata_populated = populate_run_metadata(run_metadata, extract_outputs)

    # Build performance summary
    ner_timing = aggregate_llm_timings(all_ner_chat_responses)
    select_timing = aggregate_llm_timings(all_select_chat_responses)

    performance = PerformanceSummary(
        total_input_entries=len(all_bs_entries),
        completed_count=len(extract_outputs),
        total_wall_sec=total_wall_sec,
        stage_timings=stage_timings_list,
        ner_llm_timing=ner_timing if ner_timing.call_count > 0 else None,
        select_llm_timing=select_timing if select_timing.call_count > 0 else None,
        disk_io=disk_io_timings,
    )

    select_result = SelectResult(
        entries=select_results,
        run_metadata=run_metadata_populated,
        evaluation=select_metrics,
        performance=performance,
    )

    select_result_file = dump_select_result(select_result, run_name)
    if status == "completed":
        remove_resume_files(run_name)
    LOGGER.info("Processing %s. Result saved to %s", status, select_result_file)
    log_performance_summary(performance, select_metrics)


def run_cli_select() -> None:
    """Run the CLI for bsllmner2 select mode in an event loop."""
    set_logging_config()
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_cli_select_async())


if __name__ == "__main__":
    run_cli_select()
