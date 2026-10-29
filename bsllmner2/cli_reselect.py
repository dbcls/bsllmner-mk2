import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from ollama import ChatResponse

from bsllmner2.benchmark import (
    aggregate_llm_timings,
    log_performance_summary,
)
from bsllmner2.cli_common import (
    BatchInfo,
    add_common_arguments,
    build_config,
    build_run_metadata,
    generate_run_name,
    process_batches,
    run_with_lifecycle,
    validate_common_args,
)
from bsllmner2.config import LOGGER, Config, set_logging_config, set_logging_level
from bsllmner2.io import dump_select_result, load_bs_entries, load_select_config, load_select_result
from bsllmner2.llm import OllamaBackend
from bsllmner2.models import (
    CliReselectArgs,
    ErrorLog,
    PerformanceSummary,
    ReselectInfo,
    SelectEntry,
    SelectResult,
    StageTimings,
)
from bsllmner2.reselect import (
    FieldValues,
    fields_with_ontology,
    reselect_entries,
    restrict_select_config,
    sha256_of_file,
    unresolved_values,
)
from bsllmner2.select import (
    TEXT2TERM_CACHE_DIR,
    SearchMemo,
    SelectStageTimings,
    build_index_map,
    build_text2term_cache,
)


def parse_args(args: list[str]) -> tuple[Config, CliReselectArgs]:
    """Parse command-line arguments for the bsllmner2 CLI reselect mode."""
    parser = argparse.ArgumentParser(
        description="Repeat ontology search and LLM selection for the values an earlier select run left unresolved.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--from-result",
        type=Path,
        required=True,
        help="Path to the SelectResult JSON of the source run.",
    )
    parser.add_argument(
        "--select-config",
        type=Path,
        required=True,
        help="Path to the select configuration file in JSON format.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Fields whose unresolved values are reselected (default: every field with an ontology_file).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_false",
        dest="include_reasoning",
        help="Disable reasoning step during selection.",
    )

    parsed_args = parser.parse_args(args)
    validate_common_args(parser, parsed_args)
    if parsed_args.resume:
        parser.error("bsllmner2_reselect does not support --resume.")
    if not parsed_args.from_result.exists():
        parser.error(f"Result file {parsed_args.from_result} does not exist.")
    if not parsed_args.select_config.exists():
        parser.error(f"Select configuration file {parsed_args.select_config} does not exist.")

    config = build_config(parsed_args)

    return config, CliReselectArgs(
        bs_entries=parsed_args.bs_entries.resolve(),
        model=parsed_args.model,
        thinking=parsed_args.thinking,
        max_entries=parsed_args.max_entries,
        run_name=parsed_args.run_name,
        num_ctx=parsed_args.num_ctx,
        resume=False,
        batch_size=parsed_args.batch_size,
        from_result=parsed_args.from_result.resolve(),
        select_config=parsed_args.select_config.resolve(),
        fields=parsed_args.fields,
        include_reasoning=parsed_args.include_reasoning,
    )


def collect_targets(entries: list[SelectEntry], fields: list[str]) -> dict[str, FieldValues]:
    """Target values per accession, for the entries that have any."""
    targets: dict[str, FieldValues] = {}
    for entry in entries:
        pending = unresolved_values(entry, fields)
        if pending:
            targets[entry.extract.accession] = pending
    return targets


async def run_cli_reselect_async() -> None:
    """Run the CLI for bsllmner2 reselect mode."""
    LOGGER.info("Starting bsllmner2 CLI reselect mode...")
    config, args = parse_args(sys.argv[1:])
    set_logging_level(config.debug)
    LOGGER.info("Config:\n%s", config.model_dump_json(indent=2))
    LOGGER.info("Args:\n%s", args.model_dump_json(indent=2))

    backend = OllamaBackend(config.ollama_host)

    full_select_config = load_select_config(args.select_config)
    fields = fields_with_ontology(full_select_config, args.fields)
    select_config = restrict_select_config(full_select_config, fields)
    LOGGER.info("Reselecting fields: %s", fields)

    source = load_select_result(args.from_result)
    if source.run_metadata.status != "completed":
        LOGGER.warning("Source run %s has status %s.", source.run_metadata.run_name, source.run_metadata.status)
    reselect_info = ReselectInfo(
        source_result_file=args.from_result.name,
        source_sha256=sha256_of_file(args.from_result),
        source_run_name=source.run_metadata.run_name,
        fields=fields,
    )

    targets_by_accession = collect_targets(source.entries, fields)
    target_entries = [e for e in source.entries if e.extract.accession in targets_by_accession]
    if args.max_entries is not None:
        target_entries = target_entries[: args.max_entries]
    LOGGER.info(
        "%d of %d entries have unresolved values in the selected fields (%d values).",
        len(target_entries),
        len(source.entries),
        sum(len(v) for e in target_entries for v in targets_by_accession[e.extract.accession].values()),
    )

    bs_entries = load_bs_entries(args.bs_entries)
    known = {b.get("accession") for b in bs_entries}
    missing = sorted(e.extract.accession for e in target_entries if e.extract.accession not in known)
    if missing:
        raise ValueError(
            f"{len(missing)} entries to reselect are not in {args.bs_entries}: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )

    run_name, start_time = generate_run_name(args.model, args.run_name)

    select_index_map, disk_io_timings = build_index_map(select_config)
    text2term_disk_io = build_text2term_cache(select_config)
    disk_io_timings.text2term_cache_build_sec.extend(text2term_disk_io.text2term_cache_build_sec)
    disk_io_timings.text2term_cache_load_sec.extend(text2term_disk_io.text2term_cache_load_sec)

    wall_start = time.perf_counter()
    search_memo: SearchMemo = {}
    t2t_memo: SearchMemo = {}

    reselected: list[SelectEntry] = []
    all_select_chat_responses: list[ChatResponse] = []
    stage_timings_list: list[StageTimings] = []

    try:
        async with run_with_lifecycle() as run_state:

            async def process_reselect_batch(
                batch_info: BatchInfo,
            ) -> tuple[list[SelectEntry], list[ChatResponse], SelectStageTimings, list[ErrorLog]]:
                batch_entries: list[SelectEntry] = [item["entry"] for item in batch_info.entries]
                LOGGER.info("Reselecting %d entries...", len(batch_entries))
                return await reselect_entries(
                    backend,
                    bs_entries,
                    args.model,
                    batch_entries,
                    targets_by_accession,
                    select_config,
                    args.thinking,
                    include_reasoning=args.include_reasoning,
                    index_map=select_index_map,
                    text2term_cache_folder=TEXT2TERM_CACHE_DIR,
                    num_ctx=args.num_ctx,
                    search_memo=search_memo,
                    t2t_memo=t2t_memo,
                )

            def on_batch_complete(
                batch_idx: int,
                batch_result: tuple[list[SelectEntry], list[ChatResponse], SelectStageTimings, list[ErrorLog]],
            ) -> None:
                batch_entries, batch_responses, select_timings, batch_errors = batch_result
                run_state.errors.extend(batch_errors)
                run_state.error_count += len(batch_errors)
                reselected.extend(batch_entries)
                all_select_chat_responses.extend(batch_responses)
                stage_timings_list.append(
                    StageTimings(
                        batch_idx=batch_idx,
                        batch_size=len(batch_entries),
                        ner_sec=0.0,
                        ontology_search_sec=select_timings["ontology_search_sec"],
                        text2term_sec=select_timings["text2term_sec"],
                        llm_select_sec=select_timings["llm_select_sec"],
                        resume_write_sec=0.0,
                    ),
                )

            items: list[dict[str, Any]] = [{"accession": e.extract.accession, "entry": e} for e in target_entries]
            await process_batches(
                entries=items,
                batch_size=args.batch_size,
                process_fn=process_reselect_batch,
                on_batch_complete=on_batch_complete,
                log_prefix="Reselecting",
            )
    finally:
        total_wall_sec = time.perf_counter() - wall_start
        status = run_state.status
        end_time = run_state.end_time

        run_metadata = build_run_metadata(run_name, args.model, args.thinking, start_time, end_time, status)
        run_metadata = run_metadata.model_copy(
            update={
                "total_entries": len(reselected),
                "processing_time_sec": (end_time - start_time).total_seconds() if end_time is not None else None,
                "reselect": reselect_info,
            },
        )

        select_timing = aggregate_llm_timings(all_select_chat_responses)
        performance = PerformanceSummary(
            total_input_entries=len(target_entries),
            completed_count=len(reselected),
            total_wall_sec=total_wall_sec,
            stage_timings=stage_timings_list,
            ner_llm_timing=None,
            select_llm_timing=select_timing if select_timing.call_count > 0 else None,
            disk_io=disk_io_timings,
        )

        select_result = SelectResult(
            entries=reselected,
            run_metadata=run_metadata,
            evaluation=None,
            performance=performance,
            errors=run_state.errors,
        )
        select_result_file = dump_select_result(select_result, run_name)
        LOGGER.info("Processing %s. Result saved to %s", status, select_result_file)
        log_performance_summary(performance, None)

    if status != "completed" or run_state.error_count > 0:
        sys.exit(1)


def run_cli_reselect() -> None:
    """Run the CLI for bsllmner2 reselect mode in an event loop."""
    set_logging_config()
    asyncio.run(run_cli_reselect_async())


if __name__ == "__main__":
    run_cli_reselect()
