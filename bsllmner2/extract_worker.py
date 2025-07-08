import sys
from pathlib import Path

from bsllmner2.client.ollama import ner
from bsllmner2.metrics import LiveMetricsCollector
from bsllmner2.schema import Result
from bsllmner2.utils import (build_error_log, compute_processing_time,
                             dump_result, evaluate_output, get_now_str)


def main() -> None:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected exactly one argument: the path to the queue file.")

        queue_file_path = Path(sys.argv[1])
        if not queue_file_path.exists():
            raise FileNotFoundError(f"Queue file {queue_file_path} does not exist.")

        with queue_file_path.open("r", encoding="utf-8") as f:
            queue_obj = Result.model_validate_json(f.read())

        config = queue_obj.input.config
        bs_entries = queue_obj.input.bs_entries
        mapping = queue_obj.input.mapping
        model = queue_obj.input.model
        prompt = queue_obj.input.prompt

        metrics_collector = LiveMetricsCollector()
        metrics_collector.start()
        try:
            output = ner(config, bs_entries, prompt, model)
            end_time = get_now_str()
        finally:
            metrics_collector.stop()
        metrics = metrics_collector.get_records()

        evaluation = evaluate_output(output, mapping)

        # Update result with the new data
        queue_obj.output = output
        queue_obj.evaluation = evaluation
        queue_obj.metrics = metrics
        queue_obj.run_metadata.end_time = end_time
        queue_obj.run_metadata.status = "completed"
        queue_obj.run_metadata.processing_time = compute_processing_time(
            queue_obj.run_metadata.start_time, end_time
        )
        queue_obj.run_metadata.matched_entries = sum(
            1 for eval_item in evaluation if eval_item.match
        )
        queue_obj.run_metadata.total_entries = len(bs_entries)
        if queue_obj.run_metadata.total_entries > 0:
            queue_obj.run_metadata.accuracy = (
                queue_obj.run_metadata.matched_entries
                / queue_obj.run_metadata.total_entries
            ) * 100

        dump_result(queue_obj, queue_obj.run_metadata.run_name)
    except Exception as e:  # pylint: disable=broad-except
        if "queue_obj" in locals():
            queue_obj.run_metadata.status = "failed"
            queue_obj.error_log = build_error_log(e)
            dump_result(queue_obj, queue_obj.run_metadata.run_name)
        sys.exit(1)


if __name__ == "__main__":
    main()
