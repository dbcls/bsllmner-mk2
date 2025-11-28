import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

HERE = Path(__file__).parent
LOG_DIR = HERE.joinpath("model-evaluation-batch-logs")
SELECT_RESULTS_DIR = Path("/app/bsllmner2-results/select")
BASE_RUN_NAME = "models-with-large-dataset"
MAPPING_FILE = Path("/app/tests/zenodo-data/biosample_cellosaurus_mapping_gold_standard.tsv")
OLLAMA_HOST = "http://bsllmner-mk2-ollama:11434"
RESULTS_TSV_PATH = HERE.joinpath("model_evaluation_results.tsv")

MODELS = [
    "deepseek-r1:8b",
    "deepseek-r1:32b",
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:27b",
    "gpt-oss:20b",
    "llama3.1:8b",
    "phi4:14b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:32b",
]


def parse_time_from_log(model: str) -> Dict[str, float]:
    model_safe = model.replace(":", "_")
    log_file_path = LOG_DIR.joinpath(f"{model_safe}.log")
    log_lines = log_file_path.read_text(encoding="utf-8").splitlines()

    TS_MAIN = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - bsllmner2 - INFO - (?P<msg>.*)$"
    )

    def _parse_ts(line: str) -> Optional[Tuple[datetime, str]]:
        m = TS_MAIN.match(line)
        if m:
            ts_str = m.group("ts")
            msg = m.group("msg")
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            return ts, msg
        return None

    P_EXTRACT = re.compile(r"Extracting entities")
    P_ONTOLOGY_SEARCH = re.compile(r"Selecting entities")
    P_T2T = re.compile(r"text2term for field: cell_line")
    P_SELECT = re.compile(r"Performing LLM selection")
    P_COMPLETE = re.compile(r"Processing complete")

    events = [_parse_ts(line) for line in log_lines]
    events = [e for e in events if e]

    ts_extract = ts_ontology_search = ts_t2t = ts_select = ts_complete = None

    for ts, msg in events:  # type: ignore
        if ts_extract is None and P_EXTRACT.search(msg):
            ts_extract = ts
        elif ts_ontology_search is None and P_ONTOLOGY_SEARCH.search(msg):
            ts_ontology_search = ts
        elif ts_t2t is None and P_T2T.search(msg):
            ts_t2t = ts
        elif ts_select is None and P_SELECT.search(msg):
            ts_select = ts
        elif ts_complete is None and P_COMPLETE.search(msg):
            ts_complete = ts

    result = {}

    if ts_extract and ts_ontology_search:
        result["extract_sec"] = (ts_ontology_search - ts_extract).total_seconds()
    if ts_ontology_search and ts_t2t:
        result["ontology_search_sec"] = (ts_t2t - ts_ontology_search).total_seconds()
    if ts_t2t and ts_select:
        result["text2term_sec"] = (ts_select - ts_t2t).total_seconds()
    if ts_select and ts_complete:
        result["selection_sec"] = (ts_complete - ts_select).total_seconds()
    if ts_extract and ts_complete:
        result["total_sec"] = (ts_complete - ts_extract).total_seconds()

    return result


def load_answer_mapping() -> Dict[str, Optional[str]]:
    with MAPPING_FILE.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    answer_mapping = {}
    for line in lines[1:]:
        if line.strip() == "":
            continue
        elements = line.split("\t")
        accession = elements[0]
        answer_id = elements[3]
        answer_mapping[accession] = answer_id if answer_id != "" else None

    return answer_mapping


def load_predicted_mapping(model: str) -> Dict[str, Optional[str]]:
    model_safe = model.replace(":", "_")
    run_name = f"{BASE_RUN_NAME}-{model_safe}"
    select_results_path = SELECT_RESULTS_DIR.joinpath(f"select_{run_name}.json")
    select_results = json.loads(select_results_path.read_text(encoding="utf-8"))
    predicted_mapping: Dict[str, Optional[str]] = {}
    for select_result in select_results:
        accession = select_result["accession"]
        results = select_result.get("results", {}) or {}
        cell_line_info = results.get("cell_line", None)
        if cell_line_info is None:
            predicted_mapping[accession] = None
        else:
            predicted_mapping[accession] = cell_line_info.get("term_id", None)

    return predicted_mapping


def evaluate_mapping(
    predicted_mapping: Dict[str, Optional[str]],
    answer_mapping: Dict[str, Optional[str]],
) -> Dict[str, Optional[float]]:
    """\
    Evaluation rules:

    answer = "A"  / pred = "A"        -> correct
    answer = "A"  / pred = "B"        -> incorrect
    answer = "A"  / pred = None       -> incorrect
    answer = None / pred = None       -> correct
    answer = None / pred = "something"-> incorrect

    None is simply one of the valid possible classes.
    Exact match is used for accuracy.

    Precision: among predicted strings, how many were correct?
        TP / (TP + FP)

    Recall: among answer strings, how many were correctly predicted?
        TP / (TP + FN)

    F1-score: harmonic mean of precision and recall.
        F1 = 2 * P * R / (P + R)
    """

    tp = 0  # correct string predictions
    fp = 0  # predicted string but incorrect
    fn = 0  # answer is string but model failed to predict it

    correct = 0
    total = len(answer_mapping)

    for key, answer in answer_mapping.items():
        pred = predicted_mapping.get(key, None)

        # exact match for accuracy
        if pred == answer:
            correct += 1

        # precision / recall counting
        if answer is not None:  # answer is a string
            if pred == answer:
                tp += 1
            else:
                fn += 1
        else:  # answer is None
            if pred is not None:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    accuracy = correct / total

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "correct": correct,
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def count_results(model: str) -> Dict[str, int]:
    model_safe = model.replace(":", "_")
    run_name = f"{BASE_RUN_NAME}-{model_safe}"
    select_results_path = SELECT_RESULTS_DIR.joinpath(f"select_{run_name}.json")
    select_results = json.loads(select_results_path.read_text(encoding="utf-8"))
    extract_count = 0
    select_count = 0
    final_count = 0
    for select_result in select_results:
        extract_output = select_result.get("extract_output", {}) or {}
        extract_count += sum(v is not None for v in extract_output.values())
        llm_output = select_result.get("llm_chat_response", {}) or {}
        select_count += sum(v is not None for v in llm_output.values())
        final_output = select_result.get("results", {}) or {}
        final_count += sum(v is not None for v in final_output.values())

    return {
        "extract_count": extract_count,
        "select_count": select_count,
        "final_count": final_count,
    }


def get_ollama_model_info(model: str) -> Dict[str, Any]:
    url = f"{OLLAMA_HOST}/api/show"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model}

    res = httpx.post(url, json=payload, headers=headers)
    res.raise_for_status()
    data = res.json()
    details = data.get("details", {})
    family = details.get("family", "unknown")
    model_info = data.get("model_info", {})

    return {
        "parameter_size": details.get("parameter_size"),
        "context_length": model_info.get(f"{family}.context_length"),
        "embedding_length": model_info.get(f"{family}.embedding_length"),
    }


def write_results_tsv(results: Dict[str, Dict[str, Any]]) -> None:
    header = [
        "Model",
        "Parameter Size",
        "Context Length",
        "Embedding Length",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-score",
        "Extract LLM Time (sec)",
        "Selection LLM Time (sec)",
        "Total Time (sec)",
        "Extracted Fields (max: 3000)",
        "Selected Fields (max: 3000)",
        "Final Fields (max: 3000)",
    ]

    with open(RESULTS_TSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        for model, data in results.items():
            extract_time = data.get("extract_sec", 0) or 0
            selection_time = data.get("selection_sec", 0) or 0
            total_time = extract_time + selection_time

            acc = data.get("accuracy")
            precision = data.get("precision")
            recall = data.get("recall")
            f1 = data.get("f1")

            acc_pct = round(acc * 100, 2) if acc is not None else ""
            precision_pct = round(precision * 100, 2) if precision is not None else ""
            recall_pct = round(recall * 100, 2) if recall is not None else ""

            f1_val = round(f1, 4) if f1 is not None else ""

            row = [
                model,
                data.get("parameter_size", ""),
                data.get("context_length", ""),
                data.get("embedding_length", ""),
                acc_pct,
                precision_pct,
                recall_pct,
                f1_val,
                extract_time,
                selection_time,
                total_time,
                data.get("extract_count", ""),
                data.get("select_count", ""),
                data.get("final_count", ""),
            ]

            writer.writerow(row)


def main() -> None:
    answer_mapping = load_answer_mapping()

    results = {}
    for model in MODELS:
        result: Dict[str, Any] = {}

        time_results = parse_time_from_log(model)
        result.update(time_results)

        predicted_mapping = load_predicted_mapping(model)
        evaluation_results = evaluate_mapping(predicted_mapping, answer_mapping)
        result.update(evaluation_results)

        results_counts = count_results(model)
        result.update(results_counts)

        ollama_info = get_ollama_model_info(model)
        result.update(ollama_info)

        results[model] = result

    write_results_tsv(results)
    print(f"Results written to {RESULTS_TSV_PATH}")


if __name__ == "__main__":
    main()
