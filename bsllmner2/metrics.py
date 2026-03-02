import argparse
import json
import logging
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from bsllmner2.config import OLLAMA_CONTAINER_NAME, RESULT_DIR
from bsllmner2.models import DockerStatsResponse, Metrics, NvidiaSmiResponse

LOGGER = logging.getLogger("bsllmner2")

METRICS_OUTPUT_FILE = RESULT_DIR.joinpath("metrics.yaml")


def parse_bytes(size_str: str) -> float:
    """Convert strings like '3.62GB' or '42.5GB' to bytes as a float."""
    size_str = size_str.strip()
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    match = re.match(r"([\d\.]+)\s*([A-Za-z]+)", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    num, unit = match.groups()
    unit_upper = unit.upper()
    if unit_upper not in units:
        raise ValueError(f"Unknown size unit: {unit}")
    try:
        value = float(num)
    except ValueError as e:
        raise ValueError(f"Invalid size format: {size_str}") from e

    return value * units[unit_upper]


def parse_percentage(perc_str: str) -> float:
    """Convert strings like '8.16%' to a float."""
    perc_str = perc_str.strip()
    if perc_str.endswith("%"):
        num_part = perc_str[:-1]
        try:
            value = float(num_part)
        except ValueError:
            pass
        else:
            if math.isfinite(value):
                return value
    raise ValueError(f"Invalid percentage format: {perc_str}")


@lru_cache(maxsize=1)
def check_ollama_container_exists(container_name: str = OLLAMA_CONTAINER_NAME) -> bool:
    try:
        result = subprocess.check_output(["docker", "ps", "-q", "-f", f"name={container_name}"])
        return bool(result.strip())
    except subprocess.CalledProcessError:
        return False


def docker_stats(container_name: str) -> DockerStatsResponse:
    result = subprocess.check_output(["docker", "stats", container_name, "--no-stream", "--format", "{{json .}}"])
    raw = json.loads(result.decode("utf-8"))
    return DockerStatsResponse(**raw)


def nvidia_smi(container_name: str) -> list[NvidiaSmiResponse]:
    result = subprocess.check_output(
        [
            "docker",
            "exec",
            container_name,
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
    )
    raw_lines = result.decode("utf-8").strip().split("\n")
    gpus = []
    for line in raw_lines:
        if not line.strip():
            continue
        uuid, name, mem_used, mem_total, util_gpu, power = [x.strip() for x in line.split(",")]
        gpus.append(
            NvidiaSmiResponse(
                uuid=uuid,
                name=name,
                memory_used_bytes=parse_bytes(f"{mem_used}MiB"),
                memory_total_bytes=parse_bytes(f"{mem_total}MiB"),
                utilization_gpu=int(util_gpu),
                power_draw=float(power),
            ),
        )

    return gpus


def collect_metrics(container_name: str) -> Metrics:
    docker_stats_response = docker_stats(container_name)

    block_io_read, block_io_write = map(parse_bytes, docker_stats_response.BlockIO.split(" / "))
    cpu_percentage = parse_percentage(docker_stats_response.CPUPerc)
    container_name = docker_stats_response.Container
    container_id = docker_stats_response.ID
    memory_percentage = parse_percentage(docker_stats_response.MemPerc)
    memory_used, memory_total = map(parse_bytes, docker_stats_response.MemUsage.split(" / "))
    net_io_received, net_io_sent = map(parse_bytes, docker_stats_response.NetIO.split(" / "))
    pids = int(docker_stats_response.PIDs)

    gpus = nvidia_smi(container_name)

    return Metrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        block_io_read_bytes=block_io_read,
        block_io_write_bytes=block_io_write,
        cpu_percentage=cpu_percentage,
        container_name=container_name,
        container_id=container_id,
        memory_percentage=memory_percentage,
        memory_used_bytes=memory_used,
        memory_total_bytes=memory_total,
        net_io_received_bytes=net_io_received,
        net_io_sent_bytes=net_io_sent,
        pids=pids,
        gpus=gpus,
    )


class LiveMetricsCollector:
    def __init__(self, container_name: str = OLLAMA_CONTAINER_NAME, interval_sec: int = 5):
        self.container_name = container_name
        self.interval_sec = interval_sec
        self.count = 0
        self.records: list[Metrics] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._collect_loop)
        self._enabled = check_ollama_container_exists(container_name)

    def start(self) -> None:
        if self._enabled:
            self._thread.start()

    def stop(self) -> None:
        if self._enabled:
            self._stop_event.set()
            self._thread.join()

    def _collect_loop(self) -> None:
        next_time = time.time()
        while not self._stop_event.is_set():
            try:
                metrics = collect_metrics(self.container_name)
            except (subprocess.CalledProcessError, OSError) as e:
                LOGGER.warning("Metrics collection failed: %s", e)
            else:
                with self._lock:
                    self.records.append(metrics)
                    self.count += 1

            next_time += self.interval_sec
            sleep_time = max(0, next_time - time.time())
            time.sleep(sleep_time)

    def get_records(self) -> list[Metrics]:
        if not self._enabled:
            return []
        with self._lock:
            return list(self.records)


# === CLI ===


class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, _indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


class Args(BaseModel):
    container_name: str = OLLAMA_CONTAINER_NAME
    interval_sec: int = 5
    count: int | None = None
    output: Literal["yaml", "stdout"] = "stdout"
    output_file: Path


def parse_args(args: list[str]) -> Args:
    parser = argparse.ArgumentParser(
        description="Collect various metrics from a Docker container running the ollama service.",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default=OLLAMA_CONTAINER_NAME,
        help="Name of the Docker container to collect metrics from.",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=5,
        help="Interval in seconds between metric collections.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of times to collect metrics. If not specified, will run indefinitely.",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["yaml", "stdout"],
        default="stdout",
        help="Output format for the collected metrics.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=METRICS_OUTPUT_FILE,
        help="File to write the collected metrics to. Only used if output is 'yaml'.",
    )

    parsed_args = parser.parse_args(args)

    return Args(
        container_name=parsed_args.container_name,
        interval_sec=parsed_args.interval_sec,
        count=parsed_args.count,
        output=parsed_args.output,
        output_file=parsed_args.output_file,
    )


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.output == "yaml":
        print(
            f"Collecting metrics from container '{args.container_name}' every {args.interval_sec} seconds and saving to '{args.output_file}'",
        )
        f = args.output_file.open("w", encoding="utf-8")
    else:
        f = None

    try:
        i = 0
        next_time = time.time()
        while args.count is None or i < args.count:
            metrics = collect_metrics(args.container_name)
            if args.output == "yaml":
                yaml.dump(
                    [metrics.model_dump()],
                    stream=f,
                    Dumper=IndentDumper,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
            elif args.output == "stdout":
                print(
                    yaml.dump(
                        metrics.model_dump(),
                        Dumper=IndentDumper,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                )
                print("-" * 40)

            next_time += args.interval_sec
            sleep_time = max(0, next_time - time.time())

            i += 1
            time.sleep(sleep_time)
    finally:
        if f:
            f.close()


if __name__ == "__main__":
    main()
