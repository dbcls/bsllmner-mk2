import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel

from bsllmner2.config import RESULT_DIR

OLLAMA_CONTAINER_NAME = "bsllmner-mk2-ollama"
METRICS_OUTPUT_FILE = RESULT_DIR.joinpath("metrics.yaml")


def parse_bytes(size_str: str) -> float:
    """
    Convert strings like '3.62GB' or '42.5GB' to bytes as a float.
    """
    size_str = size_str.strip()
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        'KIB': 1024,
        'MIB': 1024 ** 2,
        'GIB': 1024 ** 3,
        'TIB': 1024 ** 4,
    }
    match = re.match(r'([\d\.]+)\s*([A-Za-z]+)', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    num, unit = match.groups()
    return float(num) * units.get(unit.upper(), 1)


def parse_percentage(perc_str: str) -> float:
    """
    Convert strings like '8.16%' to a float.
    """
    perc_str = perc_str.strip()
    if perc_str.endswith('%'):
        return float(perc_str[:-1])
    raise ValueError(f"Invalid percentage format: {perc_str}")


class DockerStatsResponse(BaseModel):
    """
    $ docker stats bsllmner-mk2-ollama --no-stream --format "{{json .}}"
    {"BlockIO": "3.62GB / 42.5GB", "CPUPerc": "0.00%", "Container": "bsllmner-mk2-ollama", "ID": "8a8678a1dd25", "MemPerc": "8.16%",
        "MemUsage": "41.09GiB / 503.6GiB", "Name": "bsllmner-mk2-ollama", "NetIO": "42.8GB / 88.8MB", "PIDs": "26"}
    """
    BlockIO: str
    CPUPerc: str
    Container: str
    ID: str
    MemPerc: str
    MemUsage: str
    Name: str
    NetIO: str
    PIDs: str


def docker_stats(container_name: str) -> DockerStatsResponse:
    result = subprocess.check_output([
        "docker",
        "stats",
        container_name,
        "--no-stream",
        "--format",
        "{{json .}}"
    ])
    raw = json.loads(result.decode("utf-8"))
    return DockerStatsResponse(**raw)


class NvidiaSmiResponse(BaseModel):
    """
    docker exec bsllmner-mk2-ollama nvidia-smi --query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
    GPU-415f6582-1df0-82e8-67fe-4577cce30c15, NVIDIA RTX 6000 Ada Generation, 37, 49140, 0, 5.24
    GPU-5b0aaa0f-cd30-62ac-a444-d489e55fe266, NVIDIA RTX 6000 Ada Generation, 18, 49140, 0, 8.43
    """
    uuid: str
    name: str
    memory_used_bytes: float  # MiB to bytes
    memory_total_bytes: float  # MiB to bytes
    utilization_gpu: int  # percentage (0-100)
    power_draw: float  # in Watts


def nvidia_smi(container_name: str) -> list[NvidiaSmiResponse]:
    result = subprocess.check_output([
        "docker",
        "exec",
        container_name,
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ])
    raw_lines = result.decode("utf-8").strip().split("\n")
    gpus = []
    for line in raw_lines:
        if not line.strip():
            continue
        uuid, name, mem_used, mem_total, util_gpu, power = [x.strip() for x in line.split(",")]
        gpus.append(NvidiaSmiResponse(
            uuid=uuid,
            name=name,
            memory_used_bytes=parse_bytes(f"{mem_used}MiB"),
            memory_total_bytes=parse_bytes(f"{mem_total}MiB"),
            utilization_gpu=int(util_gpu),
            power_draw=float(power)
        ))

    return gpus


class Metrics(BaseModel):
    timestamp: str  # e.g., "2023-10-01T00:00:00Z"
    block_io_read_bytes: float
    block_io_write_bytes: float
    cpu_percentage: float
    container_name: str
    container_id: str
    memory_percentage: float
    memory_used_bytes: float
    memory_total_bytes: float
    net_io_received_bytes: float
    net_io_sent_bytes: float
    pids: int
    gpus: list[NvidiaSmiResponse]


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
        self.records: List[Metrics] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._collect_loop)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _collect_loop(self) -> None:
        next_time = time.time()
        while not self._stop_event.is_set():
            metrics = collect_metrics(self.container_name)
            self.records.append(metrics)

            next_time += self.interval_sec
            sleep_time = max(0, next_time - time.time())
            time.sleep(sleep_time)

            self.count += 1

    def get_records(self) -> List[Metrics]:
        return self.records


# === CLI ===


class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False) -> None:  # type: ignore
        return super().increase_indent(flow, False)


class Args(BaseModel):
    container_name: str = OLLAMA_CONTAINER_NAME
    interval_sec: int = 5
    count: Optional[int] = None
    output: Literal["yaml", "stdout"] = "stdout"
    output_file: Path


def parse_args(args: List[str]) -> Args:
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
        print(f"Collecting metrics from container '{args.container_name}' every {args.interval_sec} seconds and saving to '{args.output_file}'")
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
                print(yaml.dump(
                    metrics.model_dump(),
                    Dumper=IndentDumper,
                    sort_keys=False,
                    allow_unicode=True,
                ))
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
