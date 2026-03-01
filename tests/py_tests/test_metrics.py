"""Tests for bsllmner2.metrics: parse helpers and subprocess-based functions."""

import json
import math
import subprocess
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.metrics import (
    DockerStatsResponse,
    check_ollama_container_exists,
    docker_stats,
    nvidia_smi,
    parse_bytes,
    parse_percentage,
)


class TestParseBytes:
    """Normal cases for parse_bytes."""

    def test_gigabytes(self) -> None:
        """GB uses binary (1024^3), not SI (10^9)."""
        assert parse_bytes("3.62GB") == pytest.approx(3.62 * 1024**3)

    def test_gibibytes_same_as_gb(self) -> None:
        """GiB and GB return the same value."""
        assert parse_bytes("1GiB") == parse_bytes("1GB")

    def test_megabytes(self) -> None:
        assert parse_bytes("88.8MB") == pytest.approx(88.8 * 1024**2)

    def test_mebibytes(self) -> None:
        """MiB used by nvidia-smi output."""
        assert parse_bytes("49140MiB") == pytest.approx(49140.0 * 1024**2)

    def test_kilobytes(self) -> None:
        assert parse_bytes("512KB") == pytest.approx(512.0 * 1024)

    def test_kibibytes(self) -> None:
        assert parse_bytes("512KiB") == pytest.approx(512.0 * 1024)

    def test_terabytes(self) -> None:
        assert parse_bytes("2TB") == pytest.approx(2.0 * 1024**4)

    def test_bytes_only(self) -> None:
        assert parse_bytes("1024B") == pytest.approx(1024.0)

    def test_zero_bytes(self) -> None:
        assert parse_bytes("0B") == 0.0

    def test_case_insensitive(self) -> None:
        assert parse_bytes("1gb") == parse_bytes("1GB")

    def test_whitespace_stripped(self) -> None:
        assert parse_bytes("  3.62GB  ") == parse_bytes("3.62GB")

    def test_space_between_number_and_unit(self) -> None:
        assert parse_bytes("3.62 GB") == parse_bytes("3.62GB")

    def test_docker_blockio_format(self) -> None:
        """Real docker stats BlockIO after split(' / ')."""
        assert parse_bytes("3.62GB") == pytest.approx(3.62 * 1024**3)
        assert parse_bytes("42.5GB") == pytest.approx(42.5 * 1024**3)

    def test_docker_memusage_format(self) -> None:
        """Real docker stats MemUsage after split(' / ')."""
        assert parse_bytes("41.09GiB") == pytest.approx(41.09 * 1024**3)
        assert parse_bytes("503.6GiB") == pytest.approx(503.6 * 1024**3)

    def test_nvidia_smi_mib_format(self) -> None:
        """Value constructed by nvidia_smi() as f'{mem_used}MiB'."""
        assert parse_bytes("49140MiB") == pytest.approx(49140.0 * 1024**2)


class TestParseBytesErrors:
    """Error cases for parse_bytes."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bytes("")

    def test_pure_text_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bytes("invalid")

    def test_number_only_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bytes("1024")

    def test_unit_only_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bytes("GB")

    def test_negative_number_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_bytes("-5GB")

    def test_unknown_unit_raises(self) -> None:
        """Bug #2: unknown unit must raise, not silently fallback to 1."""
        with pytest.raises(ValueError, match="Unknown size unit"):
            parse_bytes("42XB")

    def test_multiple_dots_raises_with_clear_message(self) -> None:
        """Bug #3: '1.2.3GB' must give a clear error, not internal ValueError."""
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_bytes("1.2.3GB")

    def test_dot_only_with_unit_raises_with_clear_message(self) -> None:
        """Bug #4: '.B' must give a clear error, not internal ValueError."""
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_bytes(".B")


class TestParseBytesProperty:
    """Property-based tests for parse_bytes."""

    @given(
        num=st.floats(min_value=0, max_value=1e12, allow_nan=False, allow_infinity=False),
        unit=st.sampled_from(["B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"]),
    )
    @settings(max_examples=200)
    def test_valid_input_returns_nonneg_float(self, num: float, unit: str) -> None:
        result = parse_bytes(f"{num:f}{unit}")
        assert isinstance(result, float)
        assert result >= 0

    @given(unit=st.sampled_from(["B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"]))
    @settings(max_examples=200)
    def test_zero_coefficient_returns_zero(self, unit: str) -> None:
        assert parse_bytes(f"0{unit}") == 0.0

    @given(num=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_gb_equals_gib(self, num: float) -> None:
        assert parse_bytes(f"{num}GB") == parse_bytes(f"{num}GiB")

    @given(num=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_unit_ordering(self, num: float) -> None:
        b = parse_bytes(f"{num}B")
        kb = parse_bytes(f"{num}KB")
        mb = parse_bytes(f"{num}MB")
        gb = parse_bytes(f"{num}GB")
        tb = parse_bytes(f"{num}TB")
        assert b < kb < mb < gb < tb

    @given(
        num=st.floats(min_value=0, max_value=1e12, allow_nan=False, allow_infinity=False),
        unit=st.sampled_from(["B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"]),
    )
    @settings(max_examples=200)
    def test_result_is_always_finite(self, num: float, unit: str) -> None:
        """parse_bytes result is always finite for valid input."""
        result = parse_bytes(f"{num:f}{unit}")
        assert math.isfinite(result)


class TestParsePercentage:
    """Normal cases for parse_percentage."""

    def test_normal(self) -> None:
        assert parse_percentage("8.16%") == pytest.approx(8.16)

    def test_zero(self) -> None:
        assert parse_percentage("0%") == 0.0

    def test_hundred(self) -> None:
        assert parse_percentage("100%") == 100.0

    def test_docker_cpuperc_idle(self) -> None:
        """Real docker stats CPUPerc for idle container."""
        assert parse_percentage("0.00%") == 0.0

    def test_multicore_over_hundred(self) -> None:
        """Multi-core CPU can exceed 100%."""
        assert parse_percentage("800.5%") == pytest.approx(800.5)

    def test_whitespace_stripped(self) -> None:
        assert parse_percentage("  8.16%  ") == parse_percentage("8.16%")


class TestParsePercentageErrors:
    """Error cases for parse_percentage."""

    def test_no_percent_sign_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_percentage("42")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_percentage("")

    def test_only_percent_sign_raises_with_clear_message(self) -> None:
        """Bug #7: bare '%' must give a clear error."""
        with pytest.raises(ValueError, match="Invalid percentage format"):
            parse_percentage("%")

    def test_double_percent_raises_with_clear_message(self) -> None:
        """Bug #8: '42%%' must give a clear error."""
        with pytest.raises(ValueError, match="Invalid percentage format"):
            parse_percentage("42%%")


class TestParsePercentageProperty:
    """Property-based tests for parse_percentage."""

    @given(num=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_valid_percentage_roundtrip(self, num: float) -> None:
        result = parse_percentage(f"{num}%")
        assert result == pytest.approx(num)

    @given(s=st.text(min_size=1).filter(lambda x: not x.strip().endswith("%")))
    @settings(max_examples=200)
    def test_no_percent_suffix_raises(self, s: str) -> None:
        with pytest.raises(ValueError):
            parse_percentage(s)

    @pytest.mark.parametrize("bad_input", ["nan%", "NaN%", "inf%", "Inf%", "INF%", "-inf%", "-Inf%"])
    def test_rejects_nan_and_inf(self, bad_input: str) -> None:
        """NaN and infinity strings must be rejected as invalid percentages."""
        with pytest.raises(ValueError, match="Invalid percentage format"):
            parse_percentage(bad_input)

    @given(num=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_result_is_always_finite(self, num: float) -> None:
        """parse_percentage result is always finite for valid input."""
        result = parse_percentage(f"{num}%")
        assert math.isfinite(result)


# === subprocess-mock tests ===


class TestCheckOllamaContainerExists:
    """Tests for check_ollama_container_exists with subprocess mocked."""

    def setup_method(self) -> None:
        check_ollama_container_exists.cache_clear()

    def teardown_method(self) -> None:
        check_ollama_container_exists.cache_clear()

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_container_found(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = b"abc123\n"
        assert check_ollama_container_exists("test-container") is True

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_container_not_found_empty(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = b""
        assert check_ollama_container_exists("test-container") is False

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_container_not_found_whitespace(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = b"  \n  "
        assert check_ollama_container_exists("test-container") is False

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_docker_command_fails(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "docker")
        assert check_ollama_container_exists("test-container") is False


class TestDockerStats:
    """Tests for docker_stats with subprocess mocked."""

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_valid_output(self, mock_check_output: pytest.fixture) -> None:
        raw = {
            "BlockIO": "3.62GB / 42.5GB",
            "CPUPerc": "8.16%",
            "Container": "test-ollama",
            "ID": "abc123",
            "MemPerc": "10.5%",
            "MemUsage": "41.09GiB / 503.6GiB",
            "Name": "test-ollama",
            "NetIO": "42.8GB / 88.8MB",
            "PIDs": "26",
        }
        mock_check_output.return_value = json.dumps(raw).encode("utf-8")
        result = docker_stats("test-ollama")
        assert isinstance(result, DockerStatsResponse)
        assert result.Container == "test-ollama"
        assert result.ID == "abc123"
        assert result.PIDs == "26"

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_all_fields_mapped(self, mock_check_output: pytest.fixture) -> None:
        raw = {
            "BlockIO": "1GB / 2GB",
            "CPUPerc": "50.0%",
            "Container": "my-container",
            "ID": "deadbeef",
            "MemPerc": "25.0%",
            "MemUsage": "8GiB / 32GiB",
            "Name": "my-container",
            "NetIO": "100MB / 200MB",
            "PIDs": "42",
        }
        mock_check_output.return_value = json.dumps(raw).encode("utf-8")
        result = docker_stats("my-container")
        assert result.BlockIO == "1GB / 2GB"
        assert result.CPUPerc == "50.0%"
        assert result.MemPerc == "25.0%"
        assert result.MemUsage == "8GiB / 32GiB"
        assert result.NetIO == "100MB / 200MB"
        assert result.Name == "my-container"


class TestNvidiaSmi:
    """Tests for nvidia_smi with subprocess mocked."""

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_single_gpu(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = (
            b"GPU-abc123, NVIDIA RTX 6000, 37, 49140, 0, 5.24\n"
        )
        gpus = nvidia_smi("test-container")
        assert len(gpus) == 1
        assert gpus[0].uuid == "GPU-abc123"
        assert gpus[0].name == "NVIDIA RTX 6000"
        assert gpus[0].utilization_gpu == 0
        assert gpus[0].power_draw == pytest.approx(5.24)

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_multiple_gpus(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = (
            b"GPU-aaa, RTX 6000, 37, 49140, 0, 5.24\n"
            b"GPU-bbb, RTX 6000, 18, 49140, 50, 8.43\n"
        )
        gpus = nvidia_smi("test-container")
        assert len(gpus) == 2
        assert gpus[0].uuid == "GPU-aaa"
        assert gpus[1].uuid == "GPU-bbb"
        assert gpus[1].utilization_gpu == 50

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_empty_lines_skipped(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = (
            b"\n"
            b"GPU-aaa, RTX 6000, 37, 49140, 0, 5.24\n"
            b"\n"
            b"GPU-bbb, RTX 6000, 18, 49140, 50, 8.43\n"
            b"\n"
        )
        gpus = nvidia_smi("test-container")
        assert len(gpus) == 2

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_trailing_newline(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = (
            b"GPU-abc, RTX 6000, 37, 49140, 0, 5.24\n"
        )
        gpus = nvidia_smi("test-container")
        assert len(gpus) == 1

    @patch("bsllmner2.metrics.subprocess.check_output")
    def test_memory_mib_to_bytes(self, mock_check_output: pytest.fixture) -> None:
        mock_check_output.return_value = (
            b"GPU-abc, RTX 6000, 37, 49140, 0, 5.24\n"
        )
        gpus = nvidia_smi("test-container")
        assert gpus[0].memory_used_bytes == pytest.approx(37 * 1024**2)
        assert gpus[0].memory_total_bytes == pytest.approx(49140 * 1024**2)


# === Exact multiplier constants ===


class TestParseBytesExactMultipliers:
    """Verify exact multiplier constants kill mutations like 1024 → 1000.

    The existing tests check approximate values but don't pin down the exact
    ratio between adjacent units. These tests ensure that mutating any
    multiplier constant in the `units` dict is immediately detected.
    """

    def test_1b_is_exactly_1(self) -> None:
        """1 B = exactly 1.0 byte."""
        assert parse_bytes("1B") == 1.0

    def test_1kb_is_exactly_1024(self) -> None:
        """1 KB = exactly 1024 bytes, not 1000."""
        assert parse_bytes("1KB") == 1024.0

    def test_1mb_is_exactly_1024_squared(self) -> None:
        """1 MB = exactly 1024^2 = 1_048_576 bytes."""
        assert parse_bytes("1MB") == 1024.0**2

    def test_1gb_is_exactly_1024_cubed(self) -> None:
        """1 GB = exactly 1024^3 bytes."""
        assert parse_bytes("1GB") == 1024.0**3

    def test_1tb_is_exactly_1024_to_the_4th(self) -> None:
        """1 TB = exactly 1024^4 bytes."""
        assert parse_bytes("1TB") == 1024.0**4

    def test_kb_to_b_ratio_is_1024(self) -> None:
        """Ratio KB/B = exactly 1024."""
        assert parse_bytes("1KB") / parse_bytes("1B") == 1024.0

    def test_mb_to_kb_ratio_is_1024(self) -> None:
        """Ratio MB/KB = exactly 1024."""
        assert parse_bytes("1MB") / parse_bytes("1KB") == 1024.0

    def test_gb_to_mb_ratio_is_1024(self) -> None:
        """Ratio GB/MB = exactly 1024."""
        assert parse_bytes("1GB") / parse_bytes("1MB") == 1024.0

    def test_tb_to_gb_ratio_is_1024(self) -> None:
        """Ratio TB/GB = exactly 1024."""
        assert parse_bytes("1TB") / parse_bytes("1GB") == 1024.0

    def test_gib_equals_gb(self) -> None:
        """GiB and GB are treated identically (both = 1024^3)."""
        assert parse_bytes("1GiB") == parse_bytes("1GB")

    def test_kib_equals_kb(self) -> None:
        """KiB and KB are treated identically (both = 1024)."""
        assert parse_bytes("1KiB") == parse_bytes("1KB")

    def test_mib_equals_mb(self) -> None:
        """MiB and MB are treated identically (both = 1024^2)."""
        assert parse_bytes("1MiB") == parse_bytes("1MB")

    def test_tib_equals_tb(self) -> None:
        """TiB and TB are treated identically (both = 1024^4)."""
        assert parse_bytes("1TiB") == parse_bytes("1TB")
