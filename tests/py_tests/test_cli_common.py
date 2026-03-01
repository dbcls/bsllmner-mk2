"""Tests for CLI common utilities."""

import argparse
from typing import Any

import pytest

from bsllmner2.cli_common import BatchInfo, process_batches, str_to_bool


class TestStrToBool:
    """Tests for str_to_bool."""

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
    def test_true_values(self, value: str) -> None:
        """All documented true-ish strings return True."""
        assert str_to_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off"])
    def test_false_values(self, value: str) -> None:
        """All documented false-ish strings return False."""
        assert str_to_bool(value) is False

    @pytest.mark.parametrize("value", ["TRUE", "True", "tRuE", "YES", "Yes", "ON", "On"])
    def test_case_insensitive_true(self, value: str) -> None:
        """Case-insensitive matching for true values."""
        assert str_to_bool(value) is True

    @pytest.mark.parametrize("value", ["FALSE", "False", "fAlSe", "NO", "No", "OFF", "Off"])
    def test_case_insensitive_false(self, value: str) -> None:
        """Case-insensitive matching for false values."""
        assert str_to_bool(value) is False

    def test_invalid_value_raises_argument_type_error(self) -> None:
        """Invalid string raises ArgumentTypeError (not ValueError)."""
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid boolean value"):
            str_to_bool("invalid")

    def test_empty_string_raises_argument_type_error(self) -> None:
        """Empty string raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            str_to_bool("")

    def test_whitespace_padded_true(self) -> None:
        """Whitespace-padded input is stripped before matching."""
        assert str_to_bool(" true ") is True

    def test_numeric_true(self) -> None:
        """'1' returns True."""
        assert str_to_bool("1") is True

    def test_numeric_false(self) -> None:
        """'0' returns False."""
        assert str_to_bool("0") is False

    def test_two_is_not_truthy(self) -> None:
        """'2' is not a valid boolean value (unlike Python's bool(2))."""
        with pytest.raises(argparse.ArgumentTypeError):
            str_to_bool("2")

    def test_error_message_includes_value(self) -> None:
        """Error message includes the invalid value for debugging."""
        with pytest.raises(argparse.ArgumentTypeError, match="banana"):
            str_to_bool("banana")


class TestBatchInfo:
    """Test cases for BatchInfo dataclass."""

    def test_batch_info_creation(self) -> None:
        """Test creating a BatchInfo instance."""
        batch_info = BatchInfo(
            batch_idx=0,
            total_batches=3,
            start_idx=0,
            end_idx=10,
            entries=[{"accession": "SAMN001"}],
        )
        assert batch_info.batch_idx == 0
        assert batch_info.total_batches == 3
        assert batch_info.start_idx == 0
        assert batch_info.end_idx == 10
        assert len(batch_info.entries) == 1


class TestProcessBatches:
    """Test cases for process_batches function."""

    @pytest.mark.asyncio
    async def test_empty_entries(self) -> None:
        """Test that empty entries returns empty list."""

        async def _noop(x: BatchInfo) -> list[Any]:
            return []

        results: list[list[Any]] = await process_batches(
            entries=[],
            batch_size=10,
            process_fn=_noop,
            on_batch_complete=lambda idx, result: None,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_single_batch(self) -> None:
        """Test processing a single batch."""
        entries = [
            {"accession": "SAMN001"},
            {"accession": "SAMN002"},
            {"accession": "SAMN003"},
        ]
        processed_batches: list[BatchInfo] = []
        completed_batches: list[tuple[int, list[str]]] = []

        async def process_fn(batch_info: BatchInfo) -> list[str]:
            processed_batches.append(batch_info)
            return [e["accession"] for e in batch_info.entries]

        def on_complete(idx: int, result: list[str]) -> None:
            completed_batches.append((idx, result))

        results = await process_batches(
            entries=entries,
            batch_size=10,  # All entries fit in one batch
            process_fn=process_fn,
            on_batch_complete=on_complete,
        )

        assert len(results) == 1
        assert results[0] == ["SAMN001", "SAMN002", "SAMN003"]
        assert len(processed_batches) == 1
        assert processed_batches[0].batch_idx == 0
        assert processed_batches[0].total_batches == 1
        assert len(completed_batches) == 1
        assert completed_batches[0] == (0, ["SAMN001", "SAMN002", "SAMN003"])

    @pytest.mark.asyncio
    async def test_multiple_batches(self) -> None:
        """Test processing multiple batches."""
        entries = [{"accession": f"SAMN{i:03d}"} for i in range(1, 8)]  # 7 entries
        processed_batches: list[BatchInfo] = []
        completed_batches: list[tuple[int, list[str]]] = []

        async def process_fn(batch_info: BatchInfo) -> list[str]:
            processed_batches.append(batch_info)
            return [e["accession"] for e in batch_info.entries]

        def on_complete(idx: int, result: list[str]) -> None:
            completed_batches.append((idx, result))

        results = await process_batches(
            entries=entries,
            batch_size=3,  # 3 batches: [3, 3, 1]
            process_fn=process_fn,
            on_batch_complete=on_complete,
        )

        assert len(results) == 3
        assert results[0] == ["SAMN001", "SAMN002", "SAMN003"]
        assert results[1] == ["SAMN004", "SAMN005", "SAMN006"]
        assert results[2] == ["SAMN007"]

        assert len(processed_batches) == 3
        assert processed_batches[0].batch_idx == 0
        assert processed_batches[0].total_batches == 3
        assert processed_batches[0].start_idx == 0
        assert processed_batches[0].end_idx == 3

        assert processed_batches[1].batch_idx == 1
        assert processed_batches[1].start_idx == 3
        assert processed_batches[1].end_idx == 6

        assert processed_batches[2].batch_idx == 2
        assert processed_batches[2].start_idx == 6
        assert processed_batches[2].end_idx == 7

        assert len(completed_batches) == 3

    @pytest.mark.asyncio
    async def test_on_batch_complete_called_for_each_batch(self) -> None:
        """Test that on_batch_complete is called for each batch."""
        entries = [{"accession": f"SAMN{i:03d}"} for i in range(1, 11)]  # 10 entries
        complete_count = 0

        async def process_fn(batch_info: BatchInfo) -> int:
            return len(batch_info.entries)

        def on_complete(idx: int, result: int) -> None:
            nonlocal complete_count
            complete_count += 1

        await process_batches(
            entries=entries,
            batch_size=3,  # 4 batches: [3, 3, 3, 1]
            process_fn=process_fn,
            on_batch_complete=on_complete,
        )

        assert complete_count == 4

    @pytest.mark.asyncio
    async def test_batch_entries_are_correct_slices(self) -> None:
        """Test that batch entries are correct slices of the original."""
        entries = [{"id": i} for i in range(5)]
        batch_entries_list: list[list[dict[str, Any]]] = []

        async def process_fn(batch_info: BatchInfo) -> None:
            batch_entries_list.append(batch_info.entries)

        await process_batches(
            entries=entries,
            batch_size=2,
            process_fn=process_fn,
            on_batch_complete=lambda idx, result: None,
        )

        assert batch_entries_list[0] == [{"id": 0}, {"id": 1}]
        assert batch_entries_list[1] == [{"id": 2}, {"id": 3}]
        assert batch_entries_list[2] == [{"id": 4}]

    @pytest.mark.asyncio
    async def test_batch_size_equals_entries(self) -> None:
        """batch_size == len(entries) → exactly 1 batch."""
        entries = [{"id": i} for i in range(5)]
        batch_count = 0

        async def process_fn(batch_info: BatchInfo) -> int:
            nonlocal batch_count
            batch_count += 1
            assert batch_info.batch_idx == 0
            assert batch_info.total_batches == 1
            assert len(batch_info.entries) == 5
            return len(batch_info.entries)

        results = await process_batches(
            entries=entries,
            batch_size=5,
            process_fn=process_fn,
            on_batch_complete=lambda idx, result: None,
        )
        assert batch_count == 1
        assert results == [5]

    @pytest.mark.asyncio
    async def test_batch_size_one(self) -> None:
        """batch_size=1 → one batch per entry."""
        entries = [{"id": i} for i in range(3)]
        batch_sizes: list[int] = []

        async def process_fn(batch_info: BatchInfo) -> str:
            batch_sizes.append(len(batch_info.entries))
            return batch_info.entries[0]["id"]

        results = await process_batches(
            entries=entries,
            batch_size=1,
            process_fn=process_fn,
            on_batch_complete=lambda idx, result: None,
        )
        assert len(results) == 3
        assert all(s == 1 for s in batch_sizes)
        assert results == [0, 1, 2]
