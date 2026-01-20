"""Tests for CLI common utilities."""
import pytest

from bsllmner2.cli_common import BatchInfo, process_batches


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
        results = await process_batches(
            entries=[],
            batch_size=10,
            process_fn=lambda x: x,  # type: ignore
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
        entries = [
            {"accession": f"SAMN{i:03d}"} for i in range(1, 8)
        ]  # 7 entries
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
        batch_entries_list: list[list[dict]] = []

        async def process_fn(batch_info: BatchInfo) -> None:
            batch_entries_list.append(batch_info.entries)
            return None

        await process_batches(
            entries=entries,
            batch_size=2,
            process_fn=process_fn,
            on_batch_complete=lambda idx, result: None,
        )

        assert batch_entries_list[0] == [{"id": 0}, {"id": 1}]
        assert batch_entries_list[1] == [{"id": 2}, {"id": 3}]
        assert batch_entries_list[2] == [{"id": 4}]
