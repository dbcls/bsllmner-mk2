"""Tests for bsllmner2.benchmark module."""

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.benchmark import (
    StageTimer,
    aggregate_from_timing_fields,
    aggregate_llm_timings,
    compute_percentile,
    stage_timer,
)
from bsllmner2.models import LlmTimingFields
from tests.py_tests.conftest import make_chat_response_with_timing


class TestStageTimer:
    def test_elapsed_nonnegative(self) -> None:
        timer = StageTimer(name="test")
        timer.stop()
        assert timer.elapsed_sec >= 0

    def test_context_manager_sets_elapsed(self) -> None:
        with stage_timer("test") as timer:
            time.sleep(0.01)
        assert timer.elapsed_sec >= 0.005

    def test_stop_returns_elapsed(self) -> None:
        timer = StageTimer(name="test")
        result = timer.stop()
        assert result == timer.elapsed_sec

    def test_name_preserved(self) -> None:
        with stage_timer("my_stage") as timer:
            pass
        assert timer.name == "my_stage"


class TestComputePercentile:
    def test_empty_returns_zero(self) -> None:
        assert compute_percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert compute_percentile([42.0], 50) == 42.0

    def test_single_value_any_percentile(self) -> None:
        for p in [0, 25, 50, 75, 99, 100]:
            assert compute_percentile([42.0], p) == 42.0

    def test_two_values_p50(self) -> None:
        result = compute_percentile([10.0, 20.0], 50)
        assert result == pytest.approx(15.0)

    def test_known_distribution(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_percentile(values, 0) == pytest.approx(1.0)
        assert compute_percentile(values, 50) == pytest.approx(3.0)
        assert compute_percentile(values, 100) == pytest.approx(5.0)

    @given(st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_ordering_p50_le_p95_le_p99(self, values: list[float]) -> None:
        p50 = compute_percentile(values, 50)
        p95 = compute_percentile(values, 95)
        p99 = compute_percentile(values, 99)
        assert p50 <= p95 + 1e-9
        assert p95 <= p99 + 1e-9

    @given(st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_bounds(self, values: list[float]) -> None:
        for p in [0, 25, 50, 75, 99, 100]:
            result = compute_percentile(values, p)
            assert min(values) - 1e-9 <= result <= max(values) + 1e-9


class TestAggregateLlmTimings:
    def test_empty_list(self) -> None:
        summary = aggregate_llm_timings([])
        assert summary.call_count == 0
        assert summary.total_duration_sec == 0.0
        assert summary.mean_tokens_per_sec is None

    def test_single_response(self) -> None:
        resp = make_chat_response_with_timing(
            content="test",
            total_duration=1_000_000_000,  # 1 sec
            load_duration=100_000_000,  # 0.1 sec
            eval_count=50,
            eval_duration=500_000_000,  # 0.5 sec
            prompt_eval_count=100,
            prompt_eval_duration=200_000_000,
        )
        summary = aggregate_llm_timings([resp])
        assert summary.call_count == 1
        assert summary.total_duration_sec == pytest.approx(1.0)
        assert summary.mean_latency_sec == pytest.approx(0.9)  # (1.0 - 0.1)
        assert summary.mean_tokens_per_sec == pytest.approx(100.0)  # 50 / 0.5
        assert summary.total_prompt_tokens == 100
        assert summary.total_eval_tokens == 50
        assert summary.mean_load_duration_sec == pytest.approx(0.1)
        assert summary.max_load_duration_sec == pytest.approx(0.1)

    def test_load_duration_max(self) -> None:
        resp1 = make_chat_response_with_timing(
            content="a",
            total_duration=1_000_000_000,
            load_duration=100_000_000,
            eval_count=10,
            eval_duration=100_000_000,
        )
        resp2 = make_chat_response_with_timing(
            content="b",
            total_duration=1_000_000_000,
            load_duration=500_000_000,
            eval_count=10,
            eval_duration=100_000_000,
        )
        summary = aggregate_llm_timings([resp1, resp2])
        assert summary.max_load_duration_sec == pytest.approx(0.5)

    def test_zero_eval_duration_no_tokens_per_sec(self) -> None:
        resp = make_chat_response_with_timing(
            content="test",
            total_duration=1_000_000_000,
            load_duration=0,
            eval_count=0,
            eval_duration=0,
        )
        summary = aggregate_llm_timings([resp])
        assert summary.mean_tokens_per_sec is None

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=10),
                st.integers(min_value=100_000_000, max_value=10_000_000_000),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=30)
    def test_total_eval_tokens_equals_sum(self, data: list[tuple[int, int]]) -> None:
        responses = [
            make_chat_response_with_timing(
                content="x",
                eval_count=ec,
                eval_duration=ed,
            )
            for ec, ed in data
        ]
        summary = aggregate_llm_timings(responses)
        assert summary.total_eval_tokens == sum(ec for ec, _ in data)

    @given(
        st.lists(
            st.integers(min_value=100_000_000, max_value=10_000_000_000),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=30)
    def test_total_duration_approx_sum(self, durations: list[int]) -> None:
        responses = [make_chat_response_with_timing(content="x", total_duration=d) for d in durations]
        summary = aggregate_llm_timings(responses)
        expected = sum(d / 1e9 for d in durations)
        assert summary.total_duration_sec == pytest.approx(expected, rel=1e-6)


class TestAggregateFromTimingFields:
    def test_empty_list(self) -> None:
        summary = aggregate_from_timing_fields([])
        assert summary.call_count == 0
        assert summary.total_duration_sec == 0.0
        assert summary.mean_tokens_per_sec is None

    def test_single_field(self) -> None:
        field = LlmTimingFields(
            total_duration=1_000_000_000,
            load_duration=100_000_000,
            eval_count=50,
            eval_duration=500_000_000,
            prompt_eval_count=100,
        )
        summary = aggregate_from_timing_fields([field])
        assert summary.call_count == 1
        assert summary.total_duration_sec == pytest.approx(1.0)
        assert summary.mean_latency_sec == pytest.approx(0.9)
        assert summary.mean_tokens_per_sec == pytest.approx(100.0)
        assert summary.total_prompt_tokens == 100
        assert summary.total_eval_tokens == 50

    def test_consistent_with_aggregate_llm_timings(self) -> None:
        """aggregate_from_timing_fields should produce the same result as aggregate_llm_timings."""
        resp = make_chat_response_with_timing(
            content="test",
            total_duration=2_000_000_000,
            load_duration=200_000_000,
            eval_count=100,
            eval_duration=1_000_000_000,
            prompt_eval_count=200,
        )
        from_responses = aggregate_llm_timings([resp])
        from_fields = aggregate_from_timing_fields(
            [
                LlmTimingFields(
                    total_duration=2_000_000_000,
                    load_duration=200_000_000,
                    eval_count=100,
                    eval_duration=1_000_000_000,
                    prompt_eval_count=200,
                )
            ]
        )
        assert from_responses.call_count == from_fields.call_count
        assert from_responses.total_duration_sec == pytest.approx(from_fields.total_duration_sec)
        assert from_responses.mean_latency_sec == pytest.approx(from_fields.mean_latency_sec)
        assert from_responses.total_prompt_tokens == from_fields.total_prompt_tokens
        assert from_responses.total_eval_tokens == from_fields.total_eval_tokens
