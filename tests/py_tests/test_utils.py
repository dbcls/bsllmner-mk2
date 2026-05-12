"""Tests for :mod:`bsllmner2.utils`."""

from datetime import datetime, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bsllmner2.utils import _FALSY, _TRUTHY, get_now, parse_bool


class TestGetNow:
    """Tests for :func:`bsllmner2.utils.get_now`."""

    def test_returns_datetime(self) -> None:
        assert isinstance(get_now(), datetime)

    def test_returns_utc_aware_datetime(self) -> None:
        assert get_now().tzinfo == timezone.utc

    def test_monotonic_between_two_snapshots(self) -> None:
        before = datetime.now(timezone.utc)
        result = get_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


class TestParseBool:
    """Tests for :func:`bsllmner2.utils.parse_bool`."""

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"])
    def test_truthy_strings_all_supported(self, value: str) -> None:
        assert parse_bool(value) is True
        assert parse_bool(value, strict=True) is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "False", "0", "no", "NO", "off", "OFF"])
    def test_falsy_strings_all_supported(self, value: str) -> None:
        assert parse_bool(value) is False
        assert parse_bool(value, strict=True) is False

    @pytest.mark.parametrize("raw", [" true ", "\tfalse\n", "  on", "off  "])
    def test_strips_surrounding_whitespace(self, raw: str) -> None:
        expected = raw.strip().lower() in _TRUTHY
        assert parse_bool(raw, strict=True) is expected

    def test_strict_mode_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Invalid boolean value"):
            parse_bool("banana", strict=True)

    def test_nonstrict_unknown_returns_false(self) -> None:
        assert parse_bool("banana") is False
        assert parse_bool("") is False

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_passthrough(self, value: bool) -> None:
        assert parse_bool(value) is value
        assert parse_bool(value, strict=True) is value

    def test_empty_string_rejected_in_strict(self) -> None:
        with pytest.raises(ValueError):
            parse_bool("", strict=True)


class TestParseBoolPBT:
    """Property-based tests for :func:`bsllmner2.utils.parse_bool`."""

    @given(value=st.text(max_size=20))
    def test_unknown_strings_return_false_in_nonstrict(self, value: str) -> None:
        lower = value.strip().lower()
        if lower in _TRUTHY:
            assert parse_bool(value) is True
        elif lower in _FALSY:
            assert parse_bool(value) is False
        else:
            assert parse_bool(value) is False

    @given(value=st.text(max_size=20))
    def test_strict_rejects_everything_outside_known_sets(self, value: str) -> None:
        lower = value.strip().lower()
        if lower in _TRUTHY:
            assert parse_bool(value, strict=True) is True
        elif lower in _FALSY:
            assert parse_bool(value, strict=True) is False
        else:
            with pytest.raises(ValueError):
                parse_bool(value, strict=True)
