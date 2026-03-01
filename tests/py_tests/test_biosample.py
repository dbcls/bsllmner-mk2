"""Tests for biosample module."""

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.biosample import (
    FilterKeys,
    _load_filter_keys,
    construct_llm_input_json,
    is_ebi_format,
)
from bsllmner2.config import FILTER_KEYS_PATH

# === is_ebi_format ===


class TestIsEbiFormat:
    """Test cases for is_ebi_format function."""

    def test_real_ebi_entry(self) -> None:
        """Real EBI-style entry with list-of-dict characteristics returns True."""
        entry: dict[str, Any] = {
            "accession": "SAMEA12345",
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "cell_line": [{"text": "HeLa"}],
            },
        }
        assert is_ebi_format(entry) is True

    def test_empty_characteristics_dict(self) -> None:
        """Empty dict characteristics still returns True."""
        entry: dict[str, Any] = {"characteristics": {}}
        assert is_ebi_format(entry) is True

    def test_flat_dict_characteristics_detected_as_ebi(self) -> None:
        """Flat dict characteristics (NCBI-like) is also detected as EBI format.

        This is the precondition for Bug #3: is_ebi_format only checks
        isinstance(characteristics, dict), so flat dicts pass.
        """
        entry: dict[str, Any] = {
            "characteristics": {"cell_line": "HeLa", "organism": "Homo sapiens"},
        }
        assert is_ebi_format(entry) is True

    def test_no_characteristics_key(self) -> None:
        """Entry without characteristics key returns False."""
        entry: dict[str, Any] = {"accession": "SAMN00000001"}
        assert is_ebi_format(entry) is False

    def test_characteristics_is_list(self) -> None:
        """List characteristics returns False."""
        entry: dict[str, Any] = {"characteristics": [{"tag": "cell_line", "value": "HeLa"}]}
        assert is_ebi_format(entry) is False

    def test_characteristics_is_string(self) -> None:
        """String characteristics returns False."""
        entry: dict[str, Any] = {"characteristics": "some text"}
        assert is_ebi_format(entry) is False

    def test_characteristics_is_none(self) -> None:
        """None characteristics returns False."""
        entry: dict[str, Any] = {"characteristics": None}
        assert is_ebi_format(entry) is False

    def test_empty_dict(self) -> None:
        """Empty dict returns False."""
        entry: dict[str, Any] = {}
        assert is_ebi_format(entry) is False


# === construct_llm_input_json: EBI path ===


class TestConstructLlmInputJsonEbi:
    """Test cases for construct_llm_input_json with EBI-format entries."""

    def test_ebi_entry_extracts_text_field(self) -> None:
        """Standard EBI entry extracts value[0]["text"]."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "cell_line": [{"text": "HeLa", "ontologyTerms": ["http://example.org/HeLa"]}],
            },
        }
        result = construct_llm_input_json(entry)
        assert result == {"organism": "Homo sapiens", "cell_line": "HeLa"}

    def test_ebi_filter_keys_excluded(self) -> None:
        """Filter keys from filter_keys.json are excluded from output."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "External Id": [{"text": "ext-123"}],
                "SRA accession": [{"text": "SRA001"}],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "External Id" not in result
        assert "SRA accession" not in result

    def test_ebi_multiple_characteristics(self) -> None:
        """Multiple characteristics are all correctly extracted."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "cell_line": [{"text": "HeLa"}],
                "tissue": [{"text": "cervix"}],
            },
        }
        result = construct_llm_input_json(entry)
        assert result == {
            "organism": "Homo sapiens",
            "cell_line": "HeLa",
            "tissue": "cervix",
        }

    def test_ebi_empty_characteristics_dict(self) -> None:
        """Empty characteristics dict returns empty dict."""
        entry: dict[str, Any] = {"characteristics": {}}
        result = construct_llm_input_json(entry)
        assert result == {}


# === construct_llm_input_json: non-EBI path ===


class TestConstructLlmInputJsonNonEbi:
    """Test cases for construct_llm_input_json with non-EBI-format entries."""

    def test_non_ebi_preserves_all_values(self) -> None:
        """Non-EBI entry preserves all key-value pairs."""
        entry: dict[str, Any] = {
            "accession": "SAMN00000001",
            "title": "Test Sample",
            "organism": "Homo sapiens",
        }
        result = construct_llm_input_json(entry)
        assert result == entry

    def test_non_ebi_filter_keys_excluded(self) -> None:
        """Filter keys are excluded from non-EBI entries too."""
        entry: dict[str, Any] = {
            "organism": "Homo sapiens",
            "project name": "My Project",
            "gap_accession": "phs000001",
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "project name" not in result
        assert "gap_accession" not in result

    def test_empty_entry(self) -> None:
        """Empty entry returns empty dict."""
        result = construct_llm_input_json({})
        assert result == {}


# === construct_llm_input_json: bug fix verification ===


class TestConstructLlmInputJsonBugFixes:
    """Test cases verifying bug fixes in construct_llm_input_json."""

    def test_ebi_empty_list_skips_key(self) -> None:
        """BUG #1: Empty list value does not raise IndexError; key is skipped."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "cell_line": [],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "cell_line" not in result

    def test_ebi_missing_text_key_skips_key(self) -> None:
        """BUG #2: Dict without "text" key does not raise KeyError; key is skipped."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "cell_line": [{"ontologyTerms": ["http://example.org/HeLa"]}],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "cell_line" not in result

    def test_flat_dict_characteristics_skips_non_list_values(self) -> None:
        """BUG #3: Flat string value does not raise TypeError; key is skipped."""
        entry: dict[str, Any] = {
            "characteristics": {
                "cell_line": "HeLa",
                "organism": "Homo sapiens",
            },
        }
        result = construct_llm_input_json(entry)
        assert result == {}

    def test_ebi_value_is_int_skips_key(self) -> None:
        """BUG #3 variant: Integer value does not crash; key is skipped."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "replicate_count": 42,
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "replicate_count" not in result

    def test_ebi_list_of_strings_skips_key(self) -> None:
        """List of strings (not dicts) does not crash; key is skipped.

        Kills mutation on isinstance(value[0], dict) guard.
        """
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "tags": ["alpha", "beta"],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "tags" not in result

    def test_ebi_list_of_ints_skips_key(self) -> None:
        """List of integers does not crash; key is skipped.

        Kills mutation on isinstance(value[0], dict) guard.
        """
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "counts": [1, 2, 3],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "counts" not in result

    def test_ebi_list_of_none_skips_key(self) -> None:
        """List containing None does not crash; key is skipped."""
        entry: dict[str, Any] = {
            "characteristics": {
                "organism": [{"text": "Homo sapiens"}],
                "unknown": [None],
            },
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "unknown" not in result


# === construct_llm_input_json: property-based tests ===


# Strategy: generate arbitrary dict entries
_any_json_value = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=20),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=10,
)

_entry_strategy = st.dictionaries(st.text(max_size=20), _any_json_value, max_size=10)


class TestConstructLlmInputJsonProperty:
    """Property-based tests for construct_llm_input_json."""

    @given(entry=_entry_strategy)
    @settings(max_examples=200)
    def test_is_ebi_format_always_returns_bool(self, entry: dict[str, Any]) -> None:
        """is_ebi_format never raises and always returns bool."""
        result = is_ebi_format(entry)
        assert isinstance(result, bool)

    @given(entry=_entry_strategy)
    @settings(max_examples=200)
    def test_output_never_contains_filter_key(self, entry: dict[str, Any]) -> None:
        """Output never contains any key from filter_keys.json."""
        filter_keys = _load_filter_keys(FILTER_KEYS_PATH)
        result = construct_llm_input_json(entry)
        for key in filter_keys.filter_keys:
            assert key not in result

    @given(entry=_entry_strategy)
    @settings(max_examples=200)
    def test_ebi_output_values_are_strings(self, entry: dict[str, Any]) -> None:
        """When entry is EBI format, all output values are strings."""
        if not is_ebi_format(entry):
            return
        result = construct_llm_input_json(entry)
        for value in result.values():
            assert isinstance(value, str)

    @given(entry=_entry_strategy)
    @settings(max_examples=200)
    def test_never_raises_on_arbitrary_input(self, entry: dict[str, Any]) -> None:
        """construct_llm_input_json never raises on arbitrary dict input."""
        construct_llm_input_json(entry)


# === _load_filter_keys ===


class TestLoadFilterKeys:
    """Test cases for _load_filter_keys function."""

    def test_load_real_file(self) -> None:
        """Real filter_keys.json loads successfully."""
        result = _load_filter_keys(FILTER_KEYS_PATH)
        assert isinstance(result, FilterKeys)
        assert len(result.filter_keys) > 0

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_filter_keys(tmp_path.joinpath("nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Invalid JSON content raises json.JSONDecodeError."""
        invalid_file = tmp_path.joinpath("invalid.json")
        invalid_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            _load_filter_keys(invalid_file)
