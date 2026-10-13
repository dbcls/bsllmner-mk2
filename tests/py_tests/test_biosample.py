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


# === construct_llm_input_json: NCBI path ===


def _ncbi_record() -> dict[str, Any]:
    """Return an NCBI-format record with every record-level key the production input carries."""
    return {
        "access": "public",
        "publication_date": "2021-04-19T22:11:48.000+09:00",
        "last_update": "2021-04-19T22:11:48.000+09:00",
        "submission_date": "2021-04-01T00:00:00.000+09:00",
        "id": 12345,
        "accession": "SAMN00000001",
        "Ids": {"Id": {"namespace": "BioSample", "is_primary": "1", "content": "SAMN00000001"}},
        "Description": {
            "SampleName": "HeLa_rep1",
            "Title": "ChIP-seq of HeLa",
            "Organism": {"taxonomy_id": "9606", "OrganismName": "Homo sapiens"},
            "Comment": {"Paragraph": "HeLa cells from a 31 years old donor"},
        },
        "Owner": {"Name": {"content": "Example University"}},
        "Models": {"Model": {"content": "Generic"}},
        "Package": {"content": "Generic.1.0"},
        "Links": {"Link": {"content": "PRJNA000001"}},
        "Status": {"status": "live"},
        "Attributes": {
            "Attribute": [
                {"attribute_name": "sample_name", "content": "HeLa_rep1"},
                {
                    "attribute_name": "cell line",
                    "content": "HeLa",
                    "harmonized_name": "cell_line",
                    "display_name": "cell line",
                },
                {"attribute_name": "External Id", "content": "SAMN00000001"},
            ],
        },
    }


class TestConstructLlmInputJsonNcbi:
    """Test cases for construct_llm_input_json with NCBI-format entries."""

    def test_ncbi_record_level_keys_dropped(self) -> None:
        """Only attributes and the three Description-derived keys reach the output."""
        result = construct_llm_input_json(_ncbi_record())
        assert result == {
            "sample_name": "HeLa_rep1",
            "cell line": "HeLa",
            "title": "ChIP-seq of HeLa",
            "organism": "Homo sapiens",
            "description": "HeLa cells from a 31 years old donor",
        }

    def test_ncbi_wrapped_entry_matches_unwrapped_entry(self) -> None:
        """A record under a BioSample key produces the same output as a top-level record."""
        record = _ncbi_record()
        wrapped: dict[str, Any] = {"BioSample": record, "accession": "SAMN00000001"}
        assert construct_llm_input_json(wrapped) == construct_llm_input_json(record)

    def test_ncbi_filter_key_attribute_excluded(self) -> None:
        """An attribute whose attribute_name is a filter key is dropped."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {"attribute_name": "cell line", "content": "HeLa"},
                    {"attribute_name": "SRA accession", "content": "DRS136282"},
                    {"attribute_name": "INSDC center name", "content": "Example University"},
                ],
            },
        }
        result = construct_llm_input_json(entry)
        assert result == {"cell line": "HeLa"}

    def test_ncbi_filter_key_dropped_even_when_repeated(self) -> None:
        """A repeated filter key never reaches the output as a list."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {"attribute_name": "vendor", "content": "Vendor A"},
                    {"attribute_name": "vendor", "content": "Vendor B"},
                ],
            },
        }
        assert construct_llm_input_json(entry) == {}

    def test_ncbi_harmonized_name_filter_key_kept(self) -> None:
        """Filter keys are matched against attribute_name only, not harmonized_name."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {
                        "attribute_name": "cell line",
                        "content": "HeLa",
                        "harmonized_name": "vendor",
                    },
                ],
            },
        }
        assert construct_llm_input_json(entry) == {"cell line": "HeLa"}

    def test_ncbi_repeated_attribute_name_becomes_list(self) -> None:
        """Distinct values of one attribute name are all kept, in input order."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {"attribute_name": "tissue", "content": "cervix"},
                    {"attribute_name": "tissue", "content": "epithelium"},
                    {"attribute_name": "tissue", "content": "uterus"},
                ],
            },
        }
        assert construct_llm_input_json(entry) == {"tissue": ["cervix", "epithelium", "uterus"]}

    def test_ncbi_repeated_identical_values_collapse(self) -> None:
        """Identical values of one attribute name collapse into a single string."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {"attribute_name": "tissue", "content": "cervix"},
                    {"attribute_name": "tissue", "content": "cervix"},
                ],
            },
        }
        assert construct_llm_input_json(entry) == {"tissue": "cervix"}

    def test_ncbi_attribute_as_single_object(self) -> None:
        """Attributes.Attribute may be a single object instead of a list."""
        entry: dict[str, Any] = {
            "Attributes": {"Attribute": {"attribute_name": "cell line", "content": "HeLa"}},
        }
        assert construct_llm_input_json(entry) == {"cell line": "HeLa"}

    def test_ncbi_attributes_none_keeps_description(self) -> None:
        """A null Attributes still yields the Description-derived keys."""
        entry: dict[str, Any] = {
            "Attributes": None,
            "Description": {
                "Title": "ChIP-seq of HeLa",
                "Organism": {"taxonomy_id": "9606", "OrganismName": "Homo sapiens"},
            },
        }
        assert construct_llm_input_json(entry) == {
            "title": "ChIP-seq of HeLa",
            "organism": "Homo sapiens",
        }

    def test_ncbi_organism_falls_back_to_taxonomy_name(self) -> None:
        """When OrganismName is absent, taxonomy_name supplies the organism."""
        entry: dict[str, Any] = {
            "Description": {"Organism": {"taxonomy_id": "9606", "taxonomy_name": "Homo sapiens"}},
        }
        assert construct_llm_input_json(entry) == {"organism": "Homo sapiens"}

    def test_ncbi_organism_prefers_organism_name(self) -> None:
        """OrganismName wins over taxonomy_name when both are present."""
        entry: dict[str, Any] = {
            "Description": {
                "Organism": {
                    "taxonomy_id": "9606",
                    "OrganismName": "Homo sapiens",
                    "taxonomy_name": "human",
                },
            },
        }
        assert construct_llm_input_json(entry) == {"organism": "Homo sapiens"}

    def test_ncbi_organism_absent_when_no_name(self) -> None:
        """A taxonomy_id alone yields no organism key."""
        entry: dict[str, Any] = {"Description": {"Organism": {"taxonomy_id": "9606"}}}
        assert construct_llm_input_json(entry) == {}

    def test_ncbi_attribute_colliding_with_title_becomes_list(self) -> None:
        """An attribute named title keeps its value alongside Description.Title."""
        entry: dict[str, Any] = {
            "Attributes": {"Attribute": [{"attribute_name": "title", "content": "HeLa rep1"}]},
            "Description": {"Title": "ChIP-seq of HeLa"},
        }
        assert construct_llm_input_json(entry) == {"title": ["HeLa rep1", "ChIP-seq of HeLa"]}

    def test_ncbi_attribute_matching_organism_stays_string(self) -> None:
        """An attribute named organism with the same value does not become a list."""
        entry: dict[str, Any] = {
            "Attributes": {"Attribute": [{"attribute_name": "organism", "content": "Homo sapiens"}]},
            "Description": {"Organism": {"OrganismName": "Homo sapiens"}},
        }
        assert construct_llm_input_json(entry) == {"organism": "Homo sapiens"}

    def test_ncbi_empty_and_non_string_values_dropped(self) -> None:
        """An attribute without a non-empty string name and content is dropped."""
        entry: dict[str, Any] = {
            "Attributes": {
                "Attribute": [
                    {"attribute_name": "cell line", "content": "HeLa"},
                    {"attribute_name": "tissue", "content": ""},
                    {"attribute_name": "", "content": "cervix"},
                    {"attribute_name": "age", "content": 42},
                    {"attribute_name": "sex", "content": None},
                    {"content": "orphan"},
                    "not an object",
                ],
            },
        }
        assert construct_llm_input_json(entry) == {"cell line": "HeLa"}

    def test_ncbi_empty_description_and_attributes(self) -> None:
        """An NCBI record with no usable content yields an empty dict."""
        entry: dict[str, Any] = {"Attributes": {}, "Description": {}}
        assert construct_llm_input_json(entry) == {}

    def test_ncbi_comment_without_paragraph_dropped(self) -> None:
        """A Comment without a Paragraph yields no description key."""
        entry: dict[str, Any] = {
            "Description": {"Title": "ChIP-seq of HeLa", "Comment": {"Table": {"caption": "x"}}},
        }
        assert construct_llm_input_json(entry) == {"title": "ChIP-seq of HeLa"}


# === construct_llm_input_json: other formats ===


class TestConstructLlmInputJsonOther:
    """Test cases for entries in neither EBI nor NCBI format."""

    def test_other_preserves_all_values(self) -> None:
        """A flat entry keeps all of its key-value pairs."""
        entry: dict[str, Any] = {
            "accession": "SAMN00000001",
            "title": "Test Sample",
            "organism": "Homo sapiens",
        }
        result = construct_llm_input_json(entry)
        assert result == entry

    def test_other_filter_keys_excluded(self) -> None:
        """Filter keys are excluded from a flat entry too."""
        entry: dict[str, Any] = {
            "organism": "Homo sapiens",
            "project name": "My Project",
            "gap_accession": "phs000001",
        }
        result = construct_llm_input_json(entry)
        assert "organism" in result
        assert "project name" not in result
        assert "gap_accession" not in result

    def test_non_dict_biosample_value_is_not_ncbi(self) -> None:
        """A BioSample key that is not an object does not select the NCBI path."""
        entry: dict[str, Any] = {"BioSample": "SAMN00000001", "organism": "Homo sapiens"}
        assert construct_llm_input_json(entry) == entry

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

# Strategy: NCBI-format records whose attribute names collide and whose values repeat
_attribute_name_strategy = st.sampled_from(
    ["cell line", "tissue", "title", "organism", "description", "SRA accession", "vendor", ""],
)
_attribute_strategy = st.fixed_dictionaries({
    "attribute_name": _attribute_name_strategy,
    "content": st.sampled_from(["HeLa", "cervix", "Homo sapiens", ""]),
})
_ncbi_record_strategy = st.fixed_dictionaries({
    "accession": st.just("SAMN00000001"),
    "Owner": st.just({"Name": {"content": "Example University"}}),
    "Attributes": st.fixed_dictionaries({
        "Attribute": st.lists(_attribute_strategy, max_size=6),
    }),
    "Description": st.fixed_dictionaries({
        "Title": st.sampled_from(["ChIP-seq of HeLa", ""]),
        "Organism": st.fixed_dictionaries({"OrganismName": st.sampled_from(["Homo sapiens", ""])}),
    }),
})


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

    @given(record=_ncbi_record_strategy)
    @settings(max_examples=200)
    def test_ncbi_output_values_are_strings_or_string_lists(self, record: dict[str, Any]) -> None:
        """NCBI output values are strings, or lists of at least two distinct strings."""
        result = construct_llm_input_json(record)
        for value in result.values():
            if isinstance(value, list):
                assert len(value) >= 2
                assert len(set(value)) == len(value)
                assert all(isinstance(item, str) for item in value)
            else:
                assert isinstance(value, str)

    @given(record=_ncbi_record_strategy)
    @settings(max_examples=200)
    def test_ncbi_keeps_every_unfiltered_attribute(self, record: dict[str, Any]) -> None:
        """Every attribute that is not a filter key reaches the output with its value."""
        filter_keys = set(_load_filter_keys(FILTER_KEYS_PATH).filter_keys)
        result = construct_llm_input_json(record)
        for attribute in record["Attributes"]["Attribute"]:
            name = attribute["attribute_name"]
            content = attribute["content"]
            if name in filter_keys or name == "" or content == "":
                continue
            value = result[name]
            assert content in value if isinstance(value, list) else content == value

    @given(record=_ncbi_record_strategy)
    @settings(max_examples=200)
    def test_ncbi_output_keys_are_attribute_names_or_description_keys(self, record: dict[str, Any]) -> None:
        """No record-level key of an NCBI entry reaches the output."""
        names = {attribute["attribute_name"] for attribute in record["Attributes"]["Attribute"]}
        allowed = names | {"title", "organism", "description"}
        assert set(construct_llm_input_json(record)) <= allowed

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
