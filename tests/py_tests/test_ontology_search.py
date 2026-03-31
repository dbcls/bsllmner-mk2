"""Tests for ontology_search module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bsllmner2.models import OntologyIndex, SearchResult, TermAnnotation
from bsllmner2.ontology_search import (
    DEFAULT_PREFIX_MAP,
    _collect_joiners,
    _expand_prop_uri,
    _generate_windows,
    _normalize_key,
    _normalize_term_id,
    _split_ws,
    _tokenize_atom,
    build_index_from_file,
    build_index_from_owl,
    build_index_from_table,
    build_word_combinations,
    is_label_prop,
    search_terms,
    search_terms_with_text2term,
)

# === _normalize_term_id ===


class TestNormalizeTermId:
    """Test cases for _normalize_term_id function."""

    def test_underscore_to_colon(self) -> None:
        """Standard underscore-separated ID is converted to colon form."""
        assert _normalize_term_id("CVCL_0384") == "CVCL:0384"

    def test_underscore_with_alpha_suffix(self) -> None:
        """Underscore ID with alphanumeric suffix is converted."""
        assert _normalize_term_id("CVCL_R965") == "CVCL:R965"

    def test_colon_form_unchanged(self) -> None:
        """Already colon-separated ID is unchanged."""
        assert _normalize_term_id("CVCL:0384") == "CVCL:0384"

    def test_obo_hash_fragment(self) -> None:
        """OBO-style hash fragment extracts the local part."""
        assert _normalize_term_id("OBO:CELLOSAURUS#CVCL_R965") == "CVCL:R965"

    def test_http_url_extracts_local_name(self) -> None:
        """HTTP URL extracts the last path segment and normalizes."""
        result = _normalize_term_id("http://purl.obolibrary.org/obo/CVCL_R965")
        assert result == "CVCL:R965"

    def test_https_url_extracts_local_name(self) -> None:
        """HTTPS URL also extracts the last path segment."""
        result = _normalize_term_id("https://purl.obolibrary.org/obo/CL_0000001")
        assert result == "CL:0000001"

    def test_url_with_fragment(self) -> None:
        """URL with fragment extracts the fragment."""
        result = _normalize_term_id("http://example.org/ontology#GO_0008150")
        assert result == "GO:0008150"

    def test_percent_encoded_url(self) -> None:
        """Percent-encoded URL is decoded."""
        result = _normalize_term_id("http://example.org/obo/CL%5F0000001")
        assert result == "CL:0000001"

    def test_nfkc_normalization(self) -> None:
        """Full-width characters are NFKC-normalized."""
        # Full-width C, L (U+FF23, U+FF2C)
        result = _normalize_term_id("\uff23\uff2c_0000001")
        assert result == "CL:0000001"

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _normalize_term_id("  CVCL_0384  ") == "CVCL:0384"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _normalize_term_id("") == ""

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns empty string."""
        assert _normalize_term_id("   ") == ""

    def test_no_separator(self) -> None:
        """ID without colon or underscore is returned as-is."""
        assert _normalize_term_id("GO0008150") == "GO0008150"

    def test_multiple_underscores_splits_first(self) -> None:
        """Multiple underscores: only the first is used for split."""
        result = _normalize_term_id("FOO_BAR_BAZ")
        assert result == "FOO:BAR_BAZ"

    def test_leading_underscore_no_prefix(self) -> None:
        """Leading underscore means empty prefix part."""
        result = _normalize_term_id("_0384")
        assert result == "_0384"

    def test_trailing_underscore_no_local(self) -> None:
        """Trailing underscore means empty local part."""
        result = _normalize_term_id("CVCL_")
        assert result == "CVCL_"

    def test_single_underscore(self) -> None:
        """Single underscore with empty parts returns as-is."""
        assert _normalize_term_id("_") == "_"

    def test_colon_with_http_prefix_prefix_returned_as_is(self) -> None:
        """Colon form with http-like prefix is returned."""
        result = _normalize_term_id("http:something")
        # http prefix triggers URL path, but no "://" so treated as colon form
        # The first part is "http" which starts with "http" → goes to underscore check
        # No underscore → returns "http:something"
        assert result == "http:something"


class TestNormalizeTermIdProperty:
    """Property-based tests for _normalize_term_id."""

    @given(text=st.text(max_size=200))
    @settings(max_examples=200)
    def test_idempotent(self, text: str) -> None:
        """Applying _normalize_term_id twice gives the same result."""
        once = _normalize_term_id(text)
        twice = _normalize_term_id(once)
        assert once == twice

    @given(text=st.text(max_size=200))
    @settings(max_examples=200)
    def test_never_raises(self, text: str) -> None:
        """_normalize_term_id never raises on arbitrary input."""
        _normalize_term_id(text)

    @given(
        prefix=st.from_regex(r"[A-Z]{2,6}", fullmatch=True), local=st.from_regex(r"[0-9A-Za-z]{1,10}", fullmatch=True)
    )
    @settings(max_examples=200)
    def test_underscore_always_becomes_colon(self, prefix: str, local: str) -> None:
        """PREFIX_LOCAL always normalizes to PREFIX:LOCAL."""
        result = _normalize_term_id(f"{prefix}_{local}")
        assert result == f"{prefix}:{local}"

    @given(
        prefix=st.from_regex(r"[A-Z]{2,6}", fullmatch=True), local=st.from_regex(r"[0-9A-Za-z]{1,10}", fullmatch=True)
    )
    @settings(max_examples=200)
    def test_colon_form_is_stable(self, prefix: str, local: str) -> None:
        """PREFIX:LOCAL is a fixed point."""
        colon_form = f"{prefix}:{local}"
        assert _normalize_term_id(colon_form) == colon_form

    @given(text=st.text(max_size=200))
    @settings(max_examples=200)
    def test_result_stripped(self, text: str) -> None:
        """Result never has leading/trailing whitespace."""
        result = _normalize_term_id(text)
        assert result == result.strip()

    @given(
        path=st.from_regex(r"/[a-z]{1,10}/[A-Z]{2,6}_[0-9]{4,8}", fullmatch=True),
    )
    @settings(max_examples=200)
    def test_url_input_no_scheme_in_result(self, path: str) -> None:
        """URL input never retains '://' in result."""
        url = f"http://example.org{path}"
        result = _normalize_term_id(url)
        assert "://" not in result


# === _expand_prop_uri ===


class TestExpandPropUri:
    """Test cases for _expand_prop_uri function."""

    def test_rdfs_label(self) -> None:
        """rdfs:label expands to full URI."""
        result = _expand_prop_uri("rdfs:label")
        assert result == "http://www.w3.org/2000/01/rdf-schema#label"

    def test_skos_pref_label_expands_to_correct_uri(self) -> None:
        """Bug #3 verification: skos:prefLabel expands to the correct SKOS core URI."""
        result = _expand_prop_uri("skos:prefLabel")
        assert result == "http://www.w3.org/2004/02/skos/core#prefLabel"

    def test_skos_alt_label(self) -> None:
        """skos:altLabel expands correctly."""
        result = _expand_prop_uri("skos:altLabel")
        assert result == "http://www.w3.org/2004/02/skos/core#altLabel"

    def test_obo_in_owl_has_exact_synonym(self) -> None:
        """oboInOwl:hasExactSynonym expands correctly."""
        result = _expand_prop_uri("oboInOwl:hasExactSynonym")
        assert result == "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"

    def test_full_uri_passthrough(self) -> None:
        """Full URI is returned unchanged."""
        uri = "http://www.w3.org/2000/01/rdf-schema#label"
        assert _expand_prop_uri(uri) == uri

    def test_unknown_prefix(self) -> None:
        """Unknown prefix returns original string."""
        assert _expand_prop_uri("foo:bar") == "foo:bar"

    def test_no_prefix_no_colon(self) -> None:
        """String without colon is returned as-is."""
        assert _expand_prop_uri("justAString") == "justAString"

    def test_empty_string(self) -> None:
        """Empty string is returned as-is."""
        assert _expand_prop_uri("") == ""

    def test_prefix_with_empty_local(self) -> None:
        """Prefix with empty local part expands to namespace."""
        result = _expand_prop_uri("rdfs:")
        assert result == "http://www.w3.org/2000/01/rdf-schema#"

    def test_https_uri_passthrough(self) -> None:
        """HTTPS URI is returned unchanged."""
        uri = "https://example.org/prop"
        assert _expand_prop_uri(uri) == uri


# === _normalize_key ===


class TestNormalizeKey:
    """Test cases for _normalize_key function."""

    def test_casefold(self) -> None:
        """Upper case is casefolded."""
        assert _normalize_key("HeLa") == "hela"

    def test_strip_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _normalize_key("  hello  ") == "hello"

    def test_nfkc_normalization(self) -> None:
        """Full-width characters are NFKC-normalized and casefolded."""
        # Full-width A (U+FF21)
        assert _normalize_key("\uff21") == "a"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert _normalize_key("") == ""


class TestNormalizeKeyProperty:
    """Property-based tests for _normalize_key."""

    @given(text=st.text(max_size=200))
    @settings(max_examples=200)
    def test_idempotent(self, text: str) -> None:
        """Applying _normalize_key twice gives the same result."""
        once = _normalize_key(text)
        twice = _normalize_key(once)
        assert once == twice

    @given(text=st.text(max_size=200))
    @settings(max_examples=200)
    def test_result_is_casefolded(self, text: str) -> None:
        """Result is always casefolded."""
        result = _normalize_key(text)
        assert result == result.casefold()


# === is_label_prop ===


class TestIsLabelProp:
    """Test cases for is_label_prop function."""

    def test_rdfs_label_is_true(self) -> None:
        """rdfs:label full URI is recognized as label prop."""
        assert is_label_prop("http://www.w3.org/2000/01/rdf-schema#label") is True

    def test_skos_pref_label_is_true(self) -> None:
        """skos:prefLabel full URI is recognized as label prop."""
        skos_prefix = DEFAULT_PREFIX_MAP["skos"]
        assert is_label_prop(skos_prefix + "prefLabel") is True

    def test_synonym_is_false(self) -> None:
        """oboInOwl:hasExactSynonym is not a label prop."""
        oio_prefix = DEFAULT_PREFIX_MAP["oboInOwl"]
        assert is_label_prop(oio_prefix + "hasExactSynonym") is False

    def test_none_is_false(self) -> None:
        """None returns False."""
        assert is_label_prop(None) is False

    def test_empty_string_is_false(self) -> None:
        """Empty string returns False."""
        assert is_label_prop("") is False

    def test_arbitrary_uri_is_false(self) -> None:
        """Arbitrary URI not in label set returns False."""
        assert is_label_prop("http://example.org/someProp") is False


# === _split_ws ===


class TestSplitWs:
    """Test cases for _split_ws function."""

    def test_single_word(self) -> None:
        """Single word returns list of one."""
        assert _split_ws("hello") == ["hello"]

    def test_multiple_spaces(self) -> None:
        """Multiple spaces between words are collapsed."""
        assert _split_ws("hello   world") == ["hello", "world"]

    def test_tabs_and_newlines(self) -> None:
        """Tabs and newlines act as separators."""
        assert _split_ws("hello\tworld\nfoo") == ["hello", "world", "foo"]

    def test_leading_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _split_ws("  hello world  ") == ["hello", "world"]

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert _split_ws("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns empty list."""
        assert _split_ws("   ") == []

    def test_single_char(self) -> None:
        """Single character returns list of one."""
        assert _split_ws("x") == ["x"]


# === _tokenize_atom ===


class TestTokenizeAtom:
    """Test cases for _tokenize_atom function."""

    def test_camel_case_inserts_space(self) -> None:
        """CamelCase boundary inserts space within single token.

        _tokenize_atom inserts spaces at boundaries via regex sub, but
        BASE_DELIMS_RE does not include space, so the result is one string.
        """
        assert _tokenize_atom("SoxGene") == ["Sox Gene"]

    def test_multi_word_camel_inserts_spaces(self) -> None:
        """Multi-word CamelCase inserts spaces at all boundaries."""
        assert _tokenize_atom("MyLongName") == ["My Long Name"]

    def test_all_lowercase(self) -> None:
        """All-lowercase returns single token."""
        assert _tokenize_atom("hello") == ["hello"]

    def test_all_uppercase(self) -> None:
        """All-uppercase returns single token (no boundary detected)."""
        assert _tokenize_atom("HELLO") == ["HELLO"]

    def test_alpha_digit_boundary_inserts_space(self) -> None:
        """Alpha-digit boundary inserts space within single token."""
        assert _tokenize_atom("Sox11") == ["Sox 11"]

    def test_digit_alpha_boundary_inserts_space(self) -> None:
        """Digit-alpha boundary inserts space within single token."""
        assert _tokenize_atom("123abc") == ["123 abc"]

    def test_hyphen_split(self) -> None:
        """Hyphen delimiter splits into separate tokens."""
        assert _tokenize_atom("cell-line") == ["cell", "line"]

    def test_underscore_split(self) -> None:
        """Underscore delimiter splits into separate tokens."""
        assert _tokenize_atom("cell_line") == ["cell", "line"]

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert _tokenize_atom("") == []

    def test_combined_camel_and_digit(self) -> None:
        """Combined CamelCase and digit boundary inserts spaces."""
        assert _tokenize_atom("MyGene2B") == ["My Gene 2 B"]


# === _collect_joiners ===


class TestCollectJoiners:
    """Test cases for _collect_joiners function."""

    def test_space_always_included(self) -> None:
        """Space is always in the result."""
        assert " " in _collect_joiners("hello world")

    def test_hyphen_detected(self) -> None:
        """Hyphen in query is detected as joiner."""
        joiners = _collect_joiners("cell-line")
        assert "-" in joiners

    def test_slash_detected(self) -> None:
        """Slash in query is detected."""
        joiners = _collect_joiners("RNA/DNA")
        assert "/" in joiners

    def test_no_extra_joiners_for_plain_text(self) -> None:
        """Plain text has only space as joiner."""
        assert _collect_joiners("hello world") == [" "]

    def test_empty_string(self) -> None:
        """Empty string has only space as joiner."""
        assert _collect_joiners("") == [" "]


# === _generate_windows ===


class TestGenerateWindows:
    """Test cases for _generate_windows function."""

    def test_single_token(self) -> None:
        """Single token generates one window."""
        windows = list(_generate_windows(["a"], 7))
        assert windows == [(0, 1)]

    def test_two_tokens(self) -> None:
        """Two tokens generate windows longest first."""
        windows = list(_generate_windows(["a", "b"], 7))
        assert windows == [(0, 2), (0, 1), (1, 2)]

    def test_max_ngram_limits_window_size(self) -> None:
        """max_ngram limits the maximum window size."""
        tokens = ["a", "b", "c", "d"]
        windows = list(_generate_windows(tokens, 2))
        # Should only have size 2 and size 1 windows
        for start, end in windows:
            assert end - start <= 2

    def test_descending_order(self) -> None:
        """Windows are generated in descending n-gram size order."""
        tokens = ["a", "b", "c"]
        windows = list(_generate_windows(tokens, 3))
        sizes = [end - start for start, end in windows]
        # Sizes should be non-increasing (descending)
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1]

    def test_empty_tokens(self) -> None:
        """Empty token list generates no windows."""
        assert list(_generate_windows([], 7)) == []


# === build_word_combinations ===


class TestBuildWordCombinations:
    """Test cases for build_word_combinations function."""

    def test_simple_query(self) -> None:
        """Simple query produces normalized original as first element."""
        result = build_word_combinations("hello world")
        assert result[0] == "hello world"

    def test_camel_case_produces_split_tokens(self) -> None:
        """Bug #2 verification: CamelCase query produces split token combinations."""
        result = build_word_combinations("SoxGene")
        assert "sox gene" in result

    def test_camel_case_original_preserved(self) -> None:
        """CamelCase query preserves normalized original."""
        result = build_word_combinations("SoxGene")
        assert result[0] == "soxgene"

    def test_digit_boundary_split(self) -> None:
        """Alpha-digit boundary split is present in combinations."""
        result = build_word_combinations("Sox11")
        assert "sox 11" in result

    def test_hyphenated_query_produces_joiner_variants(self) -> None:
        """Hyphenated query produces hyphen-joined variants."""
        result = build_word_combinations("cell-line")
        assert "cell-line" in result
        assert "cell line" in result

    def test_no_duplicates(self) -> None:
        """Result contains no duplicates."""
        result = build_word_combinations("hello world")
        assert len(result) == len(set(result))

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert build_word_combinations("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns empty list."""
        assert build_word_combinations("   ") == []

    def test_max_ngram_zero(self) -> None:
        """max_ngram=0 returns empty list."""
        assert build_word_combinations("hello", max_ngram=0) == []

    def test_single_token(self) -> None:
        """Single token returns at least the normalized original."""
        result = build_word_combinations("hello")
        assert "hello" in result

    def test_multi_camel_case_produces_split_form(self) -> None:
        """Multi-part CamelCase produces space-separated form.

        Bug #2 verification: after fix, _tokenize_atom runs on original case,
        producing "My Long Name" as one token. After casefold, "my long name"
        appears as a combination.
        """
        result = build_word_combinations("MyLongName")
        assert "my long name" in result

    def test_underscore_query(self) -> None:
        """Underscore in query produces underscore-joined variant."""
        result = build_word_combinations("cell_line")
        assert "cell_line" in result
        assert "cell line" in result

    def test_slash_query(self) -> None:
        """Slash in query produces slash-joined variant."""
        result = build_word_combinations("RNA/DNA")
        assert "rna/dna" in result
        assert "rna dna" in result

    def test_first_element_is_normalized_original(self) -> None:
        """First element is always the normalized original query."""
        result = build_word_combinations("  Hello World  ")
        assert result[0] == "hello world"

    def test_results_are_casefolded(self) -> None:
        """All results are casefolded."""
        result = build_word_combinations("SoxGene")
        for r in result:
            assert r == r.casefold()


class TestBuildWordCombinationsProperty:
    """Property-based tests for build_word_combinations."""

    @given(query=st.text(min_size=1, max_size=50, alphabet=st.characters(categories=("L", "N", "Zs"))))
    @settings(max_examples=200)
    def test_no_duplicates(self, query: str) -> None:
        """Result never contains duplicates."""
        result = build_word_combinations(query)
        assert len(result) == len(set(result))

    @given(query=st.text(min_size=1, max_size=50, alphabet=st.characters(categories=("L", "N", "Zs"))))
    @settings(max_examples=200)
    def test_first_element_is_normalized(self, query: str) -> None:
        """First element is the normalized original when result is non-empty."""
        result = build_word_combinations(query)
        if result:
            assert result[0] == _normalize_key(query)

    @given(query=st.text(max_size=200))
    @settings(max_examples=200)
    def test_never_raises(self, query: str) -> None:
        """build_word_combinations never raises on arbitrary input."""
        build_word_combinations(query)

    @given(query=st.text(min_size=1, max_size=50, alphabet=st.characters(categories=("L", "N", "Zs"))))
    @settings(max_examples=200)
    def test_all_results_casefolded(self, query: str) -> None:
        """All results are casefolded."""
        for r in build_word_combinations(query):
            assert r == r.casefold()

    @given(
        query=st.text(min_size=1, max_size=50),
        max_ngram=st.integers(max_value=0),
    )
    @settings(max_examples=200)
    def test_empty_for_nonpositive_ngram(self, query: str, max_ngram: int) -> None:
        """max_ngram <= 0 always returns empty list."""
        assert build_word_combinations(query, max_ngram=max_ngram) == []


# === build_index_from_table ===


class TestBuildIndexFromTable:
    """Test cases for build_index_from_table function."""

    def test_tsv_basic(self, tmp_path: Path) -> None:
        """Basic TSV file is read and indexed correctly."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CL:0000001\trdfs:label\tneuron\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "neuron" in index.value_to_annotations
        assert index.value_to_annotations["neuron"][0].term_id == "CL:0000001"

    def test_csv_basic(self, tmp_path: Path) -> None:
        """Basic CSV file is read and indexed correctly."""
        csv_file = tmp_path.joinpath("test.csv")
        csv_file.write_text("CL:0000001,rdfs:label,neuron\n", encoding="utf-8")
        index = build_index_from_table(csv_file)
        assert "neuron" in index.value_to_annotations

    def test_label_found_for_underscore_term_id(self, tmp_path: Path) -> None:
        """Bug #1 verification: TSV with underscore term_id has labels keyed by normalized ID."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CVCL_0384\trdfs:label\tHeLa\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        # The normalized form is "CVCL:0384"
        assert "CVCL:0384" in index.term_id_to_labels
        assert "HeLa" in index.term_id_to_labels["CVCL:0384"]

    def test_label_not_stored_under_raw_id(self, tmp_path: Path) -> None:
        """Bug #1 verification: raw underscore ID should not be in labels dict."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CVCL_0384\trdfs:label\tHeLa\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "CVCL_0384" not in index.term_id_to_labels

    def test_synonym_not_in_labels(self, tmp_path: Path) -> None:
        """Synonym properties are not added to term_id_to_labels."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CL:0000001\toboInOwl:hasExactSynonym\tnerve cell\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "CL:0000001" not in index.term_id_to_labels
        # But it should be in value_to_annotations
        assert "nerve cell" in index.value_to_annotations

    def test_multiple_labels_same_term(self, tmp_path: Path) -> None:
        """Multiple labels for same term are all stored."""
        tsv = tmp_path.joinpath("test.tsv")
        lines = "CL:0000001\trdfs:label\tneuron\nCL:0000001\tskos:prefLabel\tnerve cell\n"
        tsv.write_text(lines, encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "CL:0000001" in index.term_id_to_labels
        labels = index.term_id_to_labels["CL:0000001"]
        assert "neuron" in labels
        assert "nerve cell" in labels

    def test_duplicate_label_deduplication(self, tmp_path: Path) -> None:
        """Duplicate labels (same normalized value) are deduplicated."""
        tsv = tmp_path.joinpath("test.tsv")
        lines = "CL:0000001\trdfs:label\tneuron\nCL:0000001\tskos:prefLabel\tNeuron\n"
        tsv.write_text(lines, encoding="utf-8")
        index = build_index_from_table(tsv)
        labels = index.term_id_to_labels["CL:0000001"]
        assert len(labels) == 1

    def test_short_rows_skipped(self, tmp_path: Path) -> None:
        """Rows with fewer than 3 columns are skipped."""
        tsv = tmp_path.joinpath("test.tsv")
        lines = "CL:0000001\trdfs:label\n"  # only 2 columns
        tsv.write_text(lines, encoding="utf-8")
        index = build_index_from_table(tsv)
        assert index.value_to_annotations == {}
        assert index.term_id_to_labels == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file produces empty index."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert index.value_to_annotations == {}
        assert index.term_id_to_labels == {}

    def test_value_to_annotations_key_is_normalized(self, tmp_path: Path) -> None:
        """value_to_annotations uses normalized key."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CL:0000001\trdfs:label\tNeuron\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        assert "neuron" in index.value_to_annotations
        assert "Neuron" not in index.value_to_annotations


# === search_terms ===


class TestSearchTerms:
    """Test cases for search_terms function."""

    @pytest.fixture
    def simple_index(self) -> OntologyIndex:
        """Create a simple OntologyIndex for testing."""
        ann = TermAnnotation(
            term_uri="http://example.org/CL_0000001",
            term_id="CL:0000001",
            prop_uri="http://www.w3.org/2000/01/rdf-schema#label",
            value="neuron",
        )

        return OntologyIndex(
            term_id_to_labels={"CL:0000001": ["neuron"]},
            value_to_annotations={"neuron": [ann]},
        )

    def test_exact_match(self, simple_index: OntologyIndex) -> None:
        """Exact match returns the expected result."""
        results = search_terms(simple_index, ["neuron"])
        assert "neuron" in results
        assert len(results["neuron"]) == 1
        assert results["neuron"][0].term_id == "CL:0000001"
        assert results["neuron"][0].exact_match is True

    def test_case_insensitive_match(self, simple_index: OntologyIndex) -> None:
        """Case-insensitive match works."""
        results = search_terms(simple_index, ["Neuron"])
        assert "Neuron" in results
        assert results["Neuron"][0].term_id == "CL:0000001"

    def test_label_populated(self, simple_index: OntologyIndex) -> None:
        """Label field is populated from term_id_to_labels."""
        results = search_terms(simple_index, ["neuron"])
        assert results["neuron"][0].label == "neuron"

    def test_no_match(self, simple_index: OntologyIndex) -> None:
        """Non-matching query returns nothing."""
        results = search_terms(simple_index, ["nonexistent"])
        assert "nonexistent" not in results

    def test_empty_queries(self, simple_index: OntologyIndex) -> None:
        """Empty queries list returns empty dict."""
        results = search_terms(simple_index, [])
        assert results == {}

    def test_empty_index(self) -> None:
        """Empty index returns no results."""
        index = OntologyIndex()
        results = search_terms(index, ["neuron"])
        assert "neuron" not in results

    def test_multiple_queries(self, simple_index: OntologyIndex) -> None:
        """Multiple queries return results for matching queries."""
        results = search_terms(simple_index, ["neuron", "astrocyte"])
        assert "neuron" in results
        assert "astrocyte" not in results

    def test_deduplication(self) -> None:
        """Duplicate annotations for the same term are deduplicated."""
        ann = TermAnnotation(
            term_uri="http://example.org/CL_0000001",
            term_id="CL:0000001",
            prop_uri="http://www.w3.org/2000/01/rdf-schema#label",
            value="neuron",
        )
        index = OntologyIndex(
            term_id_to_labels={"CL:0000001": ["neuron"]},
            value_to_annotations={"neuron": [ann, ann]},
        )
        results = search_terms(index, ["neuron"])
        assert len(results["neuron"]) == 1

    def test_label_missing_returns_none(self) -> None:
        """When term_id_to_labels has no entry, label is None."""
        ann = TermAnnotation(
            term_uri="http://example.org/CL_0000001",
            term_id="CL:0000001",
            prop_uri="http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            value="nerve cell",
        )
        index = OntologyIndex(
            term_id_to_labels={},
            value_to_annotations={"nerve cell": [ann]},
        )
        results = search_terms(index, ["nerve cell"])
        assert results["nerve cell"][0].label is None

    def test_exact_match_flag_false_for_partial(self) -> None:
        """exact_match is False when query differs from annotation value."""
        ann = TermAnnotation(
            term_uri="http://example.org/CL_0000001",
            term_id="CL:0000001",
            prop_uri="http://www.w3.org/2000/01/rdf-schema#label",
            value="motor neuron",
        )
        index = OntologyIndex(
            term_id_to_labels={"CL:0000001": ["motor neuron"]},
            value_to_annotations={"motor neuron": [ann], "motor": [ann]},
        )
        # Query "motor" matches via build_word_combinations
        results = search_terms(index, ["motor"])
        if "motor" in results:
            # The annotation value is "motor neuron" but query is "motor"
            for r in results["motor"]:
                if r.value == "motor neuron":
                    assert r.exact_match is False


# === Search integration: TSV -> index -> search ===


class TestSearchTermsIntegration:
    """Integration tests: TSV -> build_index_from_table -> search_terms."""

    def test_tsv_to_search_with_underscore_id(self, tmp_path: Path) -> None:
        """Bug #1 integration: search finds label for underscore-format term_id."""
        tsv = tmp_path.joinpath("test.tsv")
        tsv.write_text("CVCL_0384\trdfs:label\tHeLa\n", encoding="utf-8")
        index = build_index_from_table(tsv)
        results = search_terms(index, ["HeLa"])
        assert "HeLa" in results
        assert results["HeLa"][0].term_id == "CVCL:0384"
        assert results["HeLa"][0].label == "HeLa"

    def test_tsv_synonym_search(self, tmp_path: Path) -> None:
        """Synonym search returns result with label from rdfs:label."""
        tsv = tmp_path.joinpath("test.tsv")
        lines = "CL:0000001\trdfs:label\tneuron\nCL:0000001\toboInOwl:hasExactSynonym\tnerve cell\n"
        tsv.write_text(lines, encoding="utf-8")
        index = build_index_from_table(tsv)
        results = search_terms(index, ["nerve cell"])
        assert "nerve cell" in results
        assert results["nerve cell"][0].label == "neuron"

    def test_tsv_multiple_terms(self, tmp_path: Path) -> None:
        """Multiple terms in TSV all searchable."""
        tsv = tmp_path.joinpath("test.tsv")
        lines = "CL:0000001\trdfs:label\tneuron\nCL:0000002\trdfs:label\tastrocyte\n"
        tsv.write_text(lines, encoding="utf-8")
        index = build_index_from_table(tsv)
        results = search_terms(index, ["neuron", "astrocyte"])
        assert "neuron" in results
        assert "astrocyte" in results


# === Pydantic models ===


class TestTermAnnotation:
    """Test cases for TermAnnotation model."""

    def test_required_fields(self) -> None:
        """TermAnnotation requires term_uri, term_id, value."""
        ann = TermAnnotation(term_uri="http://example.org/t1", term_id="T:1", value="label")
        assert ann.term_uri == "http://example.org/t1"
        assert ann.term_id == "T:1"
        assert ann.value == "label"
        assert ann.prop_uri is None

    def test_with_prop_uri(self) -> None:
        """TermAnnotation with prop_uri set."""
        ann = TermAnnotation(
            term_uri="http://example.org/t1",
            term_id="T:1",
            prop_uri="http://www.w3.org/2000/01/rdf-schema#label",
            value="label",
        )
        assert ann.prop_uri == "http://www.w3.org/2000/01/rdf-schema#label"

    def test_missing_required_field_raises(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            TermAnnotation(term_uri="http://example.org/t1", term_id="T:1")  # type: ignore[call-arg]


class TestOntologyIndex:
    """Test cases for OntologyIndex model."""

    def test_defaults_empty(self) -> None:
        """OntologyIndex defaults to empty dicts."""
        index = OntologyIndex()
        assert index.term_id_to_labels == {}
        assert index.value_to_annotations == {}

    def test_with_data(self) -> None:
        """OntologyIndex with data is constructed correctly."""
        ann = TermAnnotation(term_uri="u", term_id="T:1", value="v")
        index = OntologyIndex(
            term_id_to_labels={"T:1": ["v"]},
            value_to_annotations={"v": [ann]},
        )
        assert index.term_id_to_labels["T:1"] == ["v"]


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_inherits_term_annotation(self) -> None:
        """SearchResult inherits TermAnnotation fields."""
        sr = SearchResult(
            term_uri="u",
            term_id="T:1",
            value="v",
            exact_match=True,
        )
        assert sr.term_uri == "u"
        assert sr.term_id == "T:1"
        assert sr.exact_match is True
        assert sr.label is None
        assert sr.text2term_score is None
        assert sr.reasoning is None


# === build_index_from_owl ===

TEST_OWL_FILE = Path(__file__).resolve().parent.parent / "data" / "test.owl"


class TestBuildIndexFromOwl:
    """Test cases for build_index_from_owl using tests/data/test.owl."""

    def test_rdfs_labels_indexed(self) -> None:
        """All rdfs:label values are indexed in term_id_to_labels."""
        index = build_index_from_owl(TEST_OWL_FILE)
        assert "TEST:0001" in index.term_id_to_labels
        assert "Alpha Cell" in index.term_id_to_labels["TEST:0001"]
        assert "TEST:0002" in index.term_id_to_labels
        assert "Beta Cell" in index.term_id_to_labels["TEST:0002"]
        assert "TEST:0003" in index.term_id_to_labels
        assert "Gamma Cell" in index.term_id_to_labels["TEST:0003"]

    def test_skos_pref_label_indexed_as_label(self) -> None:
        """skos:prefLabel is indexed in term_id_to_labels."""
        index = build_index_from_owl(TEST_OWL_FILE)
        labels = index.term_id_to_labels["TEST:0002"]
        assert "Beta" in labels

    def test_synonym_in_value_to_annotations(self) -> None:
        """oboInOwl:hasExactSynonym is indexed in value_to_annotations."""
        index = build_index_from_owl(TEST_OWL_FILE)
        assert "alpha" in index.value_to_annotations
        anns = index.value_to_annotations["alpha"]
        assert any(a.term_id == "TEST:0001" for a in anns)

    def test_synonym_not_in_labels(self) -> None:
        """oboInOwl:hasExactSynonym is NOT in term_id_to_labels."""
        index = build_index_from_owl(TEST_OWL_FILE)
        labels = index.term_id_to_labels.get("TEST:0001", [])
        assert "alpha" not in labels

    def test_additional_conditions_filter(self) -> None:
        """additional_conditions filters classes that lack the property value."""
        index = build_index_from_owl(TEST_OWL_FILE, additional_conditions={"hasDbXref": "XREF:001"})
        # Only TEST_0003 has hasDbXref="XREF:001"
        assert "TEST:0003" in index.term_id_to_labels
        assert "TEST:0001" not in index.term_id_to_labels
        assert "TEST:0002" not in index.term_id_to_labels

    def test_search_after_owl_index(self) -> None:
        """Integration: build_index_from_owl + search_terms finds label."""
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms(index, ["Alpha Cell"])
        assert "Alpha Cell" in results
        assert results["Alpha Cell"][0].term_id == "TEST:0001"

    def test_comments_collected_in_term_id_to_comments(self) -> None:
        """rdfs:comment values are stored in term_id_to_comments."""
        index = build_index_from_owl(TEST_OWL_FILE)
        assert "TEST:0001" in index.term_id_to_comments
        assert index.term_id_to_comments["TEST:0001"] == ["Disease: Alpha disease"]

    def test_multiple_comments_collected(self) -> None:
        """Multiple rdfs:comment values on a single term are all collected."""
        index = build_index_from_owl(TEST_OWL_FILE)
        comments = index.term_id_to_comments["TEST:0003"]
        assert "Disease: Gamma disease" in comments
        assert "derived_from: Delta Cell" in comments
        assert len(comments) == 2

    def test_no_comment_term_absent_from_comments(self) -> None:
        """Terms without rdfs:comment are absent from term_id_to_comments."""
        index = build_index_from_owl(TEST_OWL_FILE)
        assert "TEST:0002" not in index.term_id_to_comments

    def test_comments_not_in_value_to_annotations(self) -> None:
        """rdfs:comment values are NOT indexed in value_to_annotations."""
        index = build_index_from_owl(TEST_OWL_FILE)
        assert "disease: alpha disease" not in index.value_to_annotations
        assert "disease: gamma disease" not in index.value_to_annotations
        assert "derived_from: delta cell" not in index.value_to_annotations

    def test_comments_filtered_by_additional_conditions(self) -> None:
        """additional_conditions also filters term_id_to_comments."""
        index = build_index_from_owl(TEST_OWL_FILE, additional_conditions={"hasDbXref": "XREF:001"})
        assert "TEST:0003" in index.term_id_to_comments
        assert "TEST:0001" not in index.term_id_to_comments

    def test_search_results_include_comments(self) -> None:
        """SearchResult.comments is populated from term_id_to_comments."""
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms(index, ["Alpha Cell"])
        assert results["Alpha Cell"][0].comments == ["Disease: Alpha disease"]

    def test_search_results_no_comments_is_none(self) -> None:
        """SearchResult.comments is None when term has no comments."""
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms(index, ["Beta Cell"])
        assert results["Beta Cell"][0].comments is None


# === build_index_from_file ===


class TestBuildIndexFromFile:
    """Test cases for build_index_from_file dispatch function."""

    def test_dispatches_owl(self) -> None:
        """OWL file is dispatched to build_index_from_owl."""
        index = build_index_from_file(TEST_OWL_FILE)
        assert "TEST:0001" in index.term_id_to_labels

    def test_dispatches_tsv(self, tmp_path: Path) -> None:
        """TSV file is dispatched to build_index_from_table."""
        tsv = tmp_path / "test.tsv"
        tsv.write_text("CL:0000001\trdfs:label\tneuron\n", encoding="utf-8")
        index = build_index_from_file(tsv)
        assert "neuron" in index.value_to_annotations

    def test_dispatches_csv(self, tmp_path: Path) -> None:
        """CSV file is dispatched to build_index_from_table."""
        csv_f = tmp_path / "test.csv"
        csv_f.write_text("CL:0000001,rdfs:label,neuron\n", encoding="utf-8")
        index = build_index_from_file(csv_f)
        assert "neuron" in index.value_to_annotations

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported extension raises ValueError."""
        txt = tmp_path / "test.txt"
        txt.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            build_index_from_file(txt)

    def test_owl_with_filter(self) -> None:
        """OWL dispatch passes ontology_filter as additional_conditions."""
        index = build_index_from_file(TEST_OWL_FILE, ontology_filter={"hasDbXref": "XREF:001"})
        assert "TEST:0003" in index.term_id_to_labels
        assert "TEST:0001" not in index.term_id_to_labels


# === search_terms_with_text2term ===


class TestSearchTermsWithText2term:
    """Test cases for search_terms_with_text2term (text2term mocked)."""

    def _make_text2term_df(self) -> pd.DataFrame:
        """Create a fake text2term output DataFrame."""
        return pd.DataFrame(
            {
                "Source Term": ["Alpha Cell", "Alpha Cell"],
                "Mapped Term Label": ["Alpha Cell", "Gamma Cell"],
                "Mapped Term IRI": [
                    "http://example.org/test-ontology#TEST_0001",
                    "http://example.org/test-ontology#TEST_0003",
                ],
                "Mapped Term CURIE": ["TEST:0001", "TEST:0003"],
                "Mapping Score": [1.0, 0.8],
            }
        )

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_basic_flow(self, mock_map_terms: MagicMock) -> None:
        """Basic flow: text2term returns mapped terms, results are built."""
        mock_map_terms.return_value = self._make_text2term_df()
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms_with_text2term(["Alpha Cell"], TEST_OWL_FILE, index=index)
        assert "Alpha Cell" in results
        term_ids = {r.term_id for r in results["Alpha Cell"]}
        assert "TEST:0001" in term_ids

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_exact_match_flag(self, mock_map_terms: MagicMock) -> None:
        """exact_match is True when source and mapped term match."""
        mock_map_terms.return_value = self._make_text2term_df()
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms_with_text2term(["Alpha Cell"], TEST_OWL_FILE, index=index)
        alpha_results = [r for r in results["Alpha Cell"] if r.term_id == "TEST:0001"]
        assert alpha_results
        assert alpha_results[0].exact_match is True

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_text2term_score_preserved(self, mock_map_terms: MagicMock) -> None:
        """text2term_score is preserved in SearchResult."""
        mock_map_terms.return_value = self._make_text2term_df()
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms_with_text2term(["Alpha Cell"], TEST_OWL_FILE, index=index)
        scores = {r.term_id: r.text2term_score for r in results["Alpha Cell"]}
        assert scores.get("TEST:0001") == pytest.approx(1.0)

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_missing_columns_raises(self, mock_map_terms: MagicMock) -> None:
        """Missing required columns in text2term output raises ValueError."""
        mock_map_terms.return_value = pd.DataFrame({"Source Term": ["A"], "Mapped Term Label": ["B"]})
        with pytest.raises(ValueError, match="Expected columns missing"):
            search_terms_with_text2term(["A"], TEST_OWL_FILE)

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_deduplicates_by_term_id_and_value(self, mock_map_terms: MagicMock) -> None:
        """Duplicate (term_id, value) pairs from text2term are deduplicated."""
        df = pd.DataFrame(
            {
                "Source Term": ["Alpha Cell", "Alpha Cell"],
                "Mapped Term Label": ["Alpha Cell", "Alpha Cell"],
                "Mapped Term IRI": [
                    "http://example.org/test-ontology#TEST_0001",
                    "http://example.org/test-ontology#TEST_0001",
                ],
                "Mapped Term CURIE": ["TEST:0001", "TEST:0001"],
                "Mapping Score": [1.0, 0.9],
            }
        )
        mock_map_terms.return_value = df
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms_with_text2term(["Alpha Cell"], TEST_OWL_FILE, index=index)
        alpha_results = [r for r in results["Alpha Cell"] if r.term_id == "TEST:0001"]
        assert len(alpha_results) == 1

    @patch("bsllmner2.ontology_search.text2term.map_terms")
    def test_label_from_index(self, mock_map_terms: MagicMock) -> None:
        """Label is populated from the OntologyIndex, not text2term output."""
        mock_map_terms.return_value = self._make_text2term_df()
        index = build_index_from_owl(TEST_OWL_FILE)
        results = search_terms_with_text2term(["Alpha Cell"], TEST_OWL_FILE, index=index)
        alpha_results = [r for r in results["Alpha Cell"] if r.term_id == "TEST:0001"]
        assert alpha_results
        assert alpha_results[0].label == "Alpha Cell"
