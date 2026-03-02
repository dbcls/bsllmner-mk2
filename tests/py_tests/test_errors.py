"""Tests for custom exception classes.

Focuses on bug-finding: serialization roundtrip, exception hierarchy,
edge cases, and exception chaining.
"""

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.errors import (
    Bsllmner2Error,
    ConfigurationError,
    OllamaConnectionError,
    OllamaProcessingError,
    ResumeDataError,
)


class TestBsllmner2Error:
    """Base exception contract tests."""

    def test_inherits_from_exception(self) -> None:
        """Subclass of Exception so except Exception catches it."""
        assert issubclass(Bsllmner2Error, Exception)

    def test_message_stores_in_args(self) -> None:
        """Args tuple must be populated for str(e) and logging."""
        e = Bsllmner2Error("test message")
        assert e.args == ("test message",)
        assert str(e) == "test message"

    def test_catchable_as_bsllmner2error(self) -> None:
        """CLI entry points use except Bsllmner2Error as handler."""
        with pytest.raises(Bsllmner2Error):
            raise Bsllmner2Error("test")


class TestOllamaConnectionError:
    """Tests for OllamaConnectionError."""

    def test_inherits_from_bsllmner2error(self) -> None:
        """Subclass of Bsllmner2Error so CLI catch block works."""
        assert issubclass(OllamaConnectionError, Bsllmner2Error)

    def test_host_attribute_stored(self) -> None:
        """Callers inspect e.host for retry logic or error reporting."""
        e = OllamaConnectionError("http://localhost:11434")
        assert e.host == "http://localhost:11434"

    def test_original_error_default_none(self) -> None:
        """Construction without original_error must not fail."""
        e = OllamaConnectionError("host")
        assert e.original_error is None

    def test_original_error_stored(self) -> None:
        """Identity of original_error must be preserved for chaining."""
        orig = ConnectionError("refused")
        e = OllamaConnectionError("host", orig)
        assert e.original_error is orig

    def test_message_contains_host(self) -> None:
        """User must see which host failed."""
        e = OllamaConnectionError("http://myhost:11434")
        assert "http://myhost:11434" in str(e)

    def test_message_contains_original_error_when_provided(self) -> None:
        """Original error detail must appear in message for debugging."""
        orig = ConnectionError("Connection refused")
        e = OllamaConnectionError("host", orig)
        assert "Connection refused" in str(e)
        assert "Original error" in str(e)

    def test_message_excludes_original_error_when_none(self) -> None:
        """No 'None' or 'Original error' in message when not provided."""
        e = OllamaConnectionError("host")
        assert "Original error" not in str(e)
        assert "None" not in str(e)

    def test_deepcopy_preserves_host(self) -> None:
        """Without __reduce__, deepcopy sets host to full message."""
        e = OllamaConnectionError("http://host:1234")
        e2 = copy.deepcopy(e)
        assert e2.host == "http://host:1234"
        assert str(e2) == str(e)

    def test_deepcopy_preserves_original_error(self) -> None:
        """Without __reduce__, deepcopy loses original_error."""
        orig = ConnectionError("refused")
        e = OllamaConnectionError("host", orig)
        e2 = copy.deepcopy(e)
        assert e2.original_error is not None
        assert str(e2.original_error) == "refused"
        assert str(e2) == str(e)

    def test_deepcopy_without_original_error(self) -> None:
        """Deepcopy with original_error=None must not crash."""
        e = OllamaConnectionError("host")
        e2 = copy.deepcopy(e)
        assert e2.host == "host"
        assert e2.original_error is None
        assert str(e2) == str(e)

    def test_original_error_truthiness_with_falsy_exception(self) -> None:
        """Truthiness check silently ignores falsy exceptions."""

        class FalsyError(Exception):
            def __bool__(self) -> bool:
                return False

        orig = FalsyError("something went wrong")
        e = OllamaConnectionError("host", orig)
        assert "Original error" in str(e)
        assert "something went wrong" in str(e)


class TestOllamaProcessingError:
    """Tests for OllamaProcessingError."""

    def test_inherits_from_bsllmner2error(self) -> None:
        """Subclass of Bsllmner2Error so CLI catch block works."""
        assert issubclass(OllamaProcessingError, Bsllmner2Error)

    def test_accession_attribute_stored(self) -> None:
        """Callers inspect e.accession for skip/retry logic."""
        e = OllamaProcessingError("SAMN001", RuntimeError("fail"))
        assert e.accession == "SAMN001"

    def test_original_error_stored(self) -> None:
        """Identity of original_error must be preserved."""
        orig = RuntimeError("fail")
        e = OllamaProcessingError("SAMN001", orig)
        assert e.original_error is orig

    def test_message_contains_accession_and_error(self) -> None:
        """Message must include both accession and original error."""
        e = OllamaProcessingError("SAMN001", RuntimeError("timeout"))
        msg = str(e)
        assert "SAMN001" in msg
        assert "timeout" in msg

    def test_deepcopy_roundtrip(self) -> None:
        """Without __reduce__, deepcopy crashes with TypeError."""
        e = OllamaProcessingError("SAMN001", RuntimeError("fail"))
        e2 = copy.deepcopy(e)
        assert e2.accession == "SAMN001"
        assert str(e2.original_error) == "fail"
        assert str(e2) == str(e)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inherits_from_bsllmner2error(self) -> None:
        """Subclass of Bsllmner2Error so CLI catch block works."""
        assert issubclass(ConfigurationError, Bsllmner2Error)

    def test_message_passthrough(self) -> None:
        """Standard Exception(msg) must work (no custom __init__)."""
        e = ConfigurationError("bad config")
        assert str(e) == "bad config"
        assert e.args == ("bad config",)

    def test_deepcopy_roundtrip(self) -> None:
        """Positive control: no custom __init__ so deepcopy should work."""
        e = ConfigurationError("bad config")
        e2 = copy.deepcopy(e)
        assert str(e2) == "bad config"


class TestResumeDataError:
    """Tests for ResumeDataError."""

    def test_inherits_from_bsllmner2error(self) -> None:
        """Subclass of Bsllmner2Error so CLI catch block works."""
        assert issubclass(ResumeDataError, Bsllmner2Error)

    def test_run_name_attribute_stored(self) -> None:
        """Callers inspect e.run_name for identifying the broken run."""
        e = ResumeDataError("my-run", "corrupted data")
        assert e.run_name == "my-run"

    def test_message_contains_run_name_and_detail(self) -> None:
        """Message must include both run_name and detail."""
        e = ResumeDataError("my-run", "data mismatch")
        msg = str(e)
        assert "my-run" in msg
        assert "data mismatch" in msg

    def test_message_starts_with_prefix(self) -> None:
        """Consistent prefix for log parsing."""
        e = ResumeDataError("run-1", "bad data")
        assert str(e).startswith("Resume data error for run")

    def test_deepcopy_roundtrip(self) -> None:
        """Without __reduce__, deepcopy crashes with TypeError."""
        e = ResumeDataError("run-1", "corrupted")
        e2 = copy.deepcopy(e)
        assert e2.run_name == "run-1"
        assert "corrupted" in str(e2)
        assert str(e2) == str(e)


class TestExceptionHierarchyCatching:
    """Tests that verify catch patterns used in CLI entry points."""

    @pytest.mark.parametrize(
        ("exc_class", "args"),
        [
            pytest.param(
                OllamaConnectionError,
                ("host",),
                id="OllamaConnectionError",
            ),
            pytest.param(
                OllamaProcessingError,
                ("SAMN001", RuntimeError("x")),
                id="OllamaProcessingError",
            ),
            pytest.param(
                ConfigurationError,
                ("msg",),
                id="ConfigurationError",
            ),
            pytest.param(
                ResumeDataError,
                ("run", "msg"),
                id="ResumeDataError",
            ),
        ],
    )
    def test_all_custom_errors_caught_by_base(self, exc_class: type, args: tuple[object, ...]) -> None:
        """All custom exceptions must be catchable as Bsllmner2Error."""
        with pytest.raises(Bsllmner2Error):
            raise exc_class(*args)

    def test_subclass_does_not_catch_base(self) -> None:
        """Catching OllamaConnectionError must not catch Bsllmner2Error."""
        caught_by_subclass = False
        try:
            raise Bsllmner2Error("generic error")
        except OllamaConnectionError:
            caught_by_subclass = True
        except Bsllmner2Error:
            pass
        assert not caught_by_subclass


class TestExceptionChaining:
    """Tests for raise ... from e compatibility."""

    def test_raise_from_preserves_cause(self) -> None:
        """Raise from e sets __cause__ for traceback display."""
        orig = ConnectionError("refused")
        with pytest.raises(OllamaConnectionError) as exc_info:
            raise OllamaConnectionError("host", orig) from orig
        assert exc_info.value.__cause__ is orig

    def test_cause_accessible_through_base_catch(self) -> None:
        """Cause must be accessible when caught as Bsllmner2Error."""
        orig = ConnectionError("refused")
        with pytest.raises(Bsllmner2Error) as exc_info:
            raise OllamaConnectionError("host", orig) from orig
        assert exc_info.value.__cause__ is orig


class TestEdgeCases:
    """Edge case inputs that might break formatting or attribute storage."""

    def test_empty_host(self) -> None:
        """Empty string host must not crash."""
        e = OllamaConnectionError("")
        assert e.host == ""
        assert "at " in str(e)

    def test_empty_accession(self) -> None:
        """Empty string accession must not crash."""
        e = OllamaProcessingError("", RuntimeError("x"))
        assert e.accession == ""

    def test_empty_run_name(self) -> None:
        """Empty string run_name must not crash."""
        e = ResumeDataError("", "msg")
        assert e.run_name == ""

    def test_host_with_special_chars(self) -> None:
        """Unicode and newlines in host must not crash formatting."""
        host = "http://\u4f8b\u3048.jp:11434\ninjected"
        e = OllamaConnectionError(host)
        assert e.host == host

    def test_message_with_curly_braces(self) -> None:
        """Curly braces must not be interpreted as format placeholders."""
        e = ResumeDataError("run", "{key}")
        assert "{key}" in str(e)

    def test_original_error_with_unicode(self) -> None:
        """Unicode in original error message must not crash formatting."""
        orig = RuntimeError("\u63a5\u7d9a\u62d2\u5426")
        e = OllamaConnectionError("host", orig)
        assert "\u63a5\u7d9a\u62d2\u5426" in str(e)


class TestOllamaConnectionErrorProperty:
    """Property-based tests for OllamaConnectionError."""

    @given(host=st.text(max_size=200))
    @settings(max_examples=200)
    def test_host_always_in_message(self, host: str) -> None:
        """Arbitrary host string always appears in str(e)."""
        e = OllamaConnectionError(host)
        assert host in str(e)

    @given(host=st.text(max_size=200))
    @settings(max_examples=200)
    def test_host_attribute_preserved(self, host: str) -> None:
        """e.host always matches the input."""
        e = OllamaConnectionError(host)
        assert e.host == host


class TestOllamaProcessingErrorProperty:
    """Property-based tests for OllamaProcessingError."""

    @given(accession=st.text(max_size=200))
    @settings(max_examples=200)
    def test_accession_always_in_message(self, accession: str) -> None:
        """Arbitrary accession string always appears in str(e)."""
        e = OllamaProcessingError(accession, RuntimeError("fail"))
        assert accession in str(e)

    @given(accession=st.text(max_size=200))
    @settings(max_examples=200)
    def test_accession_attribute_preserved(self, accession: str) -> None:
        """e.accession always matches the input."""
        e = OllamaProcessingError(accession, RuntimeError("fail"))
        assert e.accession == accession


class TestResumeDataErrorProperty:
    """Property-based tests for ResumeDataError."""

    @given(run_name=st.text(max_size=200))
    @settings(max_examples=200)
    def test_run_name_always_in_message(self, run_name: str) -> None:
        """Arbitrary run_name string always appears in str(e)."""
        e = ResumeDataError(run_name, "detail")
        assert run_name in str(e)

    @given(run_name=st.text(max_size=200))
    @settings(max_examples=200)
    def test_run_name_attribute_preserved(self, run_name: str) -> None:
        """e.run_name always matches the input."""
        e = ResumeDataError(run_name, "detail")
        assert e.run_name == run_name


class TestAllCustomErrorsDeepCopyProperty:
    """Property-based tests for deepcopy of all custom exceptions."""

    @given(host=st.text(max_size=200))
    @settings(max_examples=200)
    def test_ollama_connection_error_deepcopy(self, host: str) -> None:
        """OllamaConnectionError survives deepcopy with arbitrary host."""
        e = OllamaConnectionError(host)
        e2 = copy.deepcopy(e)
        assert e2.host == host
        assert str(e2) == str(e)

    @given(accession=st.text(max_size=200))
    @settings(max_examples=200)
    def test_ollama_processing_error_deepcopy(self, accession: str) -> None:
        """OllamaProcessingError survives deepcopy with arbitrary accession."""
        orig = RuntimeError("fail")
        e = OllamaProcessingError(accession, orig)
        e2 = copy.deepcopy(e)
        assert e2.accession == accession
        assert str(e2) == str(e)

    @given(run_name=st.text(max_size=200), message=st.text(max_size=200))
    @settings(max_examples=200)
    def test_resume_data_error_deepcopy(self, run_name: str, message: str) -> None:
        """ResumeDataError survives deepcopy with arbitrary run_name + message."""
        e = ResumeDataError(run_name, message)
        e2 = copy.deepcopy(e)
        assert e2.run_name == run_name
        assert str(e2) == str(e)
