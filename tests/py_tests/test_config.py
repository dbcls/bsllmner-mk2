"""Tests for configuration module."""

import os
from collections.abc import Generator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsllmner2.config import Config, _parse_bool_env, default_config, get_config


class TestConfig:
    """Test cases for Config class."""

    def test_default_values(self) -> None:
        """Test that default Config has expected values."""
        config = Config()
        assert config.ollama_host == "http://localhost:11434"
        assert config.debug is False

    def test_default_config_instance(self) -> None:
        """Test that default_config is a valid Config instance."""
        assert isinstance(default_config, Config)
        assert default_config.ollama_host == "http://localhost:11434"


class TestGetConfig:
    """Test cases for get_config function."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, clean_env: Generator[None, None, None]) -> None:
        """Ensure clean environment for each test."""
        return

    def test_default_config_no_env(self) -> None:
        """Test get_config returns defaults when no env vars are set."""
        config = get_config()
        assert config.ollama_host == default_config.ollama_host
        assert config.debug == default_config.debug

    def test_ollama_host_from_env(self) -> None:
        """Test OLLAMA_HOST env var is respected."""
        os.environ["OLLAMA_HOST"] = "http://custom-host:11434"
        config = get_config()
        assert config.ollama_host == "http://custom-host:11434"

    def test_debug_true_from_env(self) -> None:
        """Test BSLLMNER2_DEBUG=true sets debug to True."""
        os.environ["BSLLMNER2_DEBUG"] = "true"
        config = get_config()
        assert config.debug is True

    def test_debug_1_from_env(self) -> None:
        """Test BSLLMNER2_DEBUG=1 sets debug to True."""
        os.environ["BSLLMNER2_DEBUG"] = "1"
        config = get_config()
        assert config.debug is True


class TestOllamaContainerNameEnv:
    """Test cases for OLLAMA_CONTAINER_NAME environment variable."""

    def test_default_value(self) -> None:
        """Default OLLAMA_CONTAINER_NAME is 'bsllmner-mk2-ollama'."""
        import bsllmner2.config as config_mod

        assert os.environ.get("BSLLMNER2_CONTAINER_NAME", "bsllmner-mk2-ollama") == config_mod.OLLAMA_CONTAINER_NAME

    def test_env_var_expression(self) -> None:
        """os.environ.get expression used in config produces correct results."""
        assert os.environ.get("BSLLMNER2_CONTAINER_NAME", "bsllmner-mk2-ollama") == "bsllmner-mk2-ollama"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BSLLMNER2_CONTAINER_NAME env var overrides the default via monkeypatch on module attr."""
        import bsllmner2.config as config_mod

        monkeypatch.setattr(config_mod, "OLLAMA_CONTAINER_NAME", "custom-container")
        assert config_mod.OLLAMA_CONTAINER_NAME == "custom-container"

    def test_empty_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty BSLLMNER2_CONTAINER_NAME is respected via monkeypatch on module attr."""
        import bsllmner2.config as config_mod

        monkeypatch.setattr(config_mod, "OLLAMA_CONTAINER_NAME", "")
        assert config_mod.OLLAMA_CONTAINER_NAME == ""


class TestGetConfigDebugParsing:
    """Test cases for debug flag parsing with various string values."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, clean_env: Generator[None, None, None]) -> None:
        """Ensure clean environment for each test."""
        return

    def test_debug_false_string(self) -> None:
        """BSLLMNER2_DEBUG='false' correctly sets debug=False."""
        os.environ["BSLLMNER2_DEBUG"] = "false"
        config = get_config()
        assert config.debug is False

    def test_debug_0_string(self) -> None:
        """BSLLMNER2_DEBUG='0' correctly sets debug=False."""
        os.environ["BSLLMNER2_DEBUG"] = "0"
        config = get_config()
        assert config.debug is False

    def test_debug_no_string(self) -> None:
        """BSLLMNER2_DEBUG='no' correctly sets debug=False."""
        os.environ["BSLLMNER2_DEBUG"] = "no"
        config = get_config()
        assert config.debug is False

    def test_debug_empty_string(self) -> None:
        """Empty string results in debug=False."""
        os.environ["BSLLMNER2_DEBUG"] = ""
        config = get_config()
        assert config.debug is False

    def test_debug_yes_string(self) -> None:
        """BSLLMNER2_DEBUG='yes' correctly sets debug=True."""
        os.environ["BSLLMNER2_DEBUG"] = "yes"
        config = get_config()
        assert config.debug is True

    def test_debug_on_string(self) -> None:
        """BSLLMNER2_DEBUG='on' correctly sets debug=True."""
        os.environ["BSLLMNER2_DEBUG"] = "on"
        config = get_config()
        assert config.debug is True

    def test_debug_case_insensitive(self) -> None:
        """Debug parsing is case-insensitive."""
        os.environ["BSLLMNER2_DEBUG"] = "TRUE"
        config = get_config()
        assert config.debug is True

        os.environ["BSLLMNER2_DEBUG"] = "False"
        config = get_config()
        assert config.debug is False


# === Property-based tests ===

_TRUTHY_STRINGS = ("true", "1", "yes", "on")


class TestParseBoolEnvPBT:
    """Property-based tests for _parse_bool_env."""

    @given(value=st.text())
    @settings(max_examples=200)
    def test_never_raises(self, value: str) -> None:
        """Any string input never raises an exception."""
        result = _parse_bool_env(value)
        assert isinstance(result, bool)

    @given(value=st.sampled_from(_TRUTHY_STRINGS))
    @settings(max_examples=200)
    def test_truthy_returns_true(self, value: str) -> None:
        """Known truthy strings always return True."""
        assert _parse_bool_env(value) is True

    @given(
        value=st.text(min_size=0, max_size=50).filter(
            lambda s: s.lower() not in _TRUTHY_STRINGS,
        ),
    )
    @settings(max_examples=200)
    def test_non_truthy_returns_false(self, value: str) -> None:
        """Non-truthy strings return False."""
        assert _parse_bool_env(value) is False
