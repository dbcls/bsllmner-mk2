"""Tests for configuration module."""

import os
from collections.abc import Generator

import pytest

from bsllmner2.config import Config, default_config, get_config


class TestConfig:
    """Test cases for Config class."""

    def test_default_values(self) -> None:
        """Test that default Config has expected values."""
        config = Config()
        assert config.ollama_host == "http://localhost:11434"
        assert config.debug is False
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8000
        assert config.api_url_prefix == ""

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
        assert config.api_host == default_config.api_host
        assert config.api_port == default_config.api_port
        assert config.api_url_prefix == default_config.api_url_prefix

    def test_ollama_host_from_env(self) -> None:
        """Test OLLAMA_HOST env var is respected."""
        os.environ["OLLAMA_HOST"] = "http://custom-host:11434"
        config = get_config()
        assert config.ollama_host == "http://custom-host:11434"

    def test_api_host_from_env(self) -> None:
        """Test BSLLMNER2_API_HOST env var is respected."""
        os.environ["BSLLMNER2_API_HOST"] = "0.0.0.0"
        config = get_config()
        assert config.api_host == "0.0.0.0"

    def test_api_port_from_env(self) -> None:
        """Test BSLLMNER2_API_PORT env var is respected."""
        os.environ["BSLLMNER2_API_PORT"] = "9000"
        config = get_config()
        assert config.api_port == 9000

    def test_api_url_prefix_from_env(self) -> None:
        """Test BSLLMNER2_API_URL_PREFIX env var is respected."""
        os.environ["BSLLMNER2_API_URL_PREFIX"] = "/api/v1"
        config = get_config()
        assert config.api_url_prefix == "/api/v1"

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
