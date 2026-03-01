"""Custom exception classes for bsllmner2."""


class Bsllmner2Error(Exception):
    """Base exception class for bsllmner2."""


class OllamaConnectionError(Bsllmner2Error):
    """Raised when connection to Ollama server fails."""

    def __init__(self, host: str, original_error: Exception | None = None):
        self.host = host
        self.original_error = original_error
        message = (
            f"Failed to connect to Ollama server at {host}\n"
            "Please ensure:\n"
            "  1. Ollama server is running\n"
            "  2. The model is available (run: ollama pull <model-name>)\n"
            "  3. Host URL is correct (--ollama-host <url>)"
        )
        if original_error is not None:
            message += f"\n\nOriginal error: {original_error}"
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[str, Exception | None]]:
        return (type(self), (self.host, self.original_error))


class OllamaProcessingError(Bsllmner2Error):
    """Raised when Ollama fails to process an entry."""

    def __init__(self, accession: str, original_error: Exception):
        self.accession = accession
        self.original_error = original_error
        message = f"Error processing entry {accession}: {original_error}"
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[str, Exception]]:
        return (type(self), (self.accession, self.original_error))


class ConfigurationError(Bsllmner2Error):
    """Raised when there's a configuration problem."""


class ResumeDataError(Bsllmner2Error):
    """Raised when resume data is corrupted or inconsistent."""

    def __init__(self, run_name: str, message: str):
        self.run_name = run_name
        self._detail = message
        full_message = f"Resume data error for run '{run_name}': {message}"
        super().__init__(full_message)

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return (type(self), (self.run_name, self._detail))
