"""
Custom exception hierarchy.

Every external call (NVIDIA, Ollama, file I/O) must be wrapped so callers
never have to catch raw `httpx`/`requests`/`OSError` exceptions. This keeps
error handling at the API layer simple and consistent.
"""

from __future__ import annotations

from typing import Any


class AgentBaseError(Exception):
    """Base class for all application-specific exceptions."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class ValidationError(AgentBaseError):
    """Raised when input to a pipeline fails schema / sanity validation."""


class ResumeParseError(AgentBaseError):
    """Raised when raw resume text cannot be extracted or structured."""


class DocumentExtractionError(ResumeParseError):
    """Raised when the PDF/DOCX -> raw text step fails."""


class NvidiaAPIError(AgentBaseError):
    """Raised when the NVIDIA AI endpoint call fails or returns malformed output."""


class OllamaConnectionError(AgentBaseError):
    """Raised when the local Ollama server is unreachable or errors out."""


class EmbeddingDimensionMismatchError(AgentBaseError):
    """Raised when an embedding response doesn't match the expected vector width."""


class JobIngestionError(AgentBaseError):
    """Raised when a scraped job listing cannot be normalized into JobListing."""


class OptimizationError(AgentBaseError):
    """Raised when the LangChain optimization agent fails to produce a valid rewrite."""
