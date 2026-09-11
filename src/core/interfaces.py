"""
Shared abstract interfaces (Protocols) implemented by concrete components.

Keeping these as Protocols (structural typing) rather than ABCs means the
embedding/generation backends can be swapped (e.g. Ollama -> a hosted API)
without changing any type that depends on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Anything that can turn text into a fixed-width vector."""

    def embed_text(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class StructuredExtractorProtocol(Protocol):
    """Anything that turns raw resume text into a structured dict/JSON."""

    def structure(self, raw_text: str) -> dict:
        ...


class AbstractDocumentParser(ABC):
    """Base interface for PDF/DOCX -> raw text extractors."""

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Return the raw, uncleaned text content of the document."""

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        """File extensions (lowercase, with dot) this parser can handle."""
