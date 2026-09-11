"""
OllamaEmbedder: wraps calls to a local Ollama server's /api/embeddings route.

Uses `nomic-embed-text` by default (see config/settings.py). Every call
validates the returned vector width against `settings.embedding_dimension`
so a silently-wrong model swap fails loudly instead of corrupting
similarity scores.
"""

from __future__ import annotations

import numpy as np

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.core.exceptions import EmbeddingDimensionMismatchError, OllamaConnectionError

logger = get_logger(__name__)


class OllamaEmbedder:
    """Local embedding client backed by Ollama."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._settings.embedding_dimension), dtype=np.float32)

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise OllamaConnectionError(
                "httpx is not installed. Run `pip install httpx`.",
            ) from exc

        vectors: list[list[float]] = []
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/embeddings"

        try:
            with httpx.Client(timeout=self._settings.ollama_request_timeout_seconds) as client:
                for text in texts:
                    response = client.post(
                        url,
                        json={"model": self._settings.ollama_embed_model, "prompt": text},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    vector = payload.get("embedding")
                    if vector is None:
                        raise OllamaConnectionError(
                            "Ollama embeddings response missing 'embedding' field.",
                            details={"payload_keys": list(payload.keys())},
                        )
                    vectors.append(vector)
        except OllamaConnectionError:
            raise
        except httpx.HTTPStatusError as exc:
            raise OllamaConnectionError(
                f"Ollama returned an error status: {exc.response.status_code}",
                details={"body": exc.response.text[:500]},
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                "Could not reach the local Ollama server. Is it running (`ollama serve`)?",
                details={"error": str(exc)},
            ) from exc

        matrix = np.array(vectors, dtype=np.float32)
        expected_dim = self._settings.embedding_dimension
        if matrix.shape[1] != expected_dim:
            raise EmbeddingDimensionMismatchError(
                f"Expected embedding dimension {expected_dim}, got {matrix.shape[1]}.",
                details={"model": self._settings.ollama_embed_model},
            )

        return matrix
