"""
VectorStore: content-addressed embedding cache.

Keyed on a hash of (model_name, text), so re-scoring a resume after editing
one bullet only re-embeds that bullet -- everything else is served from
cache. In-memory by default; swap `_cache` for a persisted backend (e.g.
sqlite, redis) later without touching callers.
"""

from __future__ import annotations

import hashlib

import numpy as np

from config.logging_config import get_logger
from src.core.interfaces import EmbedderProtocol

logger = get_logger(__name__)


class VectorStore:
    """Simple content-addressed in-memory embedding cache."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get_or_compute(self, text: str, embedder: EmbedderProtocol, model_name: str) -> np.ndarray:
        key = self._make_key(model_name, text)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        vector = embedder.embed_text(text)
        self._cache[key] = vector
        return vector

    def get_or_compute_batch(
        self, texts: list[str], embedder: EmbedderProtocol, model_name: str
    ) -> np.ndarray:
        keys = [self._make_key(model_name, t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            computed = embedder.embed_batch(missing_texts)
            for idx, vector in zip(missing_indices, computed):
                self._cache[keys[idx]] = vector
                self._misses += 1
        self._hits += len(texts) - len(missing_indices)

        return np.array([self._cache[k] for k in keys], dtype=np.float32)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "cached_entries": len(self._cache)}

    @staticmethod
    def _make_key(model_name: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model_name}:{digest}"
