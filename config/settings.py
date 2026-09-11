"""
Centralized application configuration.

All external model choices, endpoints, and tunable thresholds live here.
Swapping the NVIDIA model, the Ollama models, or the similarity threshold
should never require touching pipeline code -- only this file (or the
corresponding environment variables).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- NVIDIA AI Endpoint (structured resume extraction) ---
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    nvidia_model: str = Field(
        default="meta/llama-3.1-70b-instruct",
        alias="NVIDIA_MODEL",
    )
    nvidia_request_timeout_seconds: float = Field(default=60.0, alias="NVIDIA_TIMEOUT")
    nvidia_max_retries: int = Field(default=3, alias="NVIDIA_MAX_RETRIES")

    # --- Ollama (local embeddings + generation) ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    ollama_generation_model: str = Field(default="llama3.1:8b-instruct", alias="OLLAMA_GEN_MODEL")
    ollama_request_timeout_seconds: float = Field(default=60.0, alias="OLLAMA_TIMEOUT")
    ollama_generation_temperature: float = Field(default=0.3, alias="OLLAMA_TEMPERATURE")

    # --- Matching / scoring ---
    low_score_threshold: float = Field(
        default=0.55,
        alias="LOW_SCORE_THRESHOLD",
        description="Cosine similarity below this flags a resume bullet as low-fit.",
    )
    embedding_dimension: int = Field(
        default=768,
        alias="EMBEDDING_DIMENSION",
        description="Expected vector width for nomic-embed-text; used to validate responses.",
    )

    # --- Text preprocessing / chunking ---
    max_chunk_chars: int = Field(default=1500, alias="MAX_CHUNK_CHARS")
    chunk_overlap_chars: int = Field(default=150, alias="CHUNK_OVERLAP_CHARS")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()
