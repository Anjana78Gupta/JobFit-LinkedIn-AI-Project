"""
Shared dependency providers (FastAPI Depends), each cached for the life of
the process. Instantiating an OllamaEmbedder or NvidiaResumeStructurer per
request would be wasteful and would defeat the VectorStore cache -- so
these are process-wide singletons wired through `lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache

from config.settings import Settings, get_settings
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.embeddings.vector_store import VectorStore
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.optimization.agent_executor import OptimizationPipeline
from src.optimization.optimizer_chain import OptimizerChain
from src.parsing.nvidia_structurer import NvidiaResumeStructurer
from src.parsing.parser_pipeline import ParserPipeline


@lru_cache
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_ollama_embedder() -> OllamaEmbedder:
    return OllamaEmbedder(settings=get_app_settings())


@lru_cache
def get_nvidia_structurer() -> NvidiaResumeStructurer:
    return NvidiaResumeStructurer(settings=get_app_settings())


@lru_cache
def get_optimizer_chain() -> OptimizerChain:
    return OptimizerChain(settings=get_app_settings())


@lru_cache
def get_parser_pipeline() -> ParserPipeline:
    return ParserPipeline(structurer=get_nvidia_structurer())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@lru_cache
def get_embedding_pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline(embedder=get_ollama_embedder(), vector_store=get_vector_store())


@lru_cache
def get_optimization_pipeline() -> OptimizationPipeline:
    return OptimizationPipeline(optimizer_chain=get_optimizer_chain())
