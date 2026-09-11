"""
Smoke test wiring IngestionPipeline -> EmbeddingPipeline together with a
fake embedder, using the real sample_jobs.json fixture. Does not touch
NVIDIA or Ollama -- exercises schema boundaries only.

Run with: pytest tests/integration/test_end_to_end_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingPipelineInput
from src.embeddings.vector_store import VectorStore
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.parsing.resume_schema import ExperienceBullet, ExperienceEntry, ResumeProfile, Skills
from tests.unit.test_similarity_engine import FakeEmbedder

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_ingestion_then_embedding_pipeline_end_to_end() -> None:
    raw_jobs = json.loads((FIXTURES / "sample_jobs.json").read_text())

    ingestion = IngestionPipeline()
    jobs = ingestion.run(raw_jobs)
    assert len(jobs) == 2

    resume = ResumeProfile(
        summary="ML engineer",
        skills=Skills(technical=["python", "langchain"]),
        experience=[
            ExperienceEntry(
                company="Acme",
                title="ML Engineer",
                bullets=[ExperienceBullet(text="Built LangChain-based RAG pipelines")],
            )
        ],
    )

    embedding_pipeline = EmbeddingPipeline(embedder=FakeEmbedder(), vector_store=VectorStore())
    results = embedding_pipeline.run(EmbeddingPipelineInput(resume=resume, jobs=jobs))

    assert len(results) == 2
    # Results should be ranked descending by fit score.
    assert results[0].overall_fit_score >= results[1].overall_fit_score
