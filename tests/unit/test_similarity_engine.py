"""
Tests SimilarityEngine's scoring logic with a fake, deterministic embedder
(no real Ollama server required) so this suite is fast and hermetic.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.embeddings.similarity_engine import SimilarityEngine
from src.embeddings.vector_store import VectorStore
from src.ingestion.job_schema import JobListing
from src.parsing.resume_schema import ExperienceBullet, ExperienceEntry, ResumeProfile, Skills


class FakeEmbedder:
    """Deterministic bag-of-words style embedder for hermetic unit tests."""

    DIM = 16

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for word in text.lower().split():
            idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)


def _build_resume() -> ResumeProfile:
    return ResumeProfile(
        summary="Machine learning engineer with LangChain and vector search experience",
        skills=Skills(technical=["python", "langchain", "vector embeddings"]),
        experience=[
            ExperienceEntry(
                company="Acme",
                title="ML Engineer",
                bullets=[
                    ExperienceBullet(text="Built retrieval augmented generation systems with LangChain"),
                    ExperienceBullet(text="Organized quarterly team lunches"),
                ],
            )
        ],
    )


def _build_job() -> JobListing:
    return JobListing(
        title="Senior Machine Learning Engineer",
        company="Acme AI",
        description="Build retrieval augmented generation systems using LangChain and vector embeddings",
        requirements=["LangChain experience", "vector embeddings experience"],
    )


def test_score_resume_against_job_flags_low_scoring_bullet() -> None:
    engine = SimilarityEngine(embedder=FakeEmbedder(), vector_store=VectorStore())
    result = engine.score_resume_against_job(_build_resume(), _build_job())

    assert 0.0 <= result.overall_fit_score <= 1.0
    bullets = {b.text: b for b in result.component_scores.experience_bullets}

    assert bullets["Built retrieval augmented generation systems with LangChain"].flagged_low is False
    assert bullets["Organized quarterly team lunches"].flagged_low is True


def test_scoring_is_deterministic_across_runs() -> None:
    resume, job = _build_resume(), _build_job()
    engine_a = SimilarityEngine(embedder=FakeEmbedder(), vector_store=VectorStore())
    engine_b = SimilarityEngine(embedder=FakeEmbedder(), vector_store=VectorStore())

    result_a = engine_a.score_resume_against_job(resume, job)
    result_b = engine_b.score_resume_against_job(resume, job)

    assert result_a.overall_fit_score == result_b.overall_fit_score
