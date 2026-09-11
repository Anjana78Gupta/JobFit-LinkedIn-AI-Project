"""
SimilarityEngine: deterministic cosine-similarity scoring.

No LLM involved -- given two vectors (or a resume + job pair, via the
vector store), the output is fully reproducible. This is what makes the
"job-fit ranking" auditable and stable across runs.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.embeddings.match_result import BulletScore, ComponentScores, MatchResult
from src.embeddings.vector_store import VectorStore
from src.ingestion.job_schema import JobListing
from src.parsing.resume_schema import ResumeProfile
from src.core.interfaces import EmbedderProtocol

logger = get_logger(__name__)


class SimilarityEngine:
    """Computes cosine-similarity-based fit scores between a resume and a job."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store or VectorStore()
        self._settings = settings or get_settings()

    @staticmethod
    def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors, as a plain float in [-1, 1]."""
        a = vec_a.reshape(1, -1)
        b = vec_b.reshape(1, -1)
        return float(cosine_similarity(a, b)[0][0])

    def score_resume_against_job(self, resume: ResumeProfile, job: JobListing) -> MatchResult:
        model_name = self._settings.ollama_embed_model
        job_text = job.full_text_for_embedding()
        job_vector = self._vector_store.get_or_compute(job_text, self._embedder, model_name)

        skills_text = ", ".join(resume.skills.as_flat_list()) or ""
        skills_score = (
            self.cosine(self._vector_store.get_or_compute(skills_text, self._embedder, model_name), job_vector)
            if skills_text
            else 0.0
        )

        summary_text = resume.summary or ""
        summary_score = (
            self.cosine(self._vector_store.get_or_compute(summary_text, self._embedder, model_name), job_vector)
            if summary_text
            else 0.0
        )

        bullet_scores: list[BulletScore] = []
        bullets = resume.all_bullets()
        if bullets:
            bullet_texts = [b.text for b in bullets]
            bullet_vectors = self._vector_store.get_or_compute_batch(bullet_texts, self._embedder, model_name)
            for bullet, vector in zip(bullets, bullet_vectors):
                score = self.cosine(vector, job_vector)
                bullet_scores.append(
                    BulletScore(
                        bullet_id=bullet.id,
                        text=bullet.text,
                        score=score,
                        flagged_low=score < self._settings.low_score_threshold,
                    )
                )

        component_weights = {"skills": 0.3, "summary": 0.2, "bullets": 0.5}
        avg_bullet_score = (
            sum(b.score for b in bullet_scores) / len(bullet_scores) if bullet_scores else 0.0
        )
        overall = (
            component_weights["skills"] * skills_score
            + component_weights["summary"] * summary_score
            + component_weights["bullets"] * avg_bullet_score
        )

        result = MatchResult(
            job_id=job.job_id,
            overall_fit_score=round(overall, 4),
            component_scores=ComponentScores(
                skills=round(skills_score, 4),
                summary=round(summary_score, 4),
                experience_bullets=bullet_scores,
            ),
            low_score_threshold=self._settings.low_score_threshold,
        )

        logger.info(
            "Computed match result",
            extra={
                "pipeline": "embedding_pipeline",
                "component": f"job_id={job.job_id} overall={result.overall_fit_score} "
                f"flagged={len(result.flagged_bullets())}",
            },
        )
        return result
