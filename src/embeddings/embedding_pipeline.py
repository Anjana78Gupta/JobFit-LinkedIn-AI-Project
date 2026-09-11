"""
EmbeddingPipeline: (ResumeProfile, JobListing) -> MatchResult.

Also supports batch mode: one resume scored against N job listings,
producing a ranked shortlist sorted by overall_fit_score descending.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.logging_config import get_logger
from src.core.exceptions import ValidationError
from src.core.interfaces import EmbedderProtocol
from src.core.pipeline import BasePipeline
from src.embeddings.match_result import MatchResult
from src.embeddings.similarity_engine import SimilarityEngine
from src.embeddings.vector_store import VectorStore
from src.ingestion.job_schema import JobListing
from src.parsing.resume_schema import ResumeProfile

logger = get_logger(__name__)


@dataclass
class EmbeddingPipelineInput:
    resume: ResumeProfile
    jobs: list[JobListing]


class EmbeddingPipeline(BasePipeline[EmbeddingPipelineInput, list[MatchResult]]):
    """Scores a resume against one or more job listings via cosine similarity."""

    name = "embedding_pipeline"

    def __init__(self, embedder: EmbedderProtocol, vector_store: VectorStore | None = None) -> None:
        self._vector_store = vector_store or VectorStore()
        self._engine = SimilarityEngine(embedder=embedder, vector_store=self._vector_store)

    def validate_input(self, raw_input: object) -> EmbeddingPipelineInput:
        if isinstance(raw_input, EmbeddingPipelineInput):
            return raw_input

        if isinstance(raw_input, tuple) and len(raw_input) == 2:
            resume, jobs = raw_input
            if isinstance(jobs, JobListing):
                jobs = [jobs]
            if isinstance(resume, ResumeProfile) and isinstance(jobs, list):
                return EmbeddingPipelineInput(resume=resume, jobs=jobs)

        raise ValidationError(
            "EmbeddingPipeline expects (ResumeProfile, JobListing | list[JobListing]).",
            details={"received_type": type(raw_input).__name__},
        )

    def execute(self, validated_input: EmbeddingPipelineInput) -> list[MatchResult]:
        results = [
            self._engine.score_resume_against_job(validated_input.resume, job)
            for job in validated_input.jobs
        ]
        results.sort(key=lambda r: r.overall_fit_score, reverse=True)
        logger.info(
            "Embedding pipeline batch complete",
            extra={"pipeline": self.name, "component": f"jobs_scored={len(results)}"},
        )
        return results
