"""Matching + optimization endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.logging_config import get_logger
from src.api.dependencies import get_embedding_pipeline, get_optimization_pipeline
from src.core.exceptions import AgentBaseError
from src.embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingPipelineInput
from src.embeddings.match_result import MatchResult
from src.ingestion.job_schema import JobListing
from src.optimization.agent_executor import (
    OptimizationPipeline,
    OptimizationPipelineInput,
    OptimizationReport,
)
from src.parsing.resume_schema import ResumeProfile

router = APIRouter(prefix="/match", tags=["match"])
logger = get_logger(__name__)


class MatchRequest(BaseModel):
    resume: ResumeProfile
    jobs: list[JobListing]


class OptimizeRequest(BaseModel):
    resume: ResumeProfile
    job: JobListing
    match_result: MatchResult


@router.post("", response_model=list[MatchResult])
async def match_resume_to_jobs(
    request: MatchRequest,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
) -> list[MatchResult]:
    """
    Score one resume against one or more job listings via Ollama embeddings +
    cosine similarity. Returns results ranked by overall_fit_score descending.
    """
    try:
        return pipeline.run(EmbeddingPipelineInput(resume=request.resume, jobs=request.jobs))
    except AgentBaseError as exc:
        logger.error("Matching failed", extra={"component": str(exc)})
        raise HTTPException(status_code=422, detail={"message": exc.message, **exc.details}) from exc


@router.post("/optimize", response_model=OptimizationReport)
async def optimize_resume(
    request: OptimizeRequest,
    pipeline: OptimizationPipeline = Depends(get_optimization_pipeline),
) -> OptimizationReport:
    """
    Given a resume, a target job, and a previously computed MatchResult,
    generate bullet-by-bullet rewrites for every flagged (low-scoring) bullet.
    """
    try:
        return pipeline.run(
            OptimizationPipelineInput(
                resume=request.resume, job=request.job, match_result=request.match_result
            )
        )
    except AgentBaseError as exc:
        logger.error("Optimization failed", extra={"component": str(exc)})
        raise HTTPException(status_code=422, detail={"message": exc.message, **exc.details}) from exc
