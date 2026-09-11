"""Job listing ingestion endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from config.logging_config import get_logger
from src.api.dependencies import get_ingestion_pipeline
from src.core.exceptions import AgentBaseError
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.job_schema import JobListing

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = get_logger(__name__)


@router.post("/ingest", response_model=list[JobListing])
async def ingest_jobs(
    raw_jobs: list[dict[str, Any]],
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> list[JobListing]:
    """
    Accept a batch of raw scraped LinkedIn job records (JSON) and return
    normalized JobListing objects. Malformed individual records are skipped,
    not fatal to the batch.
    """
    try:
        return pipeline.run(raw_jobs)
    except AgentBaseError as exc:
        logger.error("Job ingestion failed", extra={"component": str(exc)})
        raise HTTPException(status_code=422, detail={"message": exc.message, **exc.details}) from exc
