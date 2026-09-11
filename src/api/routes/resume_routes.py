"""Resume upload + parsing endpoints."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from config.logging_config import get_logger
from src.api.dependencies import get_parser_pipeline
from src.core.exceptions import AgentBaseError
from src.parsing.parser_pipeline import ParserPipeline
from src.parsing.resume_schema import ResumeProfile

router = APIRouter(prefix="/resume", tags=["resume"])
logger = get_logger(__name__)


@router.post("/parse", response_model=ResumeProfile)
async def parse_resume(
    file: UploadFile = File(...),
    pipeline: ParserPipeline = Depends(get_parser_pipeline),
) -> ResumeProfile:
    """Upload a PDF or DOCX resume and receive back a structured ResumeProfile."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / (file.filename or f"upload{suffix}")
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            return pipeline.run(tmp_path)
        except AgentBaseError as exc:
            logger.error("Resume parsing failed", extra={"component": str(exc)})
            # Surface details (e.g. the raw NVIDIA output that failed validation)
            # directly in the response so mismatches are debuggable without
            # digging through server logs.
            raise HTTPException(
                status_code=422,
                detail={"message": exc.message, **exc.details},
            ) from exc
