"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from config.logging_config import configure_logging, get_logger
from config.settings import get_settings
from src.api.routes import job_routes, match_routes, resume_routes

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="LinkedIn Job Search Agent & Resume Optimizer",
    version="0.1.0",
    description=(
        "Parses resumes into structured profiles, ingests scraped job listings, "
        "scores job fit via local embeddings + cosine similarity, and generates "
        "targeted bullet-by-bullet resume improvements."
    ),
)

app.include_router(resume_routes.router)
app.include_router(job_routes.router)
app.include_router(match_routes.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "Application started",
        extra={
            "component": (
                f"nvidia_model={settings.nvidia_model} "
                f"ollama_embed_model={settings.ollama_embed_model} "
                f"ollama_gen_model={settings.ollama_generation_model}"
            )
        },
    )
