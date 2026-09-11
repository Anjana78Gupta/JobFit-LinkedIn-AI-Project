#!/usr/bin/env python3
"""
CLI entry point for running the full pipeline locally, without the API layer.

Usage:
    python scripts/run_pipeline_cli.py \
        --resume path/to/resume.pdf \
        --jobs path/to/scraped_jobs.json \
        --optimize

Requires a running local Ollama server (`ollama serve`) with
`nomic-embed-text` and `llama3.1:8b-instruct` pulled, plus a valid
NVIDIA_API_KEY in the environment (or a .env file) for resume structuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import configure_logging, get_logger  # noqa: E402
from config.settings import get_settings  # noqa: E402
from src.core.exceptions import AgentBaseError  # noqa: E402
from src.embeddings.embedding_pipeline import EmbeddingPipeline, EmbeddingPipelineInput  # noqa: E402
from src.embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from src.ingestion.ingestion_pipeline import IngestionPipeline  # noqa: E402
from src.ingestion.linkedin_scraper import LinkedInScraper  # noqa: E402
from src.optimization.agent_executor import OptimizationPipeline, OptimizationPipelineInput  # noqa: E402
from src.parsing.parser_pipeline import ParserPipeline  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the resume-job matching pipeline end to end.")
    parser.add_argument("--resume", required=True, type=Path, help="Path to a PDF or DOCX resume")
    parser.add_argument("--jobs", required=True, type=Path, help="Path to a JSON file of scraped jobs")
    parser.add_argument("--optimize", action="store_true", help="Also run the optimization agent on the top job")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        print("Parsing resume...", file=sys.stderr)
        resume_profile = ParserPipeline().run(args.resume)

        print("Ingesting job listings...", file=sys.stderr)
        raw_jobs = json.loads(args.jobs.read_text())
        jobs = IngestionPipeline().run(LinkedInScraper(raw_jobs))

        print("Scoring resume against jobs...", file=sys.stderr)
        embedder = OllamaEmbedder(settings=settings)
        match_results = EmbeddingPipeline(embedder=embedder).run(
            EmbeddingPipelineInput(resume=resume_profile, jobs=jobs)
        )

        print(json.dumps([m.model_dump() for m in match_results], indent=2))

        if args.optimize and match_results:
            top_job = next(j for j in jobs if j.job_id == match_results[0].job_id)
            print("Optimizing resume against top-ranked job...", file=sys.stderr)
            report = OptimizationPipeline().run(
                OptimizationPipelineInput(
                    resume=resume_profile, job=top_job, match_result=match_results[0]
                )
            )
            print(json.dumps(report.__dict__, default=lambda o: o.__dict__, indent=2))

    except AgentBaseError as exc:
        logger.error("Pipeline run failed", extra={"component": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
