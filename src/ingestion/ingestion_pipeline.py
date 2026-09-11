"""
IngestionPipeline: raw scraped job records -> list[JobListing].

Individual bad records are logged and skipped rather than failing the
whole batch -- one malformed scrape shouldn't block ranking the other 49.
"""

from __future__ import annotations

from typing import Any

from config.logging_config import get_logger
from src.core.exceptions import JobIngestionError, ValidationError
from src.core.pipeline import BasePipeline
from src.ingestion.base_scraper import AbstractScraper
from src.ingestion.job_schema import JobListing
from src.ingestion.normalizer import JobNormalizer

logger = get_logger(__name__)


class IngestionPipeline(BasePipeline[list[dict[str, Any]], list[JobListing]]):
    """Normalizes a batch of raw scraped job dicts into JobListing objects."""

    name = "ingestion_pipeline"

    def __init__(self, normalizer: JobNormalizer | None = None) -> None:
        self._normalizer = normalizer or JobNormalizer()

    def validate_input(self, raw_input: object) -> list[dict[str, Any]]:
        if isinstance(raw_input, AbstractScraper):
            raw_input = raw_input.fetch_raw()

        if not isinstance(raw_input, list):
            raise ValidationError(
                "IngestionPipeline expects a list of raw dicts or an AbstractScraper.",
                details={"received_type": type(raw_input).__name__},
            )

        if not all(isinstance(item, dict) for item in raw_input):
            raise ValidationError("All items in the ingestion batch must be dicts.")

        return raw_input

    def execute(self, validated_input: list[dict[str, Any]]) -> list[JobListing]:
        listings: list[JobListing] = []
        skipped = 0

        for raw in validated_input:
            try:
                listings.append(self._normalizer.normalize(raw))
            except JobIngestionError as exc:
                skipped += 1
                logger.warning(
                    "Skipped malformed job listing",
                    extra={"component": f"error={exc.message}"},
                )

        logger.info(
            "Ingestion batch complete",
            extra={"component": f"accepted={len(listings)} skipped={skipped}"},
        )
        return listings
