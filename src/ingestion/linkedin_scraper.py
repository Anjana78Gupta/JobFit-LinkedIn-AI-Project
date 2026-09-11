"""
LinkedInScraper: adapter for pre-scraped LinkedIn job listing JSON.

Accepts either a path to a JSON file (a list of raw job dicts) or an
in-memory list of dicts already produced by an external scraping tool
(e.g. a browser extension export, a third-party scraping API, or a
manually curated JSON file).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.logging_config import get_logger
from src.core.exceptions import JobIngestionError
from src.ingestion.base_scraper import AbstractScraper

logger = get_logger(__name__)


class LinkedInScraper(AbstractScraper):
    """Loads pre-scraped LinkedIn job data from a file path or in-memory list."""

    def __init__(self, source: str | Path | list[dict[str, Any]]) -> None:
        self._source = source

    def fetch_raw(self) -> list[dict[str, Any]]:
        if isinstance(self._source, list):
            return self._source

        path = Path(self._source)
        if not path.exists():
            raise JobIngestionError(f"Job listing source file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise JobIngestionError(
                f"Job listing file is not valid JSON: {path}",
                details={"error": str(exc)},
            ) from exc

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise JobIngestionError(
                f"Expected a JSON list of job records in {path}, got {type(data).__name__}",
            )

        logger.info("Loaded raw job listings", extra={"component": f"count={len(data)}"})
        return data
