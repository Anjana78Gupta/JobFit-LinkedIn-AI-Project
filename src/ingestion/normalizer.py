"""
JobNormalizer: raw scraped dict -> validated JobListing.

Handles the reality of scraped data: inconsistent key names, missing
fields, requirements sometimes as a single blob of text instead of a list,
and whitespace noise from HTML scraping.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.exceptions import JobIngestionError
from src.ingestion.job_schema import JobListing
from src.parsing.text_cleaner import TextCleaner

# Maps common alternate key names (from different scrapers) to our schema fields.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "job_title", "position"),
    "company": ("company", "company_name", "employer"),
    "location": ("location", "job_location"),
    "seniority_level": ("seniority_level", "seniority", "experience_level"),
    "description": ("description", "job_description", "details"),
    "requirements": ("requirements", "qualifications", "requirements_list"),
    "posted_date": ("posted_date", "date_posted", "posted_at"),
    "source_url": ("source_url", "url", "job_url", "link"),
}


class JobNormalizer:
    """Converts a raw scraped job dict into a validated JobListing."""

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self._cleaner = cleaner or TextCleaner()

    def normalize(self, raw: dict[str, Any]) -> JobListing:
        try:
            fields = {key: self._first_present(raw, aliases) for key, aliases in _KEY_ALIASES.items()}

            title = self._cleaner.clean_text(str(fields["title"] or "")).strip()
            company = self._cleaner.clean_text(str(fields["company"] or "")).strip()
            description = self._cleaner.clean_text(str(fields["description"] or "")).strip()

            if not title or not company or not description:
                raise JobIngestionError(
                    "Job listing missing required field(s): title, company, or description.",
                    details={"raw_keys": list(raw.keys())},
                )

            requirements = self._normalize_requirements(fields["requirements"])

            return JobListing(
                title=title,
                company=company,
                location=self._clean_optional(fields["location"]),
                seniority_level=self._clean_optional(fields["seniority_level"]),
                description=description,
                requirements=requirements,
                posted_date=self._clean_optional(fields["posted_date"]),
                source_url=self._clean_optional(fields["source_url"]),
            )
        except JobIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise JobIngestionError(
                "Failed to normalize raw job listing.",
                details={"error": str(exc), "raw_keys": list(raw.keys())},
            ) from exc

    def _normalize_requirements(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._cleaner.clean_text(str(v)).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            # split a requirements blob on newlines/semicolons/bullets
            parts = re.split(r"[\n;]|(?:^|\n)[-•]\s*", value)
            return [self._cleaner.clean_text(p).strip() for p in parts if p.strip()]
        return []

    def _clean_optional(self, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = self._cleaner.clean_text(str(value)).strip()
        return cleaned or None

    @staticmethod
    def _first_present(raw: dict[str, Any], aliases: tuple[str, ...]) -> Any:
        for alias in aliases:
            if alias in raw and raw[alias] not in (None, ""):
                return raw[alias]
        return None
