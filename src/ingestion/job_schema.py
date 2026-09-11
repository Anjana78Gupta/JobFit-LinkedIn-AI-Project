"""
JobListing: the structured output of the Ingestion Pipeline.

Like ResumeProfile, this is a schema boundary -- the Vector Engine only
ever consumes JobListing objects, never raw scraped dicts.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:10]}"


class JobListing(BaseModel):
    job_id: str = Field(default_factory=_new_job_id)
    title: str
    company: str
    location: str | None = None
    seniority_level: str | None = None
    description: str
    requirements: list[str] = Field(default_factory=list)
    posted_date: str | None = None
    source_url: str | None = None

    def full_text_for_embedding(self) -> str:
        """Concatenate the fields that matter for semantic matching."""
        parts = [self.title, self.description, *self.requirements]
        return "\n".join(p for p in parts if p)
