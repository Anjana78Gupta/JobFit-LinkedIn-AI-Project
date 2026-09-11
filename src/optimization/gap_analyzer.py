"""
GapAnalyzer: turns a MatchResult's flagged bullets into optimization targets.

Pure data-shaping logic, no LLM calls here -- it just cross-references
MatchResult.flagged_bullets() against the ResumeProfile and JobListing to
build the (bullet, job_requirements) pairs the Optimization Agent needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.embeddings.match_result import MatchResult
from src.ingestion.job_schema import JobListing
from src.parsing.resume_schema import ResumeProfile


@dataclass
class OptimizationTarget:
    bullet_id: str
    original_text: str
    current_score: float
    job_title: str
    job_requirements: list[str]
    job_description: str


class GapAnalyzer:
    """Identifies low-scoring resume bullets and packages them with job context."""

    def build_targets(
        self, resume: ResumeProfile, job: JobListing, match_result: MatchResult
    ) -> list[OptimizationTarget]:
        targets: list[OptimizationTarget] = []
        for bullet_score in match_result.flagged_bullets():
            bullet = resume.find_bullet(bullet_score.bullet_id)
            if bullet is None:
                continue
            targets.append(
                OptimizationTarget(
                    bullet_id=bullet.id,
                    original_text=bullet.text,
                    current_score=bullet_score.score,
                    job_title=job.title,
                    job_requirements=job.requirements,
                    job_description=job.description,
                )
            )
        return targets
