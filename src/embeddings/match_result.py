"""
MatchResult: deterministic output of the Vector Engine.

This is the schema boundary between scoring (pure vector math, reproducible)
and the Optimization Agent (LLM-driven, stochastic). The agent never
recomputes scores -- it only reads which bullet_ids are flagged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BulletScore(BaseModel):
    bullet_id: str
    text: str
    score: float
    flagged_low: bool


class ComponentScores(BaseModel):
    skills: float
    summary: float
    experience_bullets: list[BulletScore] = Field(default_factory=list)


class MatchResult(BaseModel):
    job_id: str
    resume_id: str | None = None
    overall_fit_score: float
    component_scores: ComponentScores
    low_score_threshold: float

    def flagged_bullets(self) -> list[BulletScore]:
        return [b for b in self.component_scores.experience_bullets if b.flagged_low]
