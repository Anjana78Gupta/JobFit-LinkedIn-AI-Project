"""
OptimizationPipeline (agent_executor): the top-level entry point for the
Agentic Optimization Loop.

Given a ResumeProfile, a JobListing, and its precomputed MatchResult, this
pipeline finds every flagged bullet (via GapAnalyzer) and produces a
before/after rewrite (via OptimizerChain) for each one -- without
re-scoring or re-embedding anything. Scoring and generation stay cleanly
separated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.logging_config import get_logger
from src.core.exceptions import OptimizationError, ValidationError
from src.core.pipeline import BasePipeline
from src.embeddings.match_result import MatchResult
from src.ingestion.job_schema import JobListing
from src.optimization.gap_analyzer import GapAnalyzer
from src.optimization.optimizer_chain import OptimizerChain
from src.parsing.resume_schema import ResumeProfile

logger = get_logger(__name__)


@dataclass
class OptimizationPipelineInput:
    resume: ResumeProfile
    job: JobListing
    match_result: MatchResult


@dataclass
class BulletRewrite:
    bullet_id: str
    original_text: str
    optimized_text: str
    original_score: float


@dataclass
class OptimizationReport:
    job_id: str
    rewrites: list[BulletRewrite] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class OptimizationPipeline(BasePipeline[OptimizationPipelineInput, OptimizationReport]):
    """Generates targeted, bullet-by-bullet resume improvements."""

    name = "optimization_pipeline"

    def __init__(self, optimizer_chain: OptimizerChain | None = None, gap_analyzer: GapAnalyzer | None = None) -> None:
        self._optimizer_chain = optimizer_chain or OptimizerChain()
        self._gap_analyzer = gap_analyzer or GapAnalyzer()

    def validate_input(self, raw_input: object) -> OptimizationPipelineInput:
        if isinstance(raw_input, OptimizationPipelineInput):
            return raw_input

        if isinstance(raw_input, tuple) and len(raw_input) == 3:
            resume, job, match_result = raw_input
            if (
                isinstance(resume, ResumeProfile)
                and isinstance(job, JobListing)
                and isinstance(match_result, MatchResult)
            ):
                return OptimizationPipelineInput(resume=resume, job=job, match_result=match_result)

        raise ValidationError(
            "OptimizationPipeline expects (ResumeProfile, JobListing, MatchResult).",
            details={"received_type": type(raw_input).__name__},
        )

    def execute(self, validated_input: OptimizationPipelineInput) -> OptimizationReport:
        targets = self._gap_analyzer.build_targets(
            validated_input.resume, validated_input.job, validated_input.match_result
        )

        report = OptimizationReport(job_id=validated_input.job.job_id)
        if not targets:
            logger.info(
                "No flagged bullets to optimize",
                extra={"pipeline": self.name, "component": f"job_id={validated_input.job.job_id}"},
            )
            return report

        for target in targets:
            try:
                optimized_text = self._optimizer_chain.rewrite_bullet(target)
                report.rewrites.append(
                    BulletRewrite(
                        bullet_id=target.bullet_id,
                        original_text=target.original_text,
                        optimized_text=optimized_text,
                        original_score=target.current_score,
                    )
                )
            except OptimizationError as exc:
                report.failures.append(f"{target.bullet_id}: {exc.message}")
                logger.error(
                    "Bullet optimization failed",
                    extra={"pipeline": self.name, "component": f"bullet_id={target.bullet_id}"},
                )

        logger.info(
            "Optimization pipeline complete",
            extra={
                "pipeline": self.name,
                "component": f"rewritten={len(report.rewrites)} failed={len(report.failures)}",
            },
        )
        return report
