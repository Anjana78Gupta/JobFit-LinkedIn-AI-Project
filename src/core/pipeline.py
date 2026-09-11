"""
BasePipeline: the common contract every pipeline in this project follows.

Every stage (Parser, Ingestion, Embedding, Optimization) subclasses this and
implements `validate_input` + `execute`. The API layer only ever calls
`.run(raw_input)` -- it never needs to know the internals of a given
pipeline, which keeps routes thin and pipelines independently testable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from config.logging_config import get_logger
from src.core.exceptions import AgentBaseError, ValidationError

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

logger = get_logger(__name__)


class BasePipeline(ABC, Generic[InputT, OutputT]):
    """
    Generic pipeline contract.

    Subclasses implement:
      - validate_input(raw_input) -> InputT   (raise ValidationError on bad input)
      - execute(validated_input) -> OutputT   (the actual pipeline logic)

    `run()` wraps both steps with timing + structured logging + consistent
    error propagation, so subclasses never need their own try/except/log
    boilerplate around the whole pipeline.
    """

    name: str = "base_pipeline"

    @abstractmethod
    def validate_input(self, raw_input: object) -> InputT:
        """Validate and coerce raw input into the pipeline's expected type."""

    @abstractmethod
    def execute(self, validated_input: InputT) -> OutputT:
        """Run the core pipeline logic on already-validated input."""

    def run(self, raw_input: object) -> OutputT:
        started_at = time.perf_counter()
        log_extra = {"pipeline": self.name}

        logger.info("Pipeline started", extra=log_extra)
        try:
            validated_input = self.validate_input(raw_input)
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert to our own error type
            raise ValidationError(
                f"{self.name}: input validation failed",
                details={"error": str(exc)},
            ) from exc

        try:
            result = self.execute(validated_input)
        except AgentBaseError:
            logger.error("Pipeline failed", extra=log_extra)
            raise
        except Exception as exc:  # noqa: BLE001 - never leak raw exceptions upward
            logger.error("Pipeline failed with unexpected error", extra=log_extra)
            raise AgentBaseError(
                f"{self.name}: unexpected failure during execution",
                details={"error": str(exc)},
            ) from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Pipeline completed",
            extra={**log_extra, "component": f"elapsed_ms={elapsed_ms:.1f}"},
        )
        return result
