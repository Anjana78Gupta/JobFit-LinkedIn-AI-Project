"""
OptimizerChain: a LangChain LCEL chain that rewrites a single low-scoring
resume bullet to better align with a target job, using a local Ollama
generation model (default: llama3.1:8b-instruct).
"""

from __future__ import annotations

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.core.exceptions import OptimizationError
from src.optimization.gap_analyzer import OptimizationTarget
from src.optimization.prompts.optimization_prompts import (
    OPTIMIZATION_SYSTEM_PROMPT_V1,
    OPTIMIZATION_USER_TEMPLATE_V1,
)

logger = get_logger(__name__)


class OptimizerChain:
    """Wraps a LangChain chain: prompt -> ChatOllama -> plain-text bullet."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chain = None  # built lazily to avoid import cost when unused

    def _build_chain(self):
        if self._chain is not None:
            return self._chain

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise OptimizationError(
                "langchain-core is not installed. Run `pip install langchain-core`.",
            ) from exc

        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise OptimizationError(
                "langchain-ollama is not installed. Run `pip install langchain-ollama`.",
            ) from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", OPTIMIZATION_SYSTEM_PROMPT_V1),
                ("human", OPTIMIZATION_USER_TEMPLATE_V1),
            ]
        )
        llm = ChatOllama(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_generation_model,
            temperature=self._settings.ollama_generation_temperature,
        )
        self._chain = prompt | llm | StrOutputParser()
        return self._chain

    def rewrite_bullet(self, target: OptimizationTarget) -> str:
        chain = self._build_chain()
        requirements_block = (
            "\n".join(f"- {r}" for r in target.job_requirements)
            if target.job_requirements
            else "(no explicit requirements list provided; use the job description)"
        )

        try:
            result = chain.invoke(
                {
                    "job_title": target.job_title,
                    "job_requirements": requirements_block,
                    "original_text": target.original_text,
                    "current_score": target.current_score,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise OptimizationError(
                f"Failed to generate optimized bullet for bullet_id={target.bullet_id}",
                details={"error": str(exc)},
            ) from exc

        rewritten = result.strip().strip('"')
        if not rewritten:
            raise OptimizationError(
                f"Optimizer chain returned an empty rewrite for bullet_id={target.bullet_id}",
            )
        return rewritten
