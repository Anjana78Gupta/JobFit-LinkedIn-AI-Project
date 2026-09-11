"""
NVIDIA AI endpoint wrapper: raw resume text -> structured ResumeProfile JSON.

The NVIDIA NIM endpoints are OpenAI-compatible, so we use the `openai`
client pointed at NVIDIA's base URL. The model is asked to return JSON only,
which we validate against ResumeProfile -- any schema drift is caught here,
not downstream.
"""

from __future__ import annotations

import json

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.core.exceptions import NvidiaAPIError, ResumeParseError
from src.parsing.resume_schema import ResumeProfile

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a precise resume-structuring engine.
Given raw, messily-extracted resume text, output ONLY a single valid JSON \
object matching this exact schema -- no markdown fences, no commentary:

{
  "contact": {"name": str, "email": str, "phone": str, "location": str, "linkedin_url": str|null},
  "summary": str|null,
  "skills": {"technical": [str], "tools": [str], "soft": [str]},
  "experience": [
    {"company": str, "title": str, "start_date": str|null, "end_date": str|null,
     "location": str|null, "bullets": [{"text": str}]}
  ],
  "education": [
    {"institution": str, "degree": str, "field": str|null,
     "start_date": str|null, "end_date": str|null}
  ],
  "certifications": [str]
}

Rules:
- Preserve the original wording of experience bullets and summary; do not \
invent or embellish accomplishments.
- If a field is not present in the source text, use null or an empty list \
(never fabricate values).
- Every bullet under "experience" must be its own list item, not merged.
"""


class NvidiaResumeStructurer:
    """Calls an NVIDIA NIM chat-completions endpoint to structure resume text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = None  # lazily constructed to avoid import cost when unused

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise NvidiaAPIError(
                "openai package is not installed. Run `pip install openai`.",
            ) from exc

        self._client = OpenAI(
            base_url=self._settings.nvidia_base_url,
            api_key=self._settings.nvidia_api_key,
            timeout=self._settings.nvidia_request_timeout_seconds,
        )
        return self._client

    def structure(self, raw_text: str) -> dict:
        """Return a raw dict parsed from the model's JSON response."""
        if not raw_text or not raw_text.strip():
            raise ResumeParseError("Cannot structure empty resume text.")

        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(1, self._settings.nvidia_max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._settings.nvidia_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": raw_text},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.error(
                    "NVIDIA response was not valid JSON",
                    extra={"component": f"attempt={attempt}"},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.error(
                    "NVIDIA API call failed",
                    extra={"component": f"attempt={attempt} error={exc}"},
                )

        raise NvidiaAPIError(
            "NVIDIA structured extraction failed after retries.",
            details={"retries": self._settings.nvidia_max_retries, "last_error": str(last_error)},
        )

    def structure_to_profile(self, raw_text: str) -> ResumeProfile:
        """Structure raw text and validate it into a ResumeProfile model."""
        raw_dict = self.structure(raw_text)
        try:
            return ResumeProfile.model_validate(raw_dict)
        except Exception as exc:  # noqa: BLE001
            raise ResumeParseError(
                "NVIDIA output did not match the ResumeProfile schema.",
                details={"error": str(exc), "raw": raw_dict},
            ) from exc
