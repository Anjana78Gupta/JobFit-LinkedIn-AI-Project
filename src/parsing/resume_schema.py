"""
ResumeProfile: the structured output of the Parser Pipeline.

This is the schema boundary between parsing and everything downstream
(Vector Engine, Optimization Agent). Nothing past this module should ever
touch a raw PDF/DOCX or unstructured text again.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Contact(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin_url: str | None = None


class Skills(BaseModel):
    technical: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_list(cls, data: Any) -> Any:
        # Some LLM responses return skills as a flat list instead of the
        # technical/tools/soft breakdown -- treat that as all "technical".
        if isinstance(data, list):
            return {"technical": data, "tools": [], "soft": []}
        return data

    def as_flat_list(self) -> list[str]:
        return [*self.technical, *self.tools, *self.soft]


class ExperienceBullet(BaseModel):
    id: str = Field(default_factory=_new_id)
    text: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, data: Any) -> Any:
        # Some LLM responses return bullets as plain strings instead of
        # {"text": ...} objects.
        if isinstance(data, str):
            return {"text": data}
        return data


class ExperienceEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    company: str = ""
    title: str = ""
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    bullets: list[ExperienceBullet] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_bullets(cls, data: Any) -> Any:
        # Some LLM responses nest bullets under "responsibilities" or
        # "description" instead of "bullets".
        if isinstance(data, dict) and "bullets" not in data:
            for alt_key in ("responsibilities", "description", "highlights", "achievements"):
                if alt_key in data and isinstance(data[alt_key], list):
                    data = {**data, "bullets": data[alt_key]}
                    break
        return data


class EducationEntry(BaseModel):
    institution: str = ""
    degree: str = ""
    field: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ResumeProfile(BaseModel):
    """Fully structured representation of a candidate's resume."""

    contact: Contact = Field(default_factory=Contact)
    summary: str | None = None
    skills: Skills = Field(default_factory=Skills)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    def all_bullets(self) -> list[ExperienceBullet]:
        """Flatten every experience bullet across all roles, for embedding."""
        return [bullet for entry in self.experience for bullet in entry.bullets]

    def find_bullet(self, bullet_id: str) -> ExperienceBullet | None:
        for bullet in self.all_bullets():
            if bullet.id == bullet_id:
                return bullet
        return None
