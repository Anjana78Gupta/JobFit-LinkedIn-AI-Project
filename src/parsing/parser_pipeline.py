"""
ParserPipeline: PDF/DOCX file -> cleaned text -> NVIDIA structuring -> ResumeProfile.

This is the single entry point the API layer calls for resume uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.logging_config import get_logger
from src.core.exceptions import DocumentExtractionError, ValidationError
from src.core.interfaces import AbstractDocumentParser
from src.core.pipeline import BasePipeline
from src.parsing.docx_extractor import DocxExtractor
from src.parsing.nvidia_structurer import NvidiaResumeStructurer
from src.parsing.pdf_extractor import PDFExtractor
from src.parsing.resume_schema import ResumeProfile
from src.parsing.text_cleaner import TextCleaner

logger = get_logger(__name__)


@dataclass
class ParserPipelineInput:
    file_path: Path


class ParserPipeline(BasePipeline[ParserPipelineInput, ResumeProfile]):
    """Extracts, cleans, and structures a resume file into a ResumeProfile."""

    name = "parser_pipeline"

    def __init__(
        self,
        structurer: NvidiaResumeStructurer | None = None,
        cleaner: TextCleaner | None = None,
    ) -> None:
        self._extractors: list[AbstractDocumentParser] = [PDFExtractor(), DocxExtractor()]
        self._structurer = structurer or NvidiaResumeStructurer()
        self._cleaner = cleaner or TextCleaner()

    def validate_input(self, raw_input: object) -> ParserPipelineInput:
        if isinstance(raw_input, ParserPipelineInput):
            path = raw_input.file_path
        elif isinstance(raw_input, (str, Path)):
            path = Path(raw_input)
        else:
            raise ValidationError(
                "ParserPipeline expects a file path or ParserPipelineInput.",
                details={"received_type": type(raw_input).__name__},
            )

        if not path.exists():
            raise ValidationError(f"Resume file does not exist: {path}")

        if path.suffix.lower() not in self._all_supported_extensions():
            raise ValidationError(
                f"Unsupported resume file type: {path.suffix}",
                details={"supported": self._all_supported_extensions()},
            )

        return ParserPipelineInput(file_path=path)

    def execute(self, validated_input: ParserPipelineInput) -> ResumeProfile:
        extractor = self._extractor_for(validated_input.file_path)
        raw_text = extractor.extract_text(validated_input.file_path)
        cleaned_text = self._cleaner.clean_text(raw_text)
        return self._structurer.structure_to_profile(cleaned_text)

    def _extractor_for(self, path: Path) -> AbstractDocumentParser:
        for extractor in self._extractors:
            if path.suffix.lower() in extractor.supported_extensions():
                return extractor
        raise DocumentExtractionError(f"No extractor registered for {path.suffix}")

    def _all_supported_extensions(self) -> tuple[str, ...]:
        exts: list[str] = []
        for extractor in self._extractors:
            exts.extend(extractor.supported_extensions())
        return tuple(exts)
