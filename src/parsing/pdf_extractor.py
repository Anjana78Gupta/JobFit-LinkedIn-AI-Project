"""PDF -> raw text extraction, using pypdf."""

from __future__ import annotations

from pathlib import Path

from config.logging_config import get_logger
from src.core.exceptions import DocumentExtractionError
from src.parsing.base_parser import AbstractDocumentParser

logger = get_logger(__name__)


class PDFExtractor(AbstractDocumentParser):
    """Extracts raw text from .pdf resumes."""

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        return (".pdf",)

    def extract_text(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DocumentExtractionError(
                "pypdf is not installed. Run `pip install pypdf`.",
            ) from exc

        if not file_path.exists():
            raise DocumentExtractionError(f"File not found: {file_path}")

        try:
            reader = PdfReader(str(file_path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # noqa: BLE001
            raise DocumentExtractionError(
                f"Failed to extract text from PDF: {file_path.name}",
                details={"error": str(exc)},
            ) from exc

        text = "\n".join(pages_text).strip()
        if not text:
            raise DocumentExtractionError(
                f"No extractable text found in PDF: {file_path.name}. "
                "It may be a scanned/image-only document requiring OCR.",
            )

        logger.info(
            "PDF text extracted",
            extra={"component": f"pages={len(reader.pages)} chars={len(text)}"},
        )
        return text
