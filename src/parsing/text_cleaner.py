"""
Deterministic text preprocessing and chunking.

Rules of thumb encoded here:
  - Never split a bullet point / line across two chunks.
  - Collapse whitespace and normalize bullet glyphs without touching wording.
  - Chunk boundaries prefer blank lines, then line breaks, then sentence
    boundaries -- word-level splitting is the last resort so a phrase like
    "led a team of 12 engineers" never gets torn in half.
"""

from __future__ import annotations

import re

from config.settings import get_settings

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_BULLET_GLYPHS_RE = re.compile(r"^[•●▪‣⁃\-\*]\s*", re.MULTILINE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class TextCleaner:
    """Stateless helper for normalizing and chunking extracted resume text."""

    def __init__(self, max_chunk_chars: int | None = None, chunk_overlap_chars: int | None = None) -> None:
        settings = get_settings()
        self.max_chunk_chars = max_chunk_chars or settings.max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars or settings.chunk_overlap_chars

    def clean_text(self, raw: str) -> str:
        """Normalize whitespace and bullet glyphs without altering wording."""
        if not raw:
            return ""

        text = _CONTROL_CHARS_RE.sub("", raw)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _WHITESPACE_RE.sub(" ", text)
        text = _BULLET_GLYPHS_RE.sub("- ", text)
        text = _MULTI_BLANK_RE.sub("\n\n", text)

        # Strip trailing whitespace per line while preserving line structure.
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def chunk_text(self, text: str, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
        """
        Split cleaned text into chunks that respect line boundaries.

        A chunk never ends mid-line. If a single line exceeds max_chars, it is
        split on sentence boundaries as a fallback, never mid-word.
        """
        max_chars = max_chars or self.max_chunk_chars
        overlap = overlap if overlap is not None else self.chunk_overlap_chars

        if not text:
            return []

        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for the newline we'll rejoin with
            if current_len + line_len > max_chars and current:
                chunks.append("\n".join(current).strip())
                # carry overlap forward from the tail of the previous chunk
                overlap_lines = self._tail_lines_within(current, overlap)
                current = list(overlap_lines)
                current_len = sum(len(l) + 1 for l in current)

            if line_len > max_chars:
                # single line too long -> split on sentence boundaries
                for sub in self._split_long_line(line, max_chars):
                    current.append(sub)
                    current_len += len(sub) + 1
            else:
                current.append(line)
                current_len += line_len

        if current:
            chunks.append("\n".join(current).strip())

        return [c for c in chunks if c]

    @staticmethod
    def _tail_lines_within(lines: list[str], overlap_chars: int) -> list[str]:
        if overlap_chars <= 0:
            return []
        tail: list[str] = []
        total = 0
        for line in reversed(lines):
            total += len(line) + 1
            if total > overlap_chars:
                break
            tail.insert(0, line)
        return tail

    @staticmethod
    def _split_long_line(line: str, max_chars: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", line)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                pieces.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current.strip())
        return pieces or [line[:max_chars]]
