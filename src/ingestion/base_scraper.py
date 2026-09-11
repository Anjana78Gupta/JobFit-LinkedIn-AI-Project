"""
AbstractScraper: interface for adapters that supply raw job listing dicts.

Note: this project does not perform live scraping of linkedin.com directly
(that's a ToS concern the user must own via their own scraping tool/browser
extension). Instead, adapters here accept already-scraped raw records --
from a JSON export, a CSV, or another tool's output -- and hand them to the
IngestionPipeline for normalization into JobListing objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractScraper(ABC):
    """Base interface for any source of raw (pre-scraped) job listing records."""

    @abstractmethod
    def fetch_raw(self) -> list[dict[str, Any]]:
        """Return a list of raw, unnormalized job listing dicts."""
