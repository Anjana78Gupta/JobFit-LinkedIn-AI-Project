import json
from pathlib import Path

import pytest

from src.core.exceptions import JobIngestionError
from src.ingestion.normalizer import JobNormalizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_sample_jobs() -> list[dict]:
    return json.loads((FIXTURES / "sample_jobs.json").read_text())


def test_normalizer_handles_standard_key_names() -> None:
    normalizer = JobNormalizer()
    raw = _load_sample_jobs()[0]
    listing = normalizer.normalize(raw)

    assert listing.title == "Senior Machine Learning Engineer"
    assert listing.company == "Acme AI"
    assert len(listing.requirements) == 4


def test_normalizer_handles_alternate_key_names_and_string_requirements() -> None:
    normalizer = JobNormalizer()
    raw = _load_sample_jobs()[1]
    listing = normalizer.normalize(raw)

    assert listing.title == "Data Engineer"
    assert listing.company == "Beta Analytics"
    # requirements was a newline-delimited string blob, not a list
    assert len(listing.requirements) == 3


def test_normalizer_raises_on_missing_required_fields() -> None:
    normalizer = JobNormalizer()
    with pytest.raises(JobIngestionError):
        normalizer.normalize({"title": "Engineer"})
