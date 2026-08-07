import pytest

from src.document_processing import DocumentProcessor


@pytest.fixture(scope="session")
def document_processor() -> DocumentProcessor:
    """Shared DocumentProcessor instance.

    Presidio/spaCy model loading in DocumentProcessor.__init__ is
    expensive, so the whole test session reuses one instance instead of
    each test module constructing its own.
    """

    return DocumentProcessor(mask_pii=False)
