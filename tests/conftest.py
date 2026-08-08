import urllib.request
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.document_processing import DocumentProcessor

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:0.5b"


def _ollama_reachable(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.0) -> bool:
    """Probe a local Ollama server without pulling in a dependency just
    for the check -- used to skip (not fail) the real-generation
    integration tests on machines/CI without Ollama running.
    """

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    return _ollama_reachable()


@pytest.fixture
def make_pdf_bytes():
    """Factory returning raw bytes for a minimal, valid, single-page PDF
    containing the given text.

    No PDF-writing library (reportlab, fpdf, ...) is a project
    dependency, so this hand-builds the PDF object graph -- including a
    correct byte-offset xref table -- directly. pdfplumber/pdfminer can
    then parse it exactly like a real PDF without needing any external
    OCR/rasterization tooling (poppler, ghostscript, tesseract), which is
    the point: these tests need to control whether a PDF "has a text
    layer" independently of what's installed on the test machine.
    """

    def _make(text: str) -> bytes:
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]

        lines: list[str] = []
        line: list[str] = []
        length = 0
        for word in text.split():
            line.append(word)
            length += len(word) + 1
            if length > 60:
                lines.append(" ".join(line))
                line, length = [], 0
        if line:
            lines.append(" ".join(line))

        stream = "BT /F1 12 Tf 50 750 Td 14 TL\n"
        for one_line in lines:
            escaped = one_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream += f"({escaped}) Tj T*\n"
        stream += "ET"
        stream_bytes = stream.encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream_bytes) + stream_bytes + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

        xref_offset = len(out)
        n = len(objects) + 1
        out += f"xref\n0 {n}\n".encode() + b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += (
            b"trailer\n"
            + f"<< /Size {n} /Root 1 0 R >>\n".encode()
            + b"startxref\n"
            + f"{xref_offset}\n".encode()
            + b"%%EOF"
        )
        return bytes(out)

    return _make


@pytest.fixture(scope="session")
def document_processor() -> DocumentProcessor:
    """Shared DocumentProcessor instance.

    Presidio/spaCy model loading in DocumentProcessor.__init__ is
    expensive, so the whole test session reuses one instance instead of
    each test module constructing its own.
    """

    return DocumentProcessor(mask_pii=False)


def _patch_chroma(monkeypatch) -> None:
    """Replace Chroma's persistent client + embedding model with mocks so
    constructing a real VectorStore (directly, or transitively via
    `main.py`/`src/mcp_server.py` module-level wiring) doesn't touch disk,
    download a sentence-transformers model, or need network access.
    """

    import chromadb
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(chromadb, "PersistentClient", MagicMock())
    monkeypatch.setattr(embedding_functions, "SentenceTransformerEmbeddingFunction", MagicMock())


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Patches `langchain_anthropic.ChatAnthropic` so any code path that
    lazily constructs a real Claude client (e.g. `SQLAgent._default_llm`,
    `AgentGraph`'s default `classifier_llm`) doesn't need `ANTHROPIC_API_KEY`
    or a live network call.
    """

    import langchain_anthropic

    mock_cls = MagicMock(return_value=MagicMock(ainvoke=AsyncMock(return_value="mocked")))
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", mock_cls)
    return mock_cls


@pytest.fixture(scope="session")
def app_module():
    """Import `main` once for the whole session with Chroma/embedding-model
    construction short-circuited and REDIS_URL/DATABASE_URL/API_KEY unset,
    so hitting the FastAPI app in tests needs no real infra.

    `main.py` builds its VectorStore/RAGAgent/SQLAgent/etc. as module-level
    globals at import time, so these patches must be in place *before* the
    first `import main` -- callers must not import `main` themselves.
    """

    mp = pytest.MonkeyPatch()
    mp.delenv("REDIS_URL", raising=False)
    mp.delenv("DATABASE_URL", raising=False)
    mp.delenv("API_KEY", raising=False)
    _patch_chroma(mp)

    import main

    # `import main` just ran `load_dotenv()`, which -- unlike
    # monkeypatch.setenv -- writes straight into os.environ with nothing
    # to undo it later. If a developer's local .env configures a
    # non-default LLM_BACKEND/OLLAMA_MODEL/OLLAMA_BASE_URL (e.g. for
    # local Ollama use), that leaks into every test for the rest of this
    # session once this session-scoped fixture is first created,
    # regardless of what the shell environment looked like beforehand.
    # Strip them back out so build_llm()'s default ("anthropic" unless
    # a caller explicitly configures otherwise) is what tests actually
    # observe -- matching a machine with no .env overrides at all.
    mp.delenv("LLM_BACKEND", raising=False)
    mp.delenv("OLLAMA_MODEL", raising=False)
    mp.delenv("OLLAMA_BASE_URL", raising=False)

    yield main

    mp.undo()


@pytest.fixture
def sales_sql_agent():
    """Factory for a `SQLAgent` backed by a persistent in-memory sqlite
    'sales' table (id, region, amount), for tests that exercise the
    NL->SQL pipeline over real (if tiny) data without a live Postgres.

    A plain `sqlite:///:memory:` engine gets a fresh, empty database per
    connection, so this uses `StaticPool` to keep one connection (and thus
    the table) alive for the agent's lifetime.
    """

    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool

    from src.agents.sql_agent import SQLAgent

    def _make(llm=None) -> SQLAgent:
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE sales (id INTEGER, region TEXT, amount REAL)"))
            conn.execute(
                text(
                    "INSERT INTO sales (id, region, amount) VALUES "
                    "(1, 'East', 100.0), (2, 'West', 150.0), (3, 'East', 50.0)"
                )
            )
        agent = SQLAgent("sqlite:///:memory:", llm=llm)
        agent.engine = engine
        return agent

    return _make
