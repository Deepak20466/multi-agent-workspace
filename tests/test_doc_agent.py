from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.doc_agent import DocAgent, extract_file_refs


def test_extract_file_refs_finds_supported_extensions():
    refs = extract_file_refs("Please summarize invoice.pdf and notes.txt, ignore image.gif")
    assert refs == ["invoice.pdf", "notes.txt"]


def test_extract_file_refs_finds_docx():
    refs = extract_file_refs("Summarize the attached report.docx for me")
    assert refs == ["report.docx"]


def test_answer_from_file_docx_resolves_via_query_and_returns_markdown_table(tmp_path, document_processor, make_docx):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    target = make_docx(
        upload_dir / "policy.docx",
        heading="Refund Policy",
        paragraphs=["Our refund policy allows returns within 30 days of purchase."],
    )

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    agent = DocAgent(document_processor=document_processor, reranker=reranker, upload_dir=upload_dir)
    response = agent.answer_from_file(None, "What does policy.docx say about refunds?")

    assert response.metadata["file_path"] == str(target)
    assert response.citations
    assert response.citations[0].source == str(target)


def test_resolve_file_path_prefers_explicit_file_path():
    agent = DocAgent(document_processor=MagicMock())
    assert agent.resolve_file_path("irrelevant query", file_path="explicit.pdf") == "explicit.pdf"


def test_resolve_file_path_raises_when_no_ref_found():
    agent = DocAgent(document_processor=MagicMock())
    with pytest.raises(FileNotFoundError):
        agent.resolve_file_path("no file mentioned here", file_path=None)


def test_resolve_file_path_finds_file_in_upload_dir(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    target = upload_dir / "report.pdf"
    target.write_text("dummy")

    agent = DocAgent(document_processor=MagicMock(), upload_dir=upload_dir)
    resolved = agent.resolve_file_path("please summarize report.pdf for me", file_path=None)

    assert resolved == str(target)


def test_resolve_file_path_uses_existing_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("local.pdf").write_text("dummy")

    agent = DocAgent(document_processor=MagicMock(), upload_dir=tmp_path / "uploads")
    resolved = agent.resolve_file_path("check local.pdf please", file_path=None)

    assert resolved == "local.pdf"


def test_answer_from_file_resolves_via_query_and_returns_markdown_table(tmp_path, document_processor):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    target = upload_dir / "policy.txt"
    target.write_text("Our refund policy allows returns within 30 days of purchase.")

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    agent = DocAgent(document_processor=document_processor, reranker=reranker, upload_dir=upload_dir)
    response = agent.answer_from_file(None, "What does policy.txt say about refunds?")

    assert response.metadata["file_path"] == str(target)
    assert "| # | Excerpt | Source |" in response.answer


# --- LLM wiring -------------------------------------------------------------
#
# Regression coverage for: DocAgent never actually reached a configured LLM.
# `main.py`/`src/mcp_server.py` constructed it with no `llm_backend` at all
# (unlike the working RAGAgent/SQLAgent), so `self.llm` stayed `None` and
# `_generate()` always took the raw-context stub path -- even when
# config.yaml configures a real backend (Ollama here). Separately,
# `_generate()` returned `self.llm.invoke(prompt)` unextracted: a real
# langchain chat model's `.invoke()` returns an `AIMessage`, not a string,
# which `format_answer_with_citations()` would crash on (`AIMessage` has no
# `.strip()`) the moment a real `llm` was ever wired in.


def test_doc_agent_builds_llm_from_configured_backend():
    """Mirrors RAGAgent/SQLAgent's contract: passing `llm_backend` (as
    main.py/mcp_server.py now do, from config.yaml) must actually build
    and wire an LLM instead of leaving `self.llm` as `None`.
    """

    sentinel_llm = MagicMock()
    with patch("src.llm_factory.build_llm", return_value=sentinel_llm) as mock_build:
        agent = DocAgent(
            document_processor=MagicMock(),
            llm_backend="ollama",
            ollama_model="qwen2.5:0.5b",
            ollama_base_url="http://localhost:11434",
        )

    assert agent.llm is sentinel_llm
    _, kwargs = mock_build.call_args
    assert kwargs["backend"] == "ollama"
    assert kwargs["ollama_model"] == "qwen2.5:0.5b"
    assert kwargs["ollama_base_url"] == "http://localhost:11434"


def test_doc_agent_without_backend_keeps_stub_behavior():
    """Callers that don't opt into a backend (tests/eval, the pre-fix
    call sites) must keep the existing llm=None stub-answer contract --
    this is what the rest of test_doc_agent.py already relies on.
    """

    agent = DocAgent(document_processor=MagicMock())
    assert agent.llm is None


def test_generate_extracts_plain_text_from_message_like_llm_response():
    """A real langchain ChatModel's `.invoke()` returns an `AIMessage`
    (a `.content` attribute), not a plain string. `_generate` must
    unwrap it -- returning the message object itself previously crashed
    `format_answer_with_citations` downstream (`AIMessage` has no
    `.strip()`).
    """

    from langchain_core.messages import AIMessage

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="Refunds are allowed within 30 days [1].")

    agent = DocAgent(document_processor=MagicMock(), llm=fake_llm)
    result = agent._generate("refund policy?", ["Our refund policy allows returns within 30 days."])

    assert result == "Refunds are allowed within 30 days [1]."


def test_answer_from_file_with_configured_llm_produces_genuine_generated_answer(tmp_path, document_processor):
    """End-to-end (mocked LLM): when a backend is wired, the final answer
    must be the LLM's own generated text, not the raw-context stub
    ("From the uploaded document: ...") -- and citations must still be
    attached.
    """

    from langchain_core.messages import AIMessage

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    target = upload_dir / "policy.txt"
    target.write_text("Our refund policy allows returns within 30 days of purchase.")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="You have 30 days to request a refund [1].")

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    agent = DocAgent(document_processor=document_processor, reranker=reranker, upload_dir=upload_dir, llm=fake_llm)
    response = agent.answer_from_file(None, "What does policy.txt say about refunds?")

    fake_llm.invoke.assert_called_once()
    assert "You have 30 days to request a refund [1]." in response.answer
    assert "From the uploaded document:" not in response.answer
    assert response.citations
    assert response.citations[0].source == str(target)


def test_answer_from_file_missing_explicit_file_raises_cleanly(document_processor):
    """An explicit `file_path` skips `resolve_file_path`'s upload-dir
    lookup entirely, so a nonexistent path must still fail cleanly (a
    plain `FileNotFoundError` from the underlying loader) rather than
    some new crash introduced by the LLM-wiring change -- with or
    without an LLM configured.
    """

    agent = DocAgent(document_processor=document_processor)
    with pytest.raises(FileNotFoundError):
        agent.answer_from_file("definitely-does-not-exist.txt", "what does it say?")

    agent_with_llm = DocAgent(document_processor=document_processor, llm=MagicMock())
    with pytest.raises(FileNotFoundError):
        agent_with_llm.answer_from_file("also-missing.pdf", "what does it say?")


def test_answer_from_file_unresolvable_query_reference_raises_file_not_found():
    agent = DocAgent(document_processor=MagicMock(), llm=MagicMock())
    with pytest.raises(FileNotFoundError):
        agent.answer_from_file(None, "no file mentioned anywhere in this query")


@pytest.mark.integration
def test_answer_from_file_real_ollama_generates_genuine_answer(tmp_path, document_processor, ollama_available):
    """Real, unmocked local-Ollama generation (skipped if Ollama isn't
    reachable): proves DocAgent, wired the same way main.py wires it
    from config.yaml, actually calls out to the configured model instead
    of silently falling back to the raw-context stub.
    """

    if not ollama_available:
        pytest.skip("Ollama not reachable at localhost:11434")

    from tests.conftest import OLLAMA_BASE_URL, OLLAMA_MODEL

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    target = upload_dir / "policy.txt"
    target.write_text(
        "Our refund policy allows returns within 30 days of purchase. "
        "Refunds are processed within 5 business days of receiving the item."
    )

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    agent = DocAgent(
        document_processor=document_processor,
        reranker=reranker,
        upload_dir=upload_dir,
        llm_backend="ollama",
        ollama_model=OLLAMA_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
    )
    response = agent.answer_from_file(None, "According to policy.txt, how many days do customers have to return an item?")

    # A real model's exact wording can't be pinned down, but it must not
    # be the raw-context stub template, and it must produce *some* new
    # text rather than an empty/failed generation.
    assert "From the uploaded document:" not in response.answer
    assert len(response.answer.strip()) > 0
    assert response.citations
