from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.doc_agent import DocAgent, extract_file_refs


def test_extract_file_refs_finds_supported_extensions():
    refs = extract_file_refs("Please summarize invoice.pdf and notes.txt, ignore image.gif")
    assert refs == ["invoice.pdf", "notes.txt"]


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
