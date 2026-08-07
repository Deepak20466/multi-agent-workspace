import sys
import types
from unittest.mock import MagicMock

from src.llm_factory import build_llm


def test_build_llm_defaults_to_anthropic(monkeypatch):
    import langchain_anthropic

    mock_cls = MagicMock()
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", mock_cls)
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    llm = build_llm("claude-haiku-4-5")

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "claude-haiku-4-5"
    assert llm is mock_cls.return_value


def _stub_ollama_module(monkeypatch) -> MagicMock:
    """`langchain-ollama` isn't installed in this environment (it's an
    optional dependency, imported lazily) -- stub it in sys.modules so
    `from langchain_ollama import ChatOllama` resolves without the real
    package.
    """

    fake_module = types.ModuleType("langchain_ollama")
    mock_cls = MagicMock()
    fake_module.ChatOllama = mock_cls
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)
    return mock_cls


def test_build_llm_respects_llm_backend_env_var(monkeypatch):
    mock_cls = _stub_ollama_module(monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    llm = build_llm("claude-haiku-4-5")

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "llama3"
    assert kwargs["base_url"] == "http://localhost:11434"
    assert llm is mock_cls.return_value


def test_build_llm_explicit_ollama_backend_overrides_model_and_base_url(monkeypatch):
    mock_cls = _stub_ollama_module(monkeypatch)
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    build_llm("claude-haiku-4-5", backend="ollama", ollama_model="mistral", ollama_base_url="http://ollama:11434")

    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "mistral"
    assert kwargs["base_url"] == "http://ollama:11434"


def test_build_llm_anthropic_kwargs_only_apply_on_anthropic_path(monkeypatch):
    import langchain_anthropic

    mock_cls = MagicMock()
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", mock_cls)
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    build_llm("claude-haiku-4-5", max_tokens=10)

    _, kwargs = mock_cls.call_args
    assert kwargs["max_tokens"] == 10
