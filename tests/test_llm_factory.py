import sys
import types
from unittest.mock import MagicMock, patch

from src.llm_factory import backend_reachable, build_llm


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
    # Isolate from whatever OLLAMA_MODEL/OLLAMA_BASE_URL a developer's
    # local .env or shell happens to have set -- this test asserts the
    # hardcoded defaults, so ambient overrides must not leak in.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

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


# --- backend_reachable --------------------------------------------------


def test_backend_reachable_anthropic_is_always_reachable():
    """Only "ollama" is actually probed -- a cloud backend's own network/
    auth failures surface as a real exception at the call site, which
    isn't this function's job to pre-empt.
    """

    reachable, reason = backend_reachable("anthropic")
    assert reachable is True
    assert reason == ""


def test_backend_reachable_ollama_true_when_server_responds():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = MagicMock()
        reachable, reason = backend_reachable("ollama", "http://localhost:11434")

    assert reachable is True
    assert reason == ""


def test_backend_reachable_ollama_false_when_connection_fails():
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("connection refused")):
        reachable, reason = backend_reachable("ollama", "http://localhost:11434")

    assert reachable is False
    assert "unreachable" in reason
    assert "http://localhost:11434" in reason


def test_backend_reachable_defaults_to_env_var_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
        reachable, _ = backend_reachable()

    assert reachable is False
