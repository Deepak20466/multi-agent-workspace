"""Shared chat-LLM factory: Claude (Anthropic) by default, or a local
Ollama model when explicitly configured -- for fully offline execution,
or as a fallback when no ANTHROPIC_API_KEY is available.

Backend selection is explicit (constructor/config param, or the
LLM_BACKEND env var), never inferred from whether an API key happens to
be set -- silently switching backends based on environment state would
make failures (or successes) depend on ambient config that's hard to
see from the call site.
"""

from __future__ import annotations

import os
import urllib.request

DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def build_llm(
    anthropic_model: str,
    backend: str | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    **anthropic_kwargs,
):
    """Build a chat LLM per `backend` ("anthropic" default, or "ollama").

    `backend` falls back to the `LLM_BACKEND` env var, then "anthropic".
    `max_tokens` is applied on both paths (mapped to Ollama's
    `num_predict`) -- previously only reached the Anthropic path via
    `**anthropic_kwargs`, silently no-op'ing on Ollama, which let local
    CPU generations run unbounded. `**anthropic_kwargs` for anything
    else stays Anthropic-only. `langchain-ollama` is imported lazily so
    it isn't a hard dependency for the (default) Anthropic-only path.
    """

    backend = (backend or os.getenv("LLM_BACKEND") or "anthropic").lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            temperature=temperature,
            num_predict=max_tokens,
        )

    from langchain_anthropic import ChatAnthropic

    if max_tokens is not None:
        anthropic_kwargs.setdefault("max_tokens", max_tokens)
    return ChatAnthropic(model=anthropic_model, temperature=temperature, **anthropic_kwargs)


def backend_reachable(
    backend: str | None = None,
    ollama_base_url: str | None = None,
    timeout: float = 2.0,
) -> tuple[bool, str]:
    """Lightweight reachability probe for the configured chat backend.

    Returns `(True, "")` when the backend is usable, or `(False, reason)`
    with a short human-readable explanation otherwise. Only "ollama" is
    actually probed (a cheap local HTTP call) -- a cloud backend is
    assumed reachable here since network/auth failures for it surface
    immediately as a real exception from the call site anyway, and this
    function's only job is catching the specific "local runtime simply
    isn't there" case (e.g. no Ollama server in CI) *before* a caller
    does real work that would otherwise fail confusingly deep inside a
    retry loop or a multi-step pipeline.

    Callers that need this distinction: eval/run_ragas_eval.py uses it
    to tell "genuinely skipped, no local LLM available" apart from "ran
    and failed for a real reason" (see its module docstring); tests use
    the identical check to skip real-Ollama integration tests instead of
    failing them on machines without Ollama running.
    """

    backend = (backend or os.getenv("LLM_BACKEND") or "anthropic").lower()
    if backend != "ollama":
        return True, ""

    base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout):
            return True, ""
    except OSError as exc:
        return False, f"Ollama backend configured but unreachable at {base_url} ({exc})"
