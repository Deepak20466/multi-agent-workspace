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

DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def build_llm(
    anthropic_model: str,
    backend: str | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    temperature: float = 0,
    **anthropic_kwargs,
):
    """Build a chat LLM per `backend` ("anthropic" default, or "ollama").

    `backend` falls back to the `LLM_BACKEND` env var, then "anthropic".
    `**anthropic_kwargs` (e.g. `max_tokens`) are only applied on the
    Anthropic path. `langchain-ollama` is imported lazily so it isn't a
    hard dependency for the (default) Anthropic-only path.
    """

    backend = (backend or os.getenv("LLM_BACKEND") or "anthropic").lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            temperature=temperature,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=anthropic_model, temperature=temperature, **anthropic_kwargs)
