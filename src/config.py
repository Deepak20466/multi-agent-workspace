"""Typed loader for config.yaml.

Most of these knobs already have env-var-driven defaults scattered
across src/ (see .env.example / retry_handler.py / cache.py etc.) —
this module exists to give the CLI (and anything else that wants it)
one validated, typed object instead of grepping for `os.getenv()`
calls, and to document every knob in one place. `${VAR}` placeholders
(e.g. sql.database_url) are expanded from the environment at load time.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = "config.yaml"

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(text: str) -> str:
    """Expand `${VAR}` placeholders from the environment. Unlike
    `os.path.expandvars`, an unset variable becomes an empty string
    rather than being left as the literal `${VAR}` text -- so e.g.
    sql.database_url safely falls back to "" (falsy) when DATABASE_URL
    isn't set, instead of an unusable literal placeholder string.
    """

    return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), text)


class AppSettings(BaseModel):
    name: str = "multi-agent-workspace"
    version: str = "3.1"
    host: str = "0.0.0.0"
    port: int = 8000


class AgentsSettings(BaseModel):
    enabled: List[str] = Field(default_factory=lambda: ["rag", "sql", "doc", "web"])
    router_model: str = "claude-3-haiku-20240307"
    rag_model: str = "claude-3-5-sonnet-20240620"
    max_iterations: int = 5
    memory: str = "redis"
    llm_backend: str = "anthropic"
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"


class ResilienceSettings(BaseModel):
    max_retries: int = 5
    backoff: float = 1
    max: float = 10
    retry_on: List[int] = Field(default_factory=lambda: [429, 503, 529])


class RetrievalSettings(BaseModel):
    top_k: int = 10
    alpha: float = 0.5
    rrf_k: int = 60
    use_multi_query: bool = True
    use_hyde: bool = True
    use_rerank: bool = True
    rerank_model: str = "rerank-english-v3.0"
    rerank_top_k: int = 5
    use_flashrank: bool = True
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"


class SqlSettings(BaseModel):
    database_url: str = ""
    max_rows: int = 1000
    enable_charts: bool = True
    allowed_tables: List[str] = Field(default_factory=list)
    ast_enforce_readonly: bool = True
    check_ambiguity: bool = True

    @field_validator("database_url", mode="before")
    @classmethod
    def _none_to_empty(cls, v: object) -> object:
        # `database_url: ${DATABASE_URL}` with DATABASE_URL unset expands
        # to an empty scalar, which YAML parses as null rather than "".
        return "" if v is None else v


class DocIntelligenceSettings(BaseModel):
    enable_ocr: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    enable_tables: bool = True
    enable_excel: bool = True
    redact_pii: bool = True


class WebSettings(BaseModel):
    provider: str = "tavily"
    max_results: int = 3
    timeout_s: int = 10


class McpSettings(BaseModel):
    enabled: bool = True
    transport: str = "stdio"


class ToolsSettings(BaseModel):
    enable_calculator: bool = True
    enable_python_repl: bool = True
    enable_email: bool = False
    timeout_s: int = 5


class CacheSettings(BaseModel):
    enabled: bool = True
    ttl: int = 3600


class TelemetrySettings(BaseModel):
    enabled: bool = True
    prometheus_port: int = 9090


class AppConfig(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    sql: SqlSettings = Field(default_factory=SqlSettings)
    doc_intelligence: DocIntelligenceSettings = Field(default_factory=DocIntelligenceSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


@functools.lru_cache(maxsize=None)
def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate `path` (default `config.yaml`), expanding
    `${VAR}` placeholders against the environment first. Results are
    cached per path; call `load_config.cache_clear()` to force a
    re-read (e.g. in tests, or after editing config.yaml in a long-lived
    process).
    """

    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    raw = config_path.read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw)
    data = yaml.safe_load(expanded) or {}
    return AppConfig.model_validate(data)
