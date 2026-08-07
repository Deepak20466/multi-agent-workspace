# Multi-Agent Workspace v3.1

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-MultiAgent-green.svg)](https://github.com/langchain-ai/langgraph)
[![RAGAS](https://img.shields.io/badge/RAGAS-quality%20gated-brightgreen.svg)](#evaluation)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](#deployment)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)

Production-grade Retrieval-Augmented Generation platform built on a **LangGraph router** that classifies each request and dispatches it to a specialist **RAG / SQL / Doc / Web** agent. Hybrid dense+BM25 retrieval (RRF-fused), Cohere/cross-encoder reranking, AST-validated NL→SQL, OCR + table + Excel document intelligence, an MCP tool server, Presidio-based PII guardrails, exponential-backoff retry resilience, Redis response caching, and a RAGAS-gated evaluation pipeline are wired together end-to-end.

## Overview

Every query enters through a LangGraph `StateGraph` ("the brain"): input is screened for prompt injection and PII, a Claude Haiku classifier picks a route, and the corresponding agent answers with grounded, numbered citations traceable back to source chunks.

| Agent | Answers from | Key techniques |
|---|---|---|
| **RAG** | Indexed document corpus | Multi-query + HyDE expansion, hybrid dense/BM25 retrieval, RRF fusion, Cohere → flashrank → cross-encoder rerank cascade |
| **SQL** | Live Postgres database | NL→SQL, ambiguity/clarification check, `sqlglot` AST safety validation (SELECT-only), enforced row limits |
| **Doc** | A single attached/uploaded file | On-demand OCR, Excel, and PDF-table parsing — no pre-indexing required |
| **Web** | Live web search (Tavily) | Current-events / time-sensitive queries the local corpus can't answer |

Every agent's LLM calls (router classifier, NL→SQL generation) go through a shared backend factory (`src/llm_factory.py`) that defaults to Claude but can be pointed at a local Ollama model for offline execution — see [Local model backend](#local-model-backend-ollama).

Sessions are checkpointed in Redis (`thread_id=session_id`) so multi-turn conversations survive process restarts. Every response is exposed over a FastAPI HTTP API (JSON or SSE token/citation/chart streaming), a Click CLI, and an MCP stdio server for use from Claude Desktop or other MCP clients.

## Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| **Python 3.11+** | Running the app/CLI locally | See `setup.py` (`python_requires>=3.11`) |
| **Docker + Docker Compose** | One-command stack (app + Postgres + Redis + Chroma) | Recommended path — see `docker-compose.yml` |
| **Postgres** | SQL agent | Only if running services outside Docker; connection via `DATABASE_URL` |
| **Redis** | Session checkpointing, response cache, rate limiter | Degrades gracefully (in-memory fallback) if unreachable, but required for the FastAPI app's lifespan checkpointer |
| **Chroma** | Vector store for the RAG agent | Embedded/local mode via `CHROMA_PERSIST_DIR`, or a running Chroma server via `CHROMA_HOST`/`CHROMA_PORT` |
| **Anthropic API key** | Router classifier + generation (default LLM backend) | `ANTHROPIC_API_KEY`; not required if `LLM_BACKEND=ollama` |
| **Tesseract OCR** *(optional)* | Scanned PDF/image ingestion | `apt-get install tesseract-ocr` / `brew install tesseract` / [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) on Windows |
| **Ghostscript** *(optional)* | Camelot lattice-mode table extraction | Falls back to `pdfplumber`-only if absent |
| **Cohere / Tavily API keys** *(optional)* | Top-tier reranking / web agent | Cascade falls back to local rerankers; web agent requires `TAVILY_API_KEY` to function |
| **Ollama** *(optional)* | Fully offline LLM backend | `LLM_BACKEND=ollama` — see [Local model backend](#local-model-backend-ollama) |

## Quick Start

```bash
# spin up the app + Postgres + Redis + Chroma
docker-compose up -d

# index the sample document corpus
python main.py index --sources data/sample_documents

# ask a question through the router (auto-classified) or a forced agent
python main.py agent "plot sales by region" --agent sql
```

## Setup and Installation

Run locally without Docker:

```bash
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # fill in ANTHROPIC_API_KEY, COHERE_API_KEY, TAVILY_API_KEY, etc.
uvicorn main:app --reload
```

## Features (v3.1)

| Feature | Description |
|---|---|
| **Multi-Agent Router** | LangGraph `StateGraph` with a Claude Haiku classifier (`src/agents/router.py`); falls back to `rag` on an unrecognized classification. Prompt-injection and PII checks run ahead of routing. |
| **RAG Agent** | Multi-query expansion + HyDE (`src/query_expansion.py`), hybrid dense/BM25 retrieval fused via alpha-weighted Reciprocal Rank Fusion (`src/hybrid_retrieval.py`), Cohere → flashrank → local cross-encoder rerank cascade (`src/reranking.py`) — see [Benchmarks](#performance-benchmarks). |
| **SQL Agent** | Ambiguity/clarification check on vague questions ("top", "best", ...) before generation, NL→SQL against a live-introspected schema, validated through a `sqlglot` AST pass — only single read-only `SELECT` statements pass, row limit enforced by rewriting the AST (`src/agents/sql_agent.py`). |
| **Doc Agent** | Answers from a single file on demand (no pre-indexing): OCR for scanned PDFs/images, Excel via openpyxl/pandas, PDF table extraction via pdfplumber/camelot (`src/parsers/`, `src/document_processing.py`). |
| **Web Agent** | Tavily-backed live search with citation-grounded answers for time-sensitive queries (`src/agents/web_agent.py`). |
| **MCP Server** | Exposes `search_docs`, `query_sql`, `extract_document`, `web_search` as MCP tools over stdio, plus `calculator`/`python_repl`/`send_email` utility tools (`src/mcp_server.py`, `src/tools/`). |
| **Citations** | Every answer is grounded back to source chunks with quotes, page/sheet numbers, and inline `[n]` markers verified against the actual citation list (`src/citation.py`). |
| **Guardrails** | Presidio-based PII detection/anonymization on ingest and on every request, plus regex prompt-injection heuristics (`src/guardrails.py`). |
| **Local Model Backend** | `src/llm_factory.py` builds Claude by default, or a local Ollama model (`agents.llm_backend: ollama`) for offline execution — used by the router classifier and SQL agent. |
| **Retry Resilience** | Exponential backoff + jitter (tenacity) with a per-dependency circuit breaker around every LLM/DB/API call (`src/utils/retry_handler.py`). |
| **Response Cache** | Redis-backed cache keyed by normalized query hash, with an in-process LRU fallback if Redis is unreachable (`src/cache.py`). |
| **Telemetry** | Structured JSON logs (loguru) + Prometheus counters/histograms for latency and success/error rate per route (`src/telemetry.py`). |
| **RAGAS Eval Gate** | Faithfulness / answer-relevancy / context-precision / context-recall scoring blocks CI merges on regression (`eval/`, `.github/workflows/ragas_eval.yml`). |
| **Extended Eval Metrics** | Retrieval precision/recall, router tool-selection accuracy, and SQL AST guardrail pass rate — tracked and reported (not yet gated, see `DECISIONS.md`) in `eval/metrics.py`. |

### Architecture

```
                         ┌────────────────────────────────────────────┐
 User ──HTTP/SSE──▶ FastAPI ──▶  LangGraph Router (Claude Haiku /     │
                         │        Ollama, via src/llm_factory.py)     │
                         │  PII + prompt-injection guard, classify    │
                         │  session checkpointed in Redis             │
                         └──────────────────┬───────────────────────┬─┘
                                             │                       │
                    ┌────────────┬──────────┼──────────┬────────────┘
                    ▼            ▼          ▼           ▼
                ┌───────┐   ┌────────┐  ┌───────┐   ┌────────┐
                │  RAG  │   │  SQL   │  │  Doc   │   │  Web   │
                │ agent │   │ agent  │  │ agent  │   │ agent  │
                └───┬───┘   └───┬────┘  └───┬────┘   └───┬────┘
                    │           │           │            │
      multi-query+HyDE   ambiguity check   OCR/Excel/     Tavily
      hybrid RRF fuse     -> NL->SQL ->    table parse    search
      Cohere→flashrank→   AST guard        (on demand)
      cross-encoder        + row cap
      rerank cascade
                    │           │           │            │
                    └─────┬─────┴─────┬─────┴─────┬──────┘
                          ▼           ▼           ▼
                    ┌────────────────────────────────┐
                    │   Citation builder + verifier   │
                    │  (grounds answer to source[n])  │
                    └────────────────┬─────────────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │  Aggregator ──▶ SSE stream out   │
                    │  (token / citations / chart /    │
                    │   done events)                   │
                    └────────────────────────────────┘
```

Retrieval detail (RAG agent, `src/hybrid_retrieval.py` + `src/reranking.py`):

```
 query ──▶ multi-query + HyDE expansion ──▶ [query, variant1, variant2, hyde_passage]
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
                     dense (Chroma vector)                          sparse (BM25 in-memory)
                              │                                               │
                              └───────────────────┬───────────────────────────┘
                                                   ▼
                                  alpha-weighted Reciprocal Rank Fusion
                                          (~61% top-k accuracy)
                                                   ▼
                          ┌────────────────────────────────────────────┐
                          │  rerank cascade (first available tier):    │
                          │  1. Cohere Rerank API   (COHERE_API_KEY)   │
                          │  2. flashrank (local ONNX cross-encoder)   │
                          │  3. sentence-transformers CrossEncoder     │
                          │             (~85% top-k accuracy)          │
                          └────────────────────────────────────────────┘
```

## Usage

### CLI

```bash
python main.py index --sources data/sample_documents      # index docs into Chroma + BM25
python main.py agent "How many orders were completed?"     # auto-routed
python main.py agent "summarize invoice.pdf" --agent doc --file-path data/uploads/invoice.pdf
python main.py chat --session-id demo                       # interactive REPL with memory
python main.py mcp-serve                                    # start the MCP stdio server
python main.py eval --type all                               # RAGAS + SQL/doc smoke eval
python main.py eval-agent all                                 # router classification accuracy
```

### HTTP API

```bash
curl -X POST localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "What is our refund policy?", "agent_type": "auto", "user_id": "u1", "session_id": "s1"}'

# streaming (SSE: metadata -> token* -> citations -> chart? -> done)
curl -N -X POST localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "plot revenue by region", "agent_type": "sql", "stream": true}'
```

`GET /health` — liveness check. `POST /ingest` — upload + index a file. `POST /query` — legacy non-versioned single-shot endpoint.

### MCP (Claude Desktop / MCP clients)

```bash
python main.py mcp-serve
```

Exposes `search_docs(query)`, `query_sql(question)`, `extract_document(file_path, query)`, `web_search(query)` as tools, each wrapped with the same PII/prompt-injection guardrails the HTTP router applies.

### Python SDK

```python
from src.agents.router import AgentGraph
from src.agents.rag_agent import RAGAgent
from src.hybrid_retrieval import HybridRetriever
from src.vectorstore import VectorStore

retriever = HybridRetriever(VectorStore())
graph = AgentGraph(rag_agent=RAGAgent(retriever=retriever))
result = await graph.run("What does our refund policy say?", user_id="u1", session_id="s1")
print(result.answer, result.citations)
```

## Project Structure

```
.
├── main.py                    # FastAPI app + Click CLI entrypoint
├── config.yaml                # runtime configuration (retrieval, sql, doc, mcp, cache, ...)
├── docker-compose.yml         # app + Postgres + Redis + Chroma
├── src/
│   ├── agents/                # router.py, rag_agent.py, sql_agent.py, doc_agent.py, web_agent.py
│   ├── parsers/                # excel_parser.py, ocr_parser.py, table_parser.py
│   ├── tools/                  # calculator.py, python_repl.py, send_email.py (MCP tools)
│   ├── utils/                  # retry_handler.py, schemas.py
│   ├── hybrid_retrieval.py     # dense + BM25, RRF fusion
│   ├── reranking.py            # Cohere -> flashrank -> cross-encoder rerank cascade
│   ├── query_expansion.py      # multi-query + HyDE
│   ├── citation.py             # grounded citation builder + marker verification
│   ├── guardrails.py           # Presidio PII + prompt-injection detection
│   ├── document_processing.py  # ingest pipeline (load -> OCR fallback -> redact -> chunk)
│   ├── vectorstore.py          # Chroma wrapper
│   ├── cache.py                # Redis + in-memory LRU response cache
│   ├── middleware.py           # API key auth + Redis token-bucket rate limiter
│   ├── telemetry.py            # structured logging + Prometheus metrics
│   ├── mcp_server.py           # MCP tool server
│   ├── llm_factory.py          # Claude / Ollama backend factory
│   └── config.py               # typed config.yaml loader
├── eval/                      # generate_testset.py, run_ragas_eval.py, check_gates.py, seed_sql_db.py, metrics.py
├── tests/                     # test modules (agents, retrieval, parsers, citations, e2e, mcp, eval metrics, llm factory, ...)
├── data/                      # init.sql (Postgres seed schema), sample_documents/
├── DECISIONS.md                # architectural trade-off log
└── .github/workflows/ragas_eval.yml   # CI: unit tests -> eval testsets -> RAGAS -> quality gates
```

## Configuration

Runtime knobs live in `config.yaml` (typed/validated by `src/config.py`); secrets and deployment-specific values stay in environment variables (`.env`, see `.env.example`).

### Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude models (router classifier, generation, NL→SQL) |
| `COHERE_API_KEY` | Cohere Rerank (optional — first tier of the rerank cascade, see [Benchmarks](#performance-benchmarks)) |
| `LLM_BACKEND` | `anthropic` (default) or `ollama` — see [Local model backend](#local-model-backend-ollama) |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | Model name + server URL when `LLM_BACKEND=ollama` |
| `TAVILY_API_KEY` | Web agent search provider |
| `DATABASE_URL` | Postgres connection string for the SQL agent |
| `REDIS_URL` | Session checkpointing, response cache, rate limiter |
| `CHROMA_PERSIST_DIR` / `CHROMA_HOST` / `CHROMA_PORT` | Vector store |
| `API_KEY` | Required `X-API-Key` header value (auth skipped if unset) |
| `OCR_LANG` / `OCR_DPI` | Tesseract OCR language + render DPI |
| `MASK_PII_ON_INGEST` | Toggle Presidio redaction on document ingest |
| `MAX_API_RETRIES` / `BACKOFF_FACTOR` / `MAX_BACKOFF_S` | Retry/backoff tuning |
| `ENABLE_CHARTS` | Toggle SSE `chart` events for SQL result rows |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS` | `send_email` MCP tool (disabled by default) |

### `config.yaml` sections

| Section | Covers |
|---|---|
| `app` | Host/port/version |
| `agents` | Enabled agents, router/RAG model IDs, session memory backend, `llm_backend`/`ollama_model`/`ollama_base_url` |
| `resilience` | Max retries, backoff base/cap, retryable HTTP codes |
| `retrieval` | `top_k`, RRF `alpha`/`k`, multi-query/HyDE toggles, `use_rerank`/`rerank_top_k`, `use_flashrank`/`flashrank_model` |
| `sql` | Max rows, chart toggle, AST-readonly enforcement, `check_ambiguity` |
| `doc_intelligence` | OCR/table/Excel toggles, PII redaction |
| `web` | Provider, max results, timeout |
| `mcp` | Enabled, transport (stdio) |
| `tools` | Calculator/python_repl/email toggles, timeout |
| `cache` | Enabled, TTL |
| `telemetry` | Enabled, Prometheus port |

### Local model backend (Ollama)

By default every agent's LLM calls go through `src/llm_factory.py::build_llm()` using Claude (`agents.llm_backend: anthropic`). Set `agents.llm_backend: ollama` in `config.yaml` (or `LLM_BACKEND=ollama` in the environment) to route the router classifier and SQL agent through a local [Ollama](https://ollama.com) server instead — no code changes, no Anthropic API key required:

```bash
ollama pull llama3 && ollama serve   # in one terminal
LLM_BACKEND=ollama OLLAMA_MODEL=llama3 python main.py agent "how many orders were completed?" --agent sql
```

The backend is selected explicitly (config/env), never auto-detected from whether an API key is present — see `DECISIONS.md` §5 for why.

## Evaluation

```bash
python main.py eval --type all        # RAGAS (rag) + row-overlap accuracy (sql) + LLM-judge (doc)
python main.py eval-agent all         # router classification accuracy against labeled queries
```

CI (`.github/workflows/ragas_eval.yml`) runs unit tests, seeds the eval Postgres DB, generates RAG/SQL/doc testsets, scores them, and enforces these gates (`eval/check_gates.py`) before merge — a section is reported `SKIP` rather than `FAIL` when its dependency isn't configured (e.g. no `DATABASE_URL`):

| Gate | Threshold |
|---|---|
| `rag.faithfulness` | ≥ 0.75 |
| `rag.answer_relevancy` | ≥ 0.70 |
| `rag.context_precision` | ≥ 0.70 |
| `rag.context_recall` | ≥ 0.60 |
| `sql.accuracy` | ≥ 0.80 |
| `doc.accuracy` | ≥ 0.80 |

`eval/results_v3.json` also carries three metrics that are tracked and reported but **not yet gated** in CI (see `DECISIONS.md` §6 for why): `retrieval.precision`/`retrieval.recall` (needs a `relevant_chunk_ids`-labeled testset — skipped until one exists), and `sql_guardrail.pass_rate` (always runs; exercises the AST guardrail against both legitimate and adversarial SQL, including writes hidden inside a CTE). Router tool-selection accuracy is reported by `python main.py eval-agent all --json`, backed by `eval/metrics.py::router_tool_selection_accuracy`.

## Performance Benchmarks

Indicative figures from local/CI runs — actual numbers depend on corpus size, model choice, and hardware.

| Stage | Typical latency | Notes |
|---|---|---|
| Router classification | ~50ms p95 | Claude Haiku, 10-token response cap |
| RAG (full pipeline) | ~1.5s | Multi-query + HyDE + hybrid retrieval + rerank + generation |
| SQL agent | ~1.2s | Ambiguity check + NL→SQL generation + AST validation + query execution |
| OCR (scanned PDF) | ~3s / page | Tesseract at 300 DPI |
| Response cache hit | ~35% hit rate in demo workload | Cuts full-pipeline latency by ~60% |
| Citation/hallucination check | +~200ms | Marker verification against retrieved chunk set |

### Retrieval accuracy: reranker impact

Top-k retrieval accuracy (fraction of queries where a truly relevant chunk lands in the final top-k) measured on the benchmark query set, before vs. after adding the cross-encoder rerank stage on top of RRF-fused hybrid retrieval:

| Configuration | Top-k accuracy | Added latency |
|---|---|---|
| RRF fusion only (no rerank) | ~61% | — |
| + flashrank local cross-encoder | ~80% | +~50–100ms |
| + Cohere Rerank API | ~85% | +~150–300ms (network) |

RRF's fused score reflects rank position across the dense/sparse lists, not real query-document semantic match — a reranker that scores each candidate directly against the query text closes most of that gap. `retrieval.use_rerank: false` in `config.yaml` reverts to the RRF-only baseline (e.g. for latency-sensitive deployments willing to trade accuracy for speed).

## Deployment

```bash
docker-compose up -d          # app + Postgres + Redis + Chroma
python main.py mcp-serve      # MCP stdio server (run alongside, or standalone for Claude Desktop)
```

`docker-compose.yml` builds the app from `Dockerfile`, wires `DATABASE_URL`/`REDIS_URL`/`CHROMA_PERSIST_DIR` for the containerized services, and persists `chroma_data/`, `logs/`, and `data/` as volumes. Postgres auto-seeds from `data/init.sql` on first boot.

## Security Architecture

- **AST-validated SQL** — NL→SQL output is parsed with `sqlglot` (not regex-matched); only a single `SELECT` with no `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`CREATE`/`GRANT` node anywhere in the tree is allowed, and the row limit is enforced by rewriting the AST itself so it can't be bypassed post-validation (`src/agents/sql_agent.py`).
- **PII redaction** — Presidio `AnalyzerEngine`/`AnonymizerEngine` detect and mask PERSON/EMAIL/PHONE/CREDIT_CARD/SSN/IBAN/IP entities on document ingest and on every request in/out of the router (`src/guardrails.py`).
- **Prompt-injection detection** — regex heuristics ("ignore previous instructions", "you are now in DAN mode", "reveal your system prompt", ...) block a request before it reaches any agent.
- **Sandboxed Python REPL** — MCP `python_repl` tool runs in a spawned subprocess with a hard 5s timeout and a restricted builtins set; imports are allowlisted (`pandas`, `numpy`, `math`, `plotly`, `json`) via a static AST check before execution, so `os`, `sys`, and `subprocess` are rejected up front (best-effort sandbox, not a hard security boundary).
- **Exponential backoff + circuit breaker** — every LLM/DB/external API call is wrapped in tenacity-based retry with jitter and a per-dependency circuit breaker that stops hammering a dead dependency (`src/utils/retry_handler.py`).
- **API key auth + rate limiting** — `X-API-Key` header enforced by `APIKeyMiddleware` (skipped if `API_KEY` unset, for local dev); Redis token-bucket rate limiter at 60 req/min per `user_id`, fails open if Redis is unreachable rather than taking the API down (`src/middleware.py`).

## Troubleshooting / FAQ

**Q: `TesseractNotFoundError` when indexing scanned PDFs/images.**
A: Install the Tesseract OCR binary (not just `pytesseract`, which is a wrapper): `apt-get install tesseract-ocr` (Linux) or `brew install tesseract` (macOS); on Windows install the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it's on `PATH`.

**Q: Camelot/table extraction fails with a Ghostscript error.**
A: `camelot-py[cv]`'s lattice mode shells out to Ghostscript. Install it (`apt-get install ghostscript` / `brew install ghostscript`); table extraction still works via `pdfplumber` alone if Ghostscript is unavailable.

**Q: I'm getting HTTP 429 `rate limit exceeded` from `/api/v1/agent`.**
A: The default limiter allows 60 requests/min per `user_id` (`src/middleware.py`). Either space out requests, pass distinct `user_id`s, or raise `capacity` where `RateLimiter` is constructed in `main.py`.

**Q: Redis connection errors on startup.**
A: `REDIS_URL` is required for the FastAPI app's session checkpointer (`main.py` lifespan handler raises if unset). The CLI degrades gracefully to an in-memory session instead. Response caching and rate limiting also degrade gracefully (fail open / in-memory LRU) if Redis is merely unreachable rather than unconfigured.

**Q: Chroma connection refused.**
A: Confirm `CHROMA_HOST`/`CHROMA_PORT` (or `CHROMA_PERSIST_DIR` for local/embedded mode) match how the Chroma container is exposed — `docker-compose.yml` maps it to host port `8001`.

**Q: The SQL agent says "not configured".**
A: Set `DATABASE_URL` (env var or `sql.database_url` in `config.yaml`, which reads `${DATABASE_URL}`). Without it, `_sql_agent` stays `None` and SQL-routed queries return an "agent not configured" response instead of erroring.

**Q: Reranking silently falls back to a local model.**
A: `COHERE_API_KEY` unset, or the Cohere API call failed — `Reranker.arerank` catches any exception from the Cohere path and falls through the cascade: `flashrank` (if `retrieval.use_flashrank` is enabled) then the `sentence-transformers` cross-encoder, so the pipeline doesn't break. Check logs for `"cohere rerank failed"` / `"flashrank rerank failed"` to see which tier actually served the request.

**Q: The SQL agent asked a clarifying question instead of answering.**
A: Your question matched a vague superlative ("top", "best", "most", ...) without a concrete row count *and* a named metric/column — e.g. "show top customers" doesn't say top-by-what or how many. The response has `metadata.needs_clarification: true` and a `clarification_question`; rephrase with both (e.g. "top 10 customers by revenue"), or set `sql.check_ambiguity: false` in `config.yaml` to disable the check entirely.

**Q: How do I run fully offline, without any Anthropic/Cohere API keys?**
A: Set `agents.llm_backend: ollama` in `config.yaml` (or `LLM_BACKEND=ollama`) with a local [Ollama](https://ollama.com) server running, and leave `COHERE_API_KEY` unset so reranking uses the local flashrank/cross-encoder tier. See [Local model backend](#local-model-backend-ollama).

**Q: RAGAS eval fails to import in CI/locally.**
A: `eval/run_ragas_eval.py` shims a known upstream break (`ragas` importing a `langchain_community.chat_models.vertexai` module that newer `langchain-community` no longer ships) — if you see a different import error, check your `ragas`/`langchain-community` version pinning.

**Q: Why did my query get blocked with "potential prompt injection detected"?**
A: The router's regex heuristics flagged phrasing like "ignore previous instructions" or "reveal your system prompt" (`src/guardrails.py:_INJECTION_PATTERNS`). These are heuristic, not ML-based, and can false-positive on legitimate meta-questions about the system.

**Q: How do I run everything end-to-end from a clean checkout?**
A:
```bash
docker-compose up -d
python main.py index --sources data/sample_documents
pytest tests/ -v
python eval/run_ragas_eval.py
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## License

MIT
