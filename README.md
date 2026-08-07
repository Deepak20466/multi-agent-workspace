# Multi-Agent Workspace v3.1

Production RAG + LangGraph router with SQL/Doc/Web agents, hybrid
retrieval (dense + BM25 fused via RRF), cross-encoder reranking,
grounded citations, PII guardrails, response caching, and retry
resilience throughout.

## Features

- **Router** (`src/agents/router.py`) — LangGraph `StateGraph` that guards input (PII + prompt-injection), classifies intent, and dispatches to the RAG, SQL, Doc, or Web agent.
- **Hybrid retrieval + RRF** (`src/hybrid_retrieval.py`) — Chroma vector search fused with an in-memory BM25 index via Reciprocal Rank Fusion.
- **Reranking** (`src/reranking.py`) — cross-encoder rerank of the fused candidate set.
- **Citations** (`src/citation.py`) — every answer is grounded back to source chunks with quotes and page numbers.
- **SQL agent** (`src/agents/sql_agent.py`) — NL→SQL validated through a `sqlglot` AST pass: only single read-only `SELECT` statements are allowed, with an enforced row limit.
- **Parsers** (`src/parsers/`) — Excel (openpyxl/pandas), OCR (pytesseract), and PDF table extraction (pdfplumber, camelot fallback).
- **MCP** (`src/tools/__init__.py:build_mcp_server`) — calculator/python_repl/send_email tools exposed over MCP.
- **Guardrails** (`src/guardrails.py`) — Presidio-based PII detection/anonymization plus prompt-injection heuristics.
- **Cache** (`src/cache.py`) — Redis-backed response cache with in-memory LRU fallback.
- **Retry resilience** (`src/utils/retry_handler.py`) — exponential backoff + jitter and a per-dependency circuit breaker.
- **Telemetry** (`src/telemetry.py`) — structured JSON logs + Prometheus metrics.
- **Eval gate** (`eval/`, `.github/workflows/ragas_eval.yml`) — RAGAS faithfulness/answer-relevancy/context-precision scoring blocks merges on regression.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
cp .env.example .env
uvicorn main:app --reload
```

Or via Docker Compose (spins up Postgres + Redis too):

```bash
docker-compose up --build
```

## Running tests

```bash
pytest tests/ -v --cov=src
```

## Running the RAGAS eval gate locally

```bash
python eval/generate_testset.py
python eval/run_ragas_eval.py
python eval/check_gates.py
```

## Project layout

See `src/`, `eval/`, `tests/` for implementation, evaluation, and test code respectively. `data/init.sql` seeds the Postgres schema the SQL agent queries against; `data/sample_documents/` seeds the RAG corpus for the eval testset generator.
