"""FastAPI entrypoint for the multi-agent workspace.

Wires VectorStore -> HybridRetriever -> RAGAgent, plus the optional
SQL/Doc/Web agents, into the LangGraph AgentRouter and exposes it over
HTTP for the demo UI / integration tests.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

load_dotenv()

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.router import AgentRouter
from src.agents.sql_agent import SQLAgent
from src.document_processing import DocumentProcessor
from src.hybrid_retrieval import HybridRetriever
from src.vectorstore import VectorStore

app = FastAPI(title="Multi-Agent Workspace", version="3.1.0")

_vector_store = VectorStore()
_retriever = HybridRetriever(_vector_store)
_rag_agent = RAGAgent(retriever=_retriever)
_document_processor = DocumentProcessor()
_doc_agent = DocAgent(document_processor=_document_processor)

_database_url = os.getenv("DATABASE_URL")
_sql_agent = SQLAgent(_database_url) if _database_url else None

_router = AgentRouter(rag_agent=_rag_agent, sql_agent=_sql_agent, doc_agent=_doc_agent)


class QueryRequest(BaseModel):
    query: str
    file_path: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "3.1.0"}


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    response = await _router.run(request.query, file_path=request.file_path)
    return response.model_dump(mode="json")


@app.post("/ingest")
def ingest(file: UploadFile) -> dict:
    from pathlib import Path

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(file.file.read())

    _, chunks = _document_processor.process(dest)
    n_added = _vector_store.add_chunks(chunks)
    _retriever.index_corpus(chunks)

    return {"file": file.filename, "chunks_indexed": n_added}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
