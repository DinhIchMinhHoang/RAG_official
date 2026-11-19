"""FastAPI service exposing the regulation RAG generator to the Vite UI."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .llm.generate import RAGGenerator
    from .llm.retrieve import RAGRetriever
except ImportError:
    # Also supports ``uvicorn api:app`` when launched from the RAG directory.
    from llm.generate import RAGGenerator
    from llm.retrieve import RAGRetriever


LOGGER = logging.getLogger("rag.api")
_generator: RAGGenerator | None = None
_generator_lock = threading.Lock()
_request_lock = threading.Lock()


def _cors_origins() -> List[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(
    title="Vietnamese Training Regulation RAG API",
    version="1.0.0",
    description="Question answering over the university training regulation corpus.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Câu hỏi tiếng Việt")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


def _get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                backend = os.getenv("RAG_BACKEND", "ollama").strip().lower()
                device = os.getenv("RAG_DEVICE") or None
                retriever = RAGRetriever(
                    dense_index=os.getenv("RAG_DENSE_INDEX") or None,
                    bm25_index=os.getenv("RAG_BM25_INDEX") or None,
                    device=device,
                )
                _generator = RAGGenerator(
                    backend=backend,
                    model=os.getenv("RAG_MODEL") or None,
                    retriever=retriever,
                    rerank_mode=os.getenv("RAG_RERANK_MODE", "passthrough"),
                    base_url=os.getenv("OLLAMA_BASE_URL") or None,
                    timeout=float(os.getenv("RAG_TIMEOUT_SECONDS", "120")),
                    temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
                    max_tokens=int(os.getenv("RAG_MAX_TOKENS", "800")),
                    top_k=int(os.getenv("RAG_TOP_K", "10")),
                    max_parents=int(os.getenv("RAG_MAX_PARENTS", "3")),
                    max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
                    device=device,
                )
    return _generator


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query không được để trống.")
    try:
        generator = _get_generator()
        # Dense encoders and local Transformers models are shared singletons;
        # serialize generation to avoid concurrent access to one model instance.
        with _request_lock:
            result = generator.generate_answer(query)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("RAG request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Không thể tạo câu trả lời lúc này. Hãy kiểm tra dịch vụ RAG và LLM.",
        ) from exc
