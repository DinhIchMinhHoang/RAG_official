"""Hybrid retrieval wrapper that returns deduplicated whole-article contexts."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _default_index_path(name: str) -> str:
    """Resolve indexes relative to RAG/, independent of the current directory."""
    rag_dir = Path(__file__).resolve().parents[1]
    return str(rag_dir / "data" / "indexes" / name)


def _load_hybrid_class() -> Any:
    """Support both ``python -m RAG.llm.generate`` and ``python -m llm.generate``."""
    try:
        from ..embed.search_hybrid import HybridRetriever
    except ImportError:
        from embed.search_hybrid import HybridRetriever
    return HybridRetriever


def _parent_breadcrumb(parent: Dict[str, Any]) -> str:
    breadcrumb = unicodedata.normalize("NFC", str(parent.get("breadcrumb", ""))).strip()
    if breadcrumb:
        return breadcrumb
    parts: List[str] = []
    chapter = parent.get("chapter")
    if chapter:
        parts.append(f"Chương {chapter}")
    article = parent.get("article")
    if article is not None:
        title = unicodedata.normalize("NFC", str(parent.get("title", ""))).strip()
        parts.append(f"Điều {article}" + (f". {title}" if title else ""))
    return " > ".join(parts) or str(parent.get("parent_id", "Quy chế đào tạo"))


def estimate_tokens(text: str) -> int:
    """Conservative rough estimate; inject a model tokenizer for exact budgeting."""
    return max(1, math.ceil(len(text) / 3)) if text else 0


class RAGRetriever:
    """Wrap :class:`HybridRetriever` and prepare unique parent/article contexts."""

    def __init__(
        self,
        hybrid_retriever: Optional[Any] = None,
        dense_index: Optional[str | Path] = None,
        bm25_index: Optional[str | Path] = None,
        device: Optional[str] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        rag_dir = Path(__file__).resolve().parents[1]
        self.dense_index = str(dense_index or rag_dir / "data/indexes/dense_bge_m3")
        self.bm25_index = str(bm25_index or rag_dir / "data/indexes/bm25")
        if hybrid_retriever is None:
            hybrid_class = _load_hybrid_class()
            hybrid_retriever = hybrid_class(
                dense_index=self.dense_index,
                bm25_index=self.bm25_index,
                device=device,
            )
        self.hybrid = hybrid_retriever
        self.token_counter = token_counter or estimate_tokens

    def search(
        self,
        query: str,
        candidate_k: int = 100,
        top_k: int = 10,
        max_parents: int = 3,
        max_context_tokens: Optional[int] = 6000,
    ) -> Dict[str, Any]:
        """Return RRF-ranked children and unique, budgeted parent contexts."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Câu hỏi không được để trống.")
        query = unicodedata.normalize("NFC", query.strip())
        if candidate_k < top_k or top_k < 1:
            raise ValueError("Cần candidate_k >= top_k >= 1.")
        if max_parents < 1:
            raise ValueError("max_parents phải >= 1.")
        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens phải >= 1 hoặc None.")

        result = self.hybrid.search(
            query,
            candidate_k=candidate_k,
            top_k=top_k,
            max_parents=max_parents,
        )
        children = list(result.get("children", []))
        parent_lookup = getattr(self.hybrid, "parents", None)
        if parent_lookup is None:
            parent_lookup = {
                row["parent_id"]: row
                for row in result.get("parents", [])
                if row.get("parent_id")
            }

        contexts: List[Dict[str, Any]] = []
        seen: set[str] = set()
        token_total = 0
        for child in children:
            parent_id = child.get("parent_id")
            if not parent_id or parent_id in seen:
                continue
            parent = parent_lookup.get(parent_id)
            if not parent or not str(parent.get("text", "")).strip():
                continue
            seen.add(parent_id)

            text = unicodedata.normalize("NFC", str(parent["text"])).strip()
            breadcrumb = _parent_breadcrumb(parent)
            token_cost = int(self.token_counter(f"{breadcrumb}\n{text}"))
            if token_cost < 0:
                raise ValueError("token_counter không được trả về số âm.")
            if max_context_tokens is not None and token_total + token_cost > max_context_tokens:
                continue

            context = dict(parent)
            context.update({
                "parent_id": parent_id,
                "breadcrumb": breadcrumb,
                "text": text,
                "score": child.get("score"),
                "matched_child_id": child.get("child_id"),
                "context_token_estimate": token_cost,
            })
            contexts.append(context)
            token_total += token_cost
            if len(contexts) >= max_parents:
                break

        return {
            "query": query,
            "method": result.get("method", "hybrid_rrf"),
            "children": children,
            "parents": contexts,
            "context_tokens_estimated": token_total,
        }

    def get_contexts(
        self,
        query: str,
        top_k: int = 10,
        max_parents: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return unique parent contexts ordered by their best child's RRF score."""
        result = self.search(
            query,
            candidate_k=max(100, top_k),
            top_k=top_k,
            max_parents=max_parents,
        )
        return result["parents"]


def retrieve_contexts(
    query: str,
    retriever: RAGRetriever,
    **search_options: Any,
) -> Dict[str, Any]:
    """Functional entry point for callers that already own a retriever."""
    return retriever.search(query, **search_options)
