"""Optional parent-context reranking with an RRF passthrough fallback."""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Rerank contexts with a cross-encoder or preserve their RRF order.

    ``passthrough`` is the safe default and requires no model or GPU.
    ``cross-encoder`` loads BGE reranker only when CUDA is available, unless a
    model object is explicitly injected (useful for CPU deployments/custom
    HuggingFace CrossEncoder instances). Loading failures fall back to RRF
    order by default; set ``fallback_to_passthrough=False`` to raise instead.
    """

    def __init__(
        self,
        mode: str = "passthrough",
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        fallback_to_passthrough: bool = True,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        if mode not in {"passthrough", "cross-encoder"}:
            raise ValueError("mode phải là 'passthrough' hoặc 'cross-encoder'.")
        if batch_size < 1 or max_length < 8:
            raise ValueError("batch_size phải >= 1 và max_length phải >= 8.")
        self.requested_mode = mode
        self.mode = mode
        self.model_name = model_name
        self.fallback_reason: Optional[str] = None
        self.model = model
        self.batch_size = batch_size

        if mode == "cross-encoder" and self.model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder

                if device is None and not torch.cuda.is_available():
                    raise RuntimeError("Không phát hiện GPU CUDA.")
                selected_device = device or "cuda"
                self.model = CrossEncoder(
                    model_name,
                    max_length=max_length,
                    device=selected_device,
                )
            except Exception as exc:
                if not fallback_to_passthrough:
                    raise RuntimeError(f"Không nạp được cross-encoder: {exc}") from exc
                self.mode = "passthrough"
                self.fallback_reason = str(exc)

    @staticmethod
    def _context_text(context: Dict[str, Any]) -> str:
        breadcrumb = unicodedata.normalize("NFC", str(context.get("breadcrumb", ""))).strip()
        text = unicodedata.normalize("NFC", str(context.get("text", ""))).strip()
        return f"{breadcrumb}\n{text}".strip()

    @staticmethod
    def _score(context: Dict[str, Any]) -> float:
        try:
            value = float(context.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def rerank(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return at most ``top_n`` context dictionaries in relevance order."""
        query = unicodedata.normalize("NFC", str(query)).strip()
        if not query:
            raise ValueError("Câu hỏi không được để trống.")
        if top_n < 1:
            raise ValueError("top_n phải >= 1.")
        rows = [dict(context) for context in contexts]
        if not rows:
            return []

        if self.mode == "passthrough" or self.model is None:
            # HybridRetriever already returns children by descending RRF; the
            # parent wrapper carries the best matching child's score.
            rows = [row for _, row in sorted(
                enumerate(rows), key=lambda pair: (-self._score(pair[1]), pair[0])
            )]
            return rows[:top_n]

        pairs: List[Tuple[str, str]] = [
            (query, self._context_text(context)) for context in rows
        ]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if len(scores) != len(rows):
            raise RuntimeError("Reranker trả số điểm không khớp số context.")

        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for index, (row, raw_score) in enumerate(zip(rows, scores)):
            if isinstance(raw_score, (list, tuple)):
                if len(raw_score) != 1:
                    raise RuntimeError("Cross-encoder cần trả một điểm cho mỗi context.")
                raw_score = raw_score[0]
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("Cross-encoder trả về điểm NaN/Inf.")
            row["rerank_score"] = score
            ranked.append((index, row))
        ranked.sort(key=lambda pair: (-pair[1]["rerank_score"], pair[0]))
        return [row for _, row in ranked[:top_n]]
