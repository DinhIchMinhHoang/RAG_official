"""Evaluate BM25, BGE-M3 dense, and RRF hybrid retrieval on eval_set.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from embed.bm25 import BM25Retriever
from embed.search_dense import DenseRetriever
from embed.search_hybrid import HybridRetriever


METHODS = ("BM25", "Dense (BGE-M3)", "Hybrid (RRF)")
K_VALUES = (1, 3, 5)


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được eval set {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("eval_set.json phải chứa một JSON array.")
    answerable = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ValueError(f"Phần tử eval #{index + 1} thiếu question hợp lệ.")
        relevant = row.get("relevant_parent_ids", [])
        if row.get("answerable") is True:
            if not isinstance(relevant, list) or not relevant or not all(isinstance(pid, str) for pid in relevant):
                raise ValueError(f"{row.get('id', index)}: câu answerable cần relevant_parent_ids là list[str] không rỗng.")
            answerable.append(row)
    if not answerable:
        raise ValueError("Không có câu hỏi answerable với relevant_parent_ids để đánh giá retrieval.")
    return answerable


def unique_parent_ranking(children: Iterable[Mapping[str, Any]]) -> List[str]:
    """Collapse child hits to parent IDs, retaining the first (best-ranked) hit."""
    ranked: List[str] = []
    seen: Set[str] = set()
    for hit in children:
        parent_id = hit.get("parent_id")
        if isinstance(parent_id, str) and parent_id not in seen:
            seen.add(parent_id)
            ranked.append(parent_id)
    return ranked


def score_ranking(
    ranked_parent_ids: Sequence[str],
    relevant_parent_ids: Sequence[str],
) -> Dict[str, float]:
    relevant = set(relevant_parent_ids)
    if not relevant:
        raise ValueError("Cần ít nhất một relevant_parent_id để tính metric.")
    rank_by_parent = {parent_id: rank for rank, parent_id in enumerate(ranked_parent_ids, 1)}
    first_relevant_rank = min(
        (rank_by_parent[parent_id] for parent_id in relevant if parent_id in rank_by_parent),
        default=None,
    )
    scores: Dict[str, float] = {}
    for k in K_VALUES:
        top_ids = set(ranked_parent_ids[:k])
        scores[f"Hit@{k}"] = float(bool(relevant & top_ids))
        if k in (3, 5):
            scores[f"Recall@{k}"] = len(relevant & top_ids) / len(relevant)
    scores["MRR"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    return scores


def evaluate(
    questions: Sequence[Mapping[str, Any]],
    bm25: Any,
    dense: Any,
    hybrid: Any,
    candidate_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    totals = {method: {"Hit@1": 0.0, "Hit@3": 0.0, "Hit@5": 0.0,
                       "Recall@3": 0.0, "Recall@5": 0.0, "MRR": 0.0}
              for method in METHODS}
    for item in questions:
        query = item["question"]
        relevant = item["relevant_parent_ids"]
        ranked_by_method = {
            "BM25": unique_parent_ranking(bm25.search(query, top_k=candidate_k)),
            "Dense (BGE-M3)": unique_parent_ranking(
                dense.search(query, top_k=candidate_k, max_parents=candidate_k)["children"]
            ),
            "Hybrid (RRF)": unique_parent_ranking(hybrid.search(
                query,
                candidate_k=candidate_k,
                top_k=candidate_k,
                max_parents=candidate_k,
            )["children"]),
        }
        for method, ranked in ranked_by_method.items():
            scores = score_ranking(ranked, relevant)
            for metric, value in scores.items():
                totals[method][metric] += value

    count = float(len(questions))
    return {
        method: {metric: value / count for metric, value in metrics.items()}
        for method, metrics in totals.items()
    }


def markdown_report(results: Mapping[str, Mapping[str, float]], query_count: int) -> str:
    metrics = ("Hit@1", "Hit@3", "Hit@5", "Recall@3", "Recall@5", "MRR")
    lines = [
        "# Retrieval evaluation",
        "",
        f"Số câu hỏi có đáp án được đánh giá: **{query_count}**.",
        "Đơn vị truy hồi là parent; các child trùng parent được gộp theo vị trí child đầu tiên.",
        "MRR tính trên toàn bộ danh sách parent trả về trong candidate set.",
        "",
        "| Phương pháp | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(metrics) + 1),
    ]
    for method in METHODS:
        values = results[method]
        lines.append("| " + method + " | " + " | ".join(f"{values[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="So sánh BM25, Dense BGE-M3 và Hybrid RRF.")
    parser.add_argument("--eval-set", type=Path, default=Path(__file__).with_name("eval_set.json"))
    parser.add_argument("--dense-index", type=Path, default=RAG_DIR / "data/indexes/dense_bge_m3")
    parser.add_argument("--bm25-index", type=Path, default=RAG_DIR / "data/indexes/bm25")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--device", default=None, help="Thiết bị BGE-M3, ví dụ cpu hoặc cuda")
    parser.add_argument("--output", type=Path, default=None, help="Ghi bảng Markdown ra file nếu được chỉ định")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.candidate_k < 5:
        parser.error("--candidate-k phải >= 5 để tính Hit@5 và Recall@5")
    try:
        questions = load_eval_set(args.eval_set)
        bm25 = BM25Retriever(str(args.bm25_index))
        dense = DenseRetriever(str(args.dense_index), device=args.device)
        hybrid = HybridRetriever(
            dense_index=str(args.dense_index),
            bm25_index=str(args.bm25_index),
            device=args.device,
            dense=dense,
            bm25=bm25,
        )
        results = evaluate(questions, bm25, dense, hybrid, candidate_k=args.candidate_k)
    except Exception as exc:
        print(f"Lỗi đánh giá retrieval: {exc}", file=sys.stderr)
        return 1

    report = markdown_report(results, len(questions))
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Đã ghi báo cáo: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
