"""Dense + BM25 -> RRF on child_id -> unique parent lookup (no reranker yet)."""
import argparse
import json
import math

from .bm25 import BM25Retriever
from .search_dense import DenseRetriever


def fuse_rrf(dense_hits, bm25_hits, rrf_k=60, dense_weight=1.0, bm25_weight=1.0):
    if not math.isfinite(rrf_k) or rrf_k <= 0:
        raise ValueError("rrf_k phải > 0 và hữu hạn.")
    if any(not math.isfinite(w) or w < 0 for w in (dense_weight, bm25_weight)) or dense_weight + bm25_weight <= 0:
        raise ValueError("Trọng số hữu hạn, không âm và ít nhất một trọng số > 0.")
    fused = {}
    for name, hits, weight in (("dense", dense_hits, dense_weight), ("bm25", bm25_hits, bm25_weight)):
        if weight == 0:
            continue
        seen = set()
        for rank, hit in enumerate(hits, 1):
            cid = hit["child_id"]
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in fused:
                fused[cid] = {**{k: v for k, v in hit.items() if k != "score"}, "score": 0.0,
                              "dense_rank": None, "bm25_rank": None, "dense_score": None, "bm25_score": None}
            row = fused[cid]
            row["score"] += weight / (rrf_k + rank)
            row[name + "_rank"] = rank
            row[name + "_score"] = hit["score"]
    return sorted(fused.values(), key=lambda h: (-h["score"], h["child_id"]))


class HybridRetriever:
    def __init__(self, dense_index="data/indexes/dense_bge_m3", bm25_index="data/indexes/bm25", device=None,
                 dense=None, bm25=None):
        # Read and compare corpora BEFORE loading the expensive dense model.
        self.bm25 = bm25 if bm25 is not None else BM25Retriever(bm25_index)
        if dense is None:
            from pathlib import Path
            from .build_dense import load_corpus
            children, parents = load_corpus(Path(dense_index) / "children.jsonl", Path(dense_index) / "parents.jsonl")
            dense_children = {c["child_id"]: c for c in children}
            dense_parents = {p["parent_id"]: p for p in parents}
        else:
            dense_children = {c["child_id"]: c for c in dense.children}
            dense_parents = dense.parents
        if dense_children != {c["child_id"]: c for c in self.bm25.children} or dense_parents != self.bm25.parents:
            raise ValueError("Dense và BM25 khác dữ liệu/metadata. Rebuild cả hai từ cùng children và parents.")
        self.dense = dense if dense is not None else DenseRetriever(dense_index, device=device)
        self.parents = self.dense.parents

    def search(self, query, candidate_k=100, top_k=10, max_parents=5, rrf_k=60, dense_weight=1., bm25_weight=1.):
        if not query.strip() or top_k < 1 or candidate_k < top_k or max_parents < 1:
            raise ValueError("Query không rỗng; candidate_k >= top_k >= 1 và max_parents >= 1.")
        # Validate fusion parameters before running the model.
        fuse_rrf([], [], rrf_k, dense_weight, bm25_weight)
        dense_hits = self.dense.search(query, top_k=candidate_k, max_parents=1)["children"] if dense_weight else []
        bm25_hits = self.bm25.search(query, top_k=candidate_k) if bm25_weight else []
        hits = fuse_rrf(dense_hits, bm25_hits, rrf_k, dense_weight, bm25_weight)[:top_k]
        contexts, seen = [], set()
        for hit in hits:
            pid = hit["parent_id"]
            if pid not in seen and len(contexts) < max_parents:
                seen.add(pid)
                contexts.append({**self.parents[pid], "score": hit["score"], "matched_child_id": hit["child_id"]})
        return {"query": query, "method": "weighted_rrf", "rrf_k": rrf_k,
                "dense_weight": dense_weight, "bm25_weight": bm25_weight,
                "children": hits, "parents": contexts}


def main():
    p = argparse.ArgumentParser(description="Tìm kiếm hybrid BM25 + BGE-M3 bằng RRF.")
    p.add_argument("query")
    p.add_argument("--dense-index", default="data/indexes/dense_bge_m3")
    p.add_argument("--bm25-index", default="data/indexes/bm25")
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-parents", type=int, default=5)
    p.add_argument("--rrf-k", type=float, default=60)
    p.add_argument("--dense-weight", type=float, default=1.)
    p.add_argument("--bm25-weight", type=float, default=1.)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    retriever = HybridRetriever(args.dense_index, args.bm25_index, device=args.device)
    print(json.dumps(retriever.search(args.query, args.candidate_k, args.top_k, args.max_parents,
                                    args.rrf_k, args.dense_weight, args.bm25_weight), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
