"""Dense retrieval returning children and unique parent contexts for later RAG."""
import argparse
import json
from pathlib import Path

from .build_dense import load_corpus, sha256
from .embedder import MODEL_NAME, Embedder


class DenseRetriever:
    def __init__(self, index_dir="data/indexes/dense_bge_m3", device=None, encoder=None):
        import faiss
        path = Path(index_dir)
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["format_version"] != 1 or self.metadata["model_name"] != MODEL_NAME:
            raise ValueError("Index không đúng format/model BGE-M3.")
        for name, expected in self.metadata["bundle_sha256"].items():
            if sha256(path / name) != expected:
                raise ValueError(f"{name} đã thay đổi; cần rebuild index.")
        self.children, parents = load_corpus(path / "children.jsonl", path / "parents.jsonl")
        self.parents = {r["parent_id"]: r for r in parents}
        faiss_file = path / "children.faiss"
        try:
            self.index = faiss.read_index(str(faiss_file))
        except RuntimeError:
            # Fallback for Windows Unicode paths (e.g. Vietnamese accented folder names)
            import numpy as np
            self.index = faiss.deserialize_index(np.frombuffer(faiss_file.read_bytes(), dtype=np.uint8))
        if self.index.ntotal != len(self.children) or self.index.d != self.metadata["dimension"]:
            raise ValueError("Index và metadata không khớp.")
        self.encoder = encoder if encoder is not None else Embedder(
            device=device, max_length=self.metadata["max_length"],
            revision=self.metadata["model_revision"],
        )

    def search(self, query, top_k=10, max_parents=5):
        import numpy as np
        import faiss
        if not query.strip() or top_k < 1 or max_parents < 1:
            raise ValueError("Câu hỏi cần có nội dung; top_k và max_parents phải >= 1.")
        vector = np.ascontiguousarray(self.encoder.encode_query(query), dtype="float32")
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Query embedding không hợp lệ hoặc khác dimension.")
        faiss.normalize_L2(vector)
        scores, positions = self.index.search(vector, min(top_k, len(self.children)))
        hits, contexts, seen = [], [], set()
        for score, position in zip(scores[0], positions[0]):
            child = self.children[int(position)]
            hit = {**child, "score": float(score)}
            hits.append(hit)
            pid = child["parent_id"]
            if pid not in seen and len(contexts) < max_parents:
                seen.add(pid)
                contexts.append({**self.parents[pid], "score": float(score), "matched_child_id": child["child_id"]})
        return {"query": query, "children": hits, "parents": contexts}


def main():
    p = argparse.ArgumentParser(description="Tìm child và lấy parent tương ứng.")
    p.add_argument("query")
    p.add_argument("--index", default="data/indexes/dense_bge_m3")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-parents", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    result = DenseRetriever(args.index, device=args.device).search(args.query, args.top_k, args.max_parents)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
