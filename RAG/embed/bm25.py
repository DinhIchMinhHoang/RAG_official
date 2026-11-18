"""Vietnamese BM25. Persist tokens as JSON, never pickle Python objects."""
import importlib.metadata
import re
import unicodedata

TOKENIZER_FORMAT = 1


def tokenize(text, mode="underthesea"):
    text = unicodedata.normalize("NFC", text).lower().strip()
    if mode == "underthesea":
        from underthesea import word_tokenize
        text = word_tokenize(text, format="text")
    elif mode != "regex":
        raise ValueError(f"Tokenizer không hỗ trợ: {mode}")
    # Keep Vietnamese accents, compound words (sinh_viên), and article numbers.
    return re.findall(r"\w+", text, flags=re.UNICODE)


def tokenizer_version(mode):
    return importlib.metadata.version("underthesea") if mode == "underthesea" else None


class BM25Retriever:
    def __init__(self, index_dir="data/indexes/bm25"):
        import json
        from pathlib import Path
        from rank_bm25 import BM25Okapi
        from .build_dense import load_corpus, sha256
        path = Path(index_dir)
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        meta = self.metadata
        if meta["format_version"] != 1 or meta["tokenizer_format"] != TOKENIZER_FORMAT:
            raise ValueError("BM25 format không tương thích; cần rebuild.")
        self.mode = meta["tokenizer"]
        if tokenizer_version(self.mode) != meta["tokenizer_version"]:
            raise ValueError("Phiên bản underthesea đã đổi; cần rebuild BM25.")
        for name, expected in meta["bundle_sha256"].items():
            if sha256(path / name) != expected:
                raise ValueError(f"{name} đã thay đổi; cần rebuild BM25.")
        self.children, parents = load_corpus(path / "children.jsonl", path / "parents.jsonl")
        self.parents = {p["parent_id"]: p for p in parents}
        self.tokens = json.loads((path / "tokens.json").read_text(encoding="utf-8"))
        if len(self.tokens) != len(self.children):
            raise ValueError("Tokens và children không khớp.")
        self.engine = BM25Okapi(self.tokens, k1=meta["k1"], b=meta["b"], epsilon=meta["epsilon"])

    def search(self, query, top_k=10):
        import numpy as np
        if not query.strip() or top_k < 1:
            raise ValueError("Query không được rỗng; top_k phải >= 1.")
        query_tokens = tokenize(query, self.mode)
        if not query_tokens:
            return []
        scores = self.engine.get_scores(query_tokens)
        # Zero-score/OOV documents must not gain arbitrary RRF contributions.
        positions = np.argsort(-scores, kind="stable")
        return [{**self.children[int(i)], "score": float(scores[i])}
                for i in positions if scores[i] > 0][:top_k]


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Tìm riêng BM25 để đối chiếu với dense/hybrid.")
    p.add_argument("query")
    p.add_argument("--index", default="data/indexes/bm25")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    print(json.dumps(BM25Retriever(args.index).search(args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
