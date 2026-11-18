"""python -m embed.build_bm25; default: segment Vietnamese with underthesea."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile

from .bm25 import TOKENIZER_FORMAT, tokenize, tokenizer_version
from .build_dense import load_corpus, sha256, write_jsonl
from .embedder import child_text


def build(children_path, parents_path, output, mode="underthesea", k1=1.2, b=0.65):
    from rank_bm25 import BM25Okapi
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Cần k1 > 0 và 0 <= b <= 1.")
    children, parents = load_corpus(children_path, parents_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} đã tồn tại; chọn --output mới.")
    tokens = [tokenize(child_text(c), mode) for c in children]
    if any(not t for t in tokens):
        raise ValueError("Có child không có token hợp lệ.")
    BM25Okapi(tokens, k1=k1, b=b)  # Validate that corpus can be indexed.
    meta = {
        "format_version": 1, "algorithm": "BM25Okapi", "k1": k1, "b": b, "epsilon": 0.25,
        "tokenizer": mode, "tokenizer_format": TOKENIZER_FORMAT,
        "tokenizer_version": tokenizer_version(mode),
        "rank_bm25_version": importlib.metadata.version("rank-bm25"),
        "num_children": len(children), "num_parents": len(parents),
        "input_template": "breadcrumb\\ntext",
        "source_sha256": {"children": sha256(children_path), "parents": sha256(parents_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bm25-", dir=output.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        write_jsonl(stage / "children.jsonl", children)
        write_jsonl(stage / "parents.jsonl", parents)
        (stage / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        meta["bundle_sha256"] = {n: sha256(stage / n) for n in ("children.jsonl", "parents.jsonl", "tokens.json")}
        (stage / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(output)
    return meta


def main():
    p = argparse.ArgumentParser(description="Tạo BM25 index từ cùng children/parents với dense.")
    p.add_argument("--children", type=Path, default=Path("data/processed/children.jsonl"))
    p.add_argument("--parents", type=Path, default=Path("data/processed/parents.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/indexes/bm25"))
    p.add_argument("--tokenizer", choices=["underthesea", "regex"], default="underthesea")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.65)
    args = p.parse_args()
    meta = build(args.children, args.parents, args.output, args.tokenizer, args.k1, args.b)
    print(f"BM25 OK: {meta['num_children']} children, tokenizer={meta['tokenizer']}, output={args.output}")


if __name__ == "__main__":
    main()