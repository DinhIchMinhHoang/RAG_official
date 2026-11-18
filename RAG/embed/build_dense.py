"""Run from project root: python -m embed.build_dense --validate-only"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .embedder import MODEL_NAME, Embedder, child_text


def read_jsonl(path, id_key, required):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: JSON lỗi: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: cần JSON object.")
            for field in required:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_no}: thiếu/rỗng {field}.")
            if row[id_key] in seen:
                raise ValueError(f"{path}:{line_no}: ID trùng {row[id_key]}.")
            seen.add(row[id_key])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: không có dữ liệu.")
    return rows


def load_corpus(children_path, parents_path):
    children = read_jsonl(children_path, "child_id", ("child_id", "parent_id", "breadcrumb", "text"))
    parents = read_jsonl(parents_path, "parent_id", ("parent_id", "text"))
    lookup = {p["parent_id"]: p for p in parents}
    for child in children:
        if child["parent_id"] not in lookup:
            raise ValueError(f"{child['child_id']}: parent không tồn tại: {child['parent_id']}")
        for field in ("article", "chapter"):
            if child.get(field) != lookup[child["parent_id"]].get(field):
                raise ValueError(f"{child['child_id']}: {field} khác parent.")
    return children, parents


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build(children_path, parents_path, output, encoder, batch_size=4):
    import numpy as np
    import faiss
    children, parents = load_corpus(children_path, parents_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} đã tồn tại. Dùng --output thư mục mới để rebuild.")
    vectors = np.ascontiguousarray(encoder.encode([child_text(c) for c in children], batch_size=batch_size), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(children) or vectors.shape[1] == 0:
        raise ValueError("Sai kích thước embeddings.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("Embeddings chứa NaN/Inf hoặc vector rỗng.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    metadata = {
        "format_version": 1, "model_name": MODEL_NAME,
        "model_revision": encoder.resolved_revision,
        "max_length": encoder.max_length, "dimension": int(vectors.shape[1]),
        "num_children": len(children), "num_parents": len(parents),
        "metric": "cosine", "normalized": True, "input_template": "breadcrumb\\ntext",
        "source_sha256": {"children": sha256(children_path), "parents": sha256(parents_path)},
    }
    # Build in a sibling staging directory; expose the completed bundle together.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dense-", dir=output.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        faiss.write_index(index, str(stage / "children.faiss"))
        np.save(stage / "embeddings.npy", vectors, allow_pickle=False)
        write_jsonl(stage / "children.jsonl", children)
        write_jsonl(stage / "parents.jsonl", parents)
        metadata["bundle_sha256"] = {name: sha256(stage / name) for name in
            ("children.faiss", "embeddings.npy", "children.jsonl", "parents.jsonl")}
        (stage / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(output)
    return metadata


def main():
    p = argparse.ArgumentParser(description="Embed children bằng BGE-M3 và tạo FAISS cosine index.")
    p.add_argument("--children", type=Path, default=Path("data/processed/children.jsonl"))
    p.add_argument("--parents", type=Path, default=Path("data/processed/parents.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/indexes/dense_bge_m3"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default=None, help="cpu, cuda, hoặc để tự chọn")
    p.add_argument("--revision", default=None, help="Hugging Face model commit/tag để tái lập")
    p.add_argument("--validate-only", action="store_true", help="Kiểm tra JSON/ID; không tải model, không đếm token")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size phải >= 1")
    children, parents = load_corpus(args.children, args.parents)
    print(f"Corpus OK: {len(children)} children, {len(parents)} parents.")
    if args.validate_only:
        print("Chưa embed và chưa kiểm tra token (cần tokenizer của model).")
        return
    if args.output.exists():
        p.error(f"{args.output} đã tồn tại; chọn --output mới.")
    encoder = Embedder(device=args.device, max_length=args.max_length, revision=args.revision)
    meta = build(args.children, args.parents, args.output, encoder, args.batch_size)
    print(f"Đã lưu {meta['num_children']} vectors x {meta['dimension']} vào {args.output}")


if __name__ == "__main__":
    main()
