"""Ingest orchestration: file -> parse -> chunk -> merge corpus -> rebuild Collection (+ optional Postgres/MinIO mirror).

Store layout:  <store>/corpus/{children,parents}.jsonl, <store>/documents.json, <store>/parsed/<doc_id>/, <store>/index/
"""
import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

from parser.chunker import ChunkingError, chunk_document, load_blocks, write_jsonl  # noqa: F401
from parser.parse_document import SUPPORTED, sha256

CORPUS_FILES = ("children.jsonl", "parents.jsonl")


# ---- helpers ----
def slugify(text):
    t = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40] or "doc"


def make_doc_id(path, sha):
    return f"{slugify(Path(path).stem)}-{sha[:6]}"


def expand_paths(paths):
    """Files kept as given; directories expanded (recursive, sorted) to supported extensions."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED
                          and "processed" not in f.relative_to(p).parts)
        else:
            out.append(p)
    return out


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_registry(store):
    f = Path(store) / "documents.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_registry(store, reg):
    (Path(store) / "documents.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_corpus(store, doc_id, parents, children, drop_ids=()):
    """Replace every row of doc_id (and drop_ids) in the corpus by the new parents/children."""
    d = Path(store) / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    gone = {doc_id, *drop_ids}
    for name, new in (("children.jsonl", children), ("parents.jsonl", parents)):
        rows = [r for r in _read_jsonl(d / name) if r.get("doc_id") not in gone] + list(new)
        write_jsonl(rows, d / name)


def _pages(payload, blocks):
    n = (payload.get("stats") or {}).get("pages")
    if n:
        return n
    return len({(b.get("location") or {}).get("page_num") for b in blocks} - {None}) or None


# ---- one file ----
def ingest_file(path, store, count_tokens, ceiling=400, force=False, parse_fn=None):
    """-> result dict. status 'skip' (same sha already ingested) or 'ok'. Raises on parse/chunk failure."""
    if parse_fn is None:
        from parser.parse_document import execute as parse_fn
    path, store = Path(path).resolve(), Path(store)
    store.mkdir(parents=True, exist_ok=True)
    sha = sha256(path)
    doc_id = make_doc_id(path, sha)
    reg = load_registry(store)
    if doc_id in reg and not force:
        return {"doc_id": doc_id, "path": str(path), "status": "skip", **{k: reg[doc_id].get(k) for k in
                ("pages", "blocks", "parents", "children", "max_tokens")}, "parse_s": 0, "chunk_s": 0}
    t = time.perf_counter()
    payload = parse_fn(path, store / "parsed" / doc_id, force)
    parse_s = time.perf_counter() - t
    blocks = payload["blocks"]
    t = time.perf_counter()
    parents, children = chunk_document(blocks, doc_id, count_tokens, ceiling=ceiling, doc_title=path.stem)
    chunk_s = time.perf_counter() - t
    stale = [d for d, v in reg.items() if d != doc_id and v.get("source") == str(path)]
    merge_corpus(store, doc_id, parents, children, drop_ids=stale)
    info = {"source": str(path), "title": path.stem, "filename": path.name, "sha256": sha,
            "size_bytes": path.stat().st_size, "pages": _pages(payload, blocks), "blocks": len(blocks),
            "parents": len(parents), "children": len(children),
            "max_tokens": max(c["tokens"] for c in children), "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for d in stale:
        reg.pop(d, None)
    reg[doc_id] = info
    save_registry(store, reg)
    return {"doc_id": doc_id, "path": str(path), "status": "ok", **{k: info[k] for k in
            ("pages", "blocks", "parents", "children", "max_tokens")}, "parse_s": parse_s, "chunk_s": chunk_s,
            "replaced": stale, "_payload": payload, "_parents": parents, "_children": children}


def ingest_paths(paths, store, count_tokens, ceiling=400, force=False, parse_fn=None, out=print):
    """Failure isolation: a document that raises is reported and the others continue. -> list of results."""
    results = []
    for p in paths:
        try:
            if not Path(p).is_file():
                raise FileNotFoundError(f"không thấy tệp: {p}")
            results.append(ingest_file(p, store, count_tokens, ceiling, force, parse_fn))
        except Exception as e:  # ChunkingError, parse/OCR errors, missing file
            out(f"!!! THẤT BẠI {p}: {type(e).__name__}: {e}")
            results.append({"doc_id": None, "path": str(p), "status": "failed", "error": f"{type(e).__name__}: {e}"})
    return results


# ---- index ----
def index_is_current(store):
    """True if <store>/index was built from exactly the current corpus files."""
    store = Path(store)
    m = store / "index" / "manifest.json"
    if not m.exists():
        return False
    from embed.collection import sha256_lf
    want = json.loads(m.read_text(encoding="utf-8")).get("sha256", {})
    return all((store / "corpus" / n).exists()
               and sha256_lf((store / "corpus" / n).read_bytes()) == want.get(n) for n in CORPUS_FILES)


def qdrant_name(store):
    """Collection name from the store directory name, so different stores never collide."""
    return "rag_" + slugify(Path(store).resolve().name).replace("-", "_")


def connect_qdrant(settings=None):
    """-> QdrantClient, or ConnectionError with the fix when the server is unreachable."""
    from qdrant_client import QdrantClient
    from store.settings import StoreSettings
    s = settings or StoreSettings.from_env()
    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=5)
        client.get_collections()
    except Exception as e:
        raise ConnectionError(f"Không kết nối được Qdrant tại {s.qdrant_url} ({type(e).__name__}). "
                              "Chạy: docker compose up -d (thư mục RAG/deploy), rồi kiểm tra: "
                              "python -m store.doctor") from e
    return client


def qdrant_is_current(store, client, name):
    """True if the Qdrant collections hold exactly the current corpus (manifest sha256 equal)."""
    from embed.collection import _normalize, _parse, _read_rows
    from embed.qdrant_collection import _corpus_hash
    if not all(client.collection_exists(n) for n in (name, name + "_parents", name + "_meta")):
        return False
    got = client.retrieve(name + "_meta", ids=[1])
    corpus = Path(store) / "corpus"
    rows, prows = _normalize(_parse(_read_rows(corpus / "children.jsonl"), "children"),
                             _parse(_read_rows(corpus / "parents.jsonl"), "parents"))
    return bool(got) and got[0].payload.get("sha256") == _corpus_hash(rows, prows)


def rebuild_index(store, device=None, builder=None, force=False, backend="faiss", client=None):
    """faiss: build into index.tmp then swap. qdrant: drop+recreate the store's collections.
    -> (rebuilt, seconds). Skips when the index already matches the corpus."""
    store = Path(store)
    corpus = store / "corpus"
    if not (corpus / "children.jsonl").exists() or not _read_jsonl(corpus / "children.jsonl"):
        return False, 0.0
    if backend == "qdrant":
        name, client = qdrant_name(store), client or connect_qdrant()
        if not force and qdrant_is_current(store, client, name):
            return False, 0.0
        if builder is None:
            from embed.qdrant_collection import QdrantCollection
            builder = lambda c, p, device=None: QdrantCollection.build(c, p, name=name, client=client, device=device)
        t = time.perf_counter()
        builder(corpus / "children.jsonl", corpus / "parents.jsonl", device=device).close()
        return True, time.perf_counter() - t
    if not force and index_is_current(store):
        return False, 0.0
    if builder is None:
        from embed.collection import Collection
        builder = Collection.build
    tmp, final = store / "index.tmp", store / "index"
    shutil.rmtree(tmp, ignore_errors=True)
    t = time.perf_counter()
    col = builder(corpus / "children.jsonl", corpus / "parents.jsonl", tmp, device=device)
    col.close()
    del col
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return True, time.perf_counter() - t


# ---- optional Postgres + MinIO mirror ----
def stack_ready(settings, out=print):
    try:
        from store import doctor
        doctor._pg(settings, False)
        doctor._minio(settings, False)
        return True
    except Exception as e:
        out(f"CẢNH BÁO: --stack bỏ qua (Postgres/MinIO không sẵn sàng: {type(e).__name__}: {str(e)[:100]}). "
            "Chạy `python -m store.doctor --init`.")
        return False


def mirror_stack(results, settings=None, out=print):
    """Mirror ok results to Postgres (documents/parents/children) and MinIO (raw file + parsed blocks)."""
    from store.settings import StoreSettings
    s = settings or StoreSettings.from_env()
    if not stack_ready(s, out):
        return False
    from store import objects, pg
    mc = objects.client(s)
    for r in results:
        if r["status"] != "ok":
            continue
        try:
            doc_id, path = r["doc_id"], Path(r["path"])
            sha = sha256(path)
            with pg.connect(s) as conn:
                for old in r.get("replaced", []):
                    pg.delete_document(conn, old)
                pg.upsert_document(conn, {
                    "doc_id": doc_id, "title": path.stem, "filename": path.name, "sha256": sha,
                    "size_bytes": path.stat().st_size, "pages": r["pages"], "status": "indexing",
                    "object_key": objects.raw_key(sha, path.name)})
                pg.replace_chunks(conn, doc_id, [{**p, "ord": i} for i, p in enumerate(r["_parents"])],
                                  [{**c, "ord": i} for i, c in enumerate(r["_children"])])
                pg.set_status(conn, doc_id, "ready", r["children"], r["children"])
            objects.put_file(mc, s.minio_bucket, path, objects.raw_key(sha, path.name))
            objects.put_bytes(mc, s.minio_bucket, json.dumps(r["_payload"], ensure_ascii=False).encode("utf-8"),
                              objects.blocks_key(doc_id), "application/json")
            out(f"stack: đã đồng bộ {doc_id}")
        except Exception as e:
            out(f"CẢNH BÁO: stack không đồng bộ được {r['doc_id']}: {type(e).__name__}: {str(e)[:150]}")
    return True


# ---- report ----
def format_table(results, index_s=0.0):
    cols = ("tài liệu", "trạng thái", "trang", "khối", "parents", "children", "max tok", "parse/OCR s",
            "chunk s", "embed+index s")
    ok = [r for r in results if r["status"] == "ok"]
    total_c = sum(r["children"] for r in ok) or 1
    rows = []
    for r in results:
        if r["status"] == "failed":
            rows.append([Path(r["path"]).name, "LỖI"] + ["-"] * 8)
            continue
        idx = f"~{index_s * r['children'] / total_c:.1f}" if r["status"] == "ok" and index_s else "-"
        rows.append([r["doc_id"], r["status"], r["pages"], r["blocks"], r["parents"], r["children"],
                     r["max_tokens"], f"{r['parse_s']:.1f}", f"{r['chunk_s']:.1f}", idx])
    rows = [[str(x) if x is not None else "-" for x in row] for row in rows]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    line = lambda row: "  ".join(v.ljust(w[i]) for i, v in enumerate(row))
    return "\n".join([line(cols), line(["-" * n for n in w])] + [line(r) for r in rows])
