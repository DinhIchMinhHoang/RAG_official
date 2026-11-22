"""Multi-document Collection: dense (FAISS) + BM25 + accent-stripped BM25, fusion, optional child-level rerank.

All models run locally (bge-m3, bge-reranker-v2-m3). Models load lazily; close() frees them.
"""
import gc
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

from .bm25 import TOKENIZER_FORMAT, tokenize, tokenizer_version
from .embedder import MODEL_NAME, child_text

FORMAT_VERSION = 1
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
FILES = ("children.jsonl", "parents.jsonl", "embeddings.npy", "tokens.json")
CHILD_DEFAULTS = {"doc_id": "doc", "breadcrumb": "", "page_start": None, "page_end": None, "bbox": None,
                  "block_ids": None, "tokens": None, "content_type": None, "chapter": None, "article": None}
PARENT_DEFAULTS = {"doc_id": "doc", "title": None, "breadcrumb": None, "page_start": None, "page_end": None,
                   "chapter": None, "article": None}


def strip_accents(text):
    text = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def _tok_stripped(text):
    # regex mode on purpose: underthesea segments unaccented text unlike accented text, unigrams match robustly.
    return tokenize(strip_accents(text), "regex")


def sha256_lf(data):
    """Hash LF-normalized bytes so CRLF/LF checkouts agree."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def _read_rows(src):
    """src: path (jsonl) or list of dicts -> LF-normalized jsonl bytes."""
    if isinstance(src, (str, Path)):
        data = Path(src).read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        return data.replace(b"\r\n", b"\n")
    return _jsonl_bytes(src)


def _parse(data, name):
    rows = []
    for n, line in enumerate(data.decode("utf-8").split("\n"), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{name}:{n}: JSON lỗi: {e}") from e
    if not rows:
        raise ValueError(f"{name}: không có dữ liệu.")
    return rows


def _normalize(children, parents):
    seen = set()
    for c in children:
        for f in ("child_id", "parent_id", "text"):
            if not isinstance(c.get(f), str) or not c[f].strip():
                raise ValueError(f"child thiếu/rỗng {f}: {c.get('child_id')}")
        if c["child_id"] in seen:
            raise ValueError(f"child_id trùng: {c['child_id']}")
        seen.add(c["child_id"])
    pmap = {}
    for p in parents:
        if not isinstance(p.get("parent_id"), str) or not p.get("text"):
            raise ValueError(f"parent thiếu parent_id/text: {p.get('parent_id')}")
        if p["parent_id"] in pmap:
            raise ValueError(f"parent_id trùng: {p['parent_id']}")
        pmap[p["parent_id"]] = p
    children = [{**CHILD_DEFAULTS, **{k: v for k, v in c.items()}} for c in children]
    for c in children:
        c["breadcrumb"] = c["breadcrumb"] or ""
        c["doc_id"] = c["doc_id"] or "doc"
        if c["parent_id"] not in pmap:
            raise ValueError(f"{c['child_id']}: parent không tồn tại: {c['parent_id']}")
    parents = [{**PARENT_DEFAULTS, **p} for p in parents]
    for p in parents:
        p["doc_id"] = p["doc_id"] or "doc"
    return children, parents


def _minmax(values):
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    return {k: ((v - lo) / (hi - lo) if hi > lo else (1.0 if hi > 0 else 0.0)) for k, v in values.items()}


class Collection:
    def __init__(self, out_dir, manifest, children, parents, emb, tokens, tokenizer_mode,
                 device=None, embedder=None, reranker=None):
        import faiss
        from rank_bm25 import BM25Okapi
        self.out_dir, self.manifest, self.device = Path(out_dir), manifest, device
        self.children, self.parents = children, {p["parent_id"]: p for p in parents}
        self.tokenizer_mode = tokenizer_mode
        self.emb = emb
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.bm25 = BM25Okapi(tokens["plain"])
        self.bm25s = BM25Okapi(tokens["stripped"])
        self.siblings = {}
        for i, c in enumerate(children):
            self.siblings.setdefault(c["parent_id"], []).append(i)
        self._embedder, self._reranker = embedder, reranker
        self._owns = {"embedder": embedder is None, "reranker": reranker is None}

    # ---- build / open ----
    @classmethod
    def build(cls, children, parents, out_dir, device=None, batch_size=4, embedder=None, reranker=None,
              tokenizer_mode="underthesea"):
        import numpy as np
        out_dir = Path(out_dir)
        if out_dir.exists():
            raise FileExistsError(f"{out_dir} đã tồn tại; dùng thư mục mới.")
        cdata, pdata = _read_rows(children), _read_rows(parents)
        rows, prows = _normalize(_parse(cdata, "children"), _parse(pdata, "parents"))
        if embedder is None:
            from .embedder import Embedder
            embedder = Embedder(device=device)
        vectors = np.ascontiguousarray(
            embedder.encode([child_text(c) for c in rows], batch_size=batch_size), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not np.isfinite(vectors).all() \
                or np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise ValueError("Embeddings sai kích thước, NaN/Inf hoặc vector rỗng.")
        tokens = {"plain": [tokenize(c["text"], tokenizer_mode) or ["_"] for c in rows],
                  "stripped": [_tok_stripped(c["text"]) or ["_"] for c in rows]}
        out_dir.mkdir(parents=True)
        (out_dir / "children.jsonl").write_bytes(cdata)
        (out_dir / "parents.jsonl").write_bytes(pdata)
        np.save(out_dir / "embeddings.npy", vectors, allow_pickle=False)
        (out_dir / "tokens.json").write_text(json.dumps(tokens), encoding="utf-8")
        manifest = {"format_version": FORMAT_VERSION, "num_children": len(rows), "num_parents": len(prows),
                    "dimension": int(vectors.shape[1]), "model_name": MODEL_NAME,
                    "embedder_revision": getattr(embedder, "resolved_revision", None),
                    "tokenizer": tokenizer_mode, "tokenizer_format": TOKENIZER_FORMAT,
                    "tokenizer_version": tokenizer_version(tokenizer_mode),
                    "sha256": {n: sha256_lf((out_dir / n).read_bytes()) for n in FILES}}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(out_dir, manifest, rows, prows, vectors, tokens, tokenizer_mode, device, embedder, reranker)

    @classmethod
    def open(cls, out_dir, device=None, embedder=None, reranker=None):
        import numpy as np
        out_dir = Path(out_dir)
        mpath = out_dir / "manifest.json"
        if not mpath.exists():
            raise ValueError(f"{out_dir}: thiếu manifest.json; không phải Collection hợp lệ.")
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("format_version") != FORMAT_VERSION or m.get("tokenizer_format") != TOKENIZER_FORMAT:
            raise ValueError("Collection format không tương thích; cần rebuild.")
        if tokenizer_version(m["tokenizer"]) != m["tokenizer_version"]:
            raise ValueError("Phiên bản tokenizer (underthesea) đã đổi; cần rebuild Collection.")
        for name, want in m["sha256"].items():
            if not (out_dir / name).exists() or sha256_lf((out_dir / name).read_bytes()) != want:
                raise ValueError(f"{name} đã thay đổi hoặc thiếu so với manifest; cần rebuild Collection.")
        children, parents = _normalize(_parse((out_dir / "children.jsonl").read_bytes().replace(b"\r\n", b"\n"), "children"),
                                       _parse((out_dir / "parents.jsonl").read_bytes().replace(b"\r\n", b"\n"), "parents"))
        if (len(children), len(parents)) != (m["num_children"], m["num_parents"]):
            raise ValueError("Số lượng children/parents không khớp manifest.")
        emb = np.ascontiguousarray(np.load(out_dir / "embeddings.npy", allow_pickle=False), dtype="float32")
        tokens = json.loads((out_dir / "tokens.json").read_text(encoding="utf-8"))
        if emb.shape[0] != len(children) or len(tokens["plain"]) != len(children):
            raise ValueError("Embeddings/tokens không khớp children.")
        return cls(out_dir, m, children, parents, emb, tokens, m["tokenizer"], device, embedder, reranker)

    # ---- lazy models ----
    def _get_embedder(self):
        if self._embedder is None:
            from .embedder import Embedder
            self._embedder = Embedder(device=self.device)
        want, got = self.manifest.get("embedder_revision"), getattr(self._embedder, "resolved_revision", None)
        if want and got and want != got:
            raise ValueError(f"Embedder revision {got} khác lúc build ({want}); cần rebuild.")
        return self._embedder

    def _get_reranker(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            cuda = (self.device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")
            kw = {"model_kwargs": {"torch_dtype": torch.float16}} if cuda else {}
            self._reranker = CrossEncoder(RERANKER_NAME, max_length=1024,
                                          device="cuda" if cuda else "cpu", **kw)
        return self._reranker

    def close(self):
        """Release owned models and GPU cache. Idempotent."""
        if self._owns["embedder"]:
            self._embedder = None
        if self._owns["reranker"]:
            self._reranker = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- search ----
    def search(self, query, top_k=10, candidate_k=100, fusion="rrf", alpha=0.7, rerank=False,
               rerank_pool=50, max_parents=5, neighbors=False):
        import numpy as np
        if not query.strip() or top_k < 1 or candidate_k < top_k or max_parents < 1 or rerank_pool < 1:
            raise ValueError("Query không rỗng; candidate_k >= top_k >= 1; max_parents, rerank_pool >= 1.")
        if fusion not in ("rrf", "convex") or not 0 <= alpha <= 1:
            raise ValueError("fusion phải là 'rrf'|'convex'; alpha trong [0,1].")
        t0 = time.perf_counter()
        tm = {}
        n = len(self.children)
        k = min(candidate_k, n)
        q = np.ascontiguousarray(self._get_embedder().encode_query(query), dtype="float32")
        tm["embed"] = time.perf_counter() - t0
        t = time.perf_counter()
        sims, ids = self.index.search(q, k)
        dense = sorted(((int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0),
                       key=lambda x: (-x[1], x[0]))  # faiss tie order is arbitrary
        tm["dense"] = time.perf_counter() - t
        t = time.perf_counter()
        lists = {"dense": dense}
        for name, engine, toks in (("bm25", self.bm25, tokenize(query, self.tokenizer_mode)),
                                   ("bm25_stripped", self.bm25s, _tok_stripped(query))):
            sc = engine.get_scores(toks) if toks else np.zeros(n)
            order = np.argsort(-sc, kind="stable")[:k]  # stable: ties keep corpus order
            lists[name] = [(int(i), float(sc[i])) for i in order if sc[i] > 0]
        tm["bm25"] = time.perf_counter() - t
        t = time.perf_counter()
        fused = {}
        if fusion == "rrf":
            for hits in lists.values():
                for rank, (i, _) in enumerate(hits, 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        else:
            nd = _minmax(dict(lists["dense"]))
            nb, ns = _minmax(dict(lists["bm25"])), _minmax(dict(lists["bm25_stripped"]))
            for i in set(nd) | set(nb) | set(ns):
                fused[i] = alpha * nd.get(i, 0.0) + (1 - alpha) * max(nb.get(i, 0.0), ns.get(i, 0.0))
        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        score = dict(fused)
        per = {name: dict(h) for name, h in lists.items()}
        tm["fuse"] = time.perf_counter() - t
        t = time.perf_counter()
        rr = {}
        if rerank and ordered:
            pool = ordered[:rerank_pool]
            raw = self._get_reranker().predict([(query, child_text(self.children[i])) for i in pool],
                                               batch_size=16, show_progress_bar=False)
            rr = {i: float(s) for i, s in zip(pool, np.asarray(raw, dtype="float64").reshape(-1))}
            ordered = sorted(pool, key=lambda i: (-rr[i], ordered.index(i)))
            score = rr
        tm["rerank"] = time.perf_counter() - t

        def child_out(i):
            c = self.children[i]
            sc = {"dense": per["dense"].get(i, float(self.emb[i] @ q[0])),
                  "bm25": per["bm25"].get(i), "bm25_stripped": per["bm25_stripped"].get(i)}
            if i in rr:
                sc["rerank"] = rr[i]
            return {"child_id": c["child_id"], "parent_id": c["parent_id"], "score": score[i], "scores": sc}

        out_children = [child_out(i) for i in ordered[:top_k]]
        groups = {}
        for i in ordered:
            groups.setdefault(self.children[i]["parent_id"], []).append(i)
        extra = []
        if neighbors and ordered:
            best = ordered[0]
            pid = self.children[best]["parent_id"]
            if len(self.siblings[pid]) < 3:
                doc = self.children[best]["doc_id"]
                extra = [j for j in (best - 1, best + 1) if 0 <= j < n and self.children[j]["doc_id"] == doc]
                groups[pid] = groups[pid] + [j for j in extra if j not in groups[pid]]
        parents = []
        for pid, idxs in sorted(groups.items(), key=lambda kv: (-max(score.get(i, float("-inf")) for i in kv[1]),
                                                                  min(kv[1]))):
            if len(parents) >= max_parents:
                break
            p = self.parents[pid]
            parents.append({"parent_id": pid, "doc_id": p["doc_id"], "title": p["title"],
                            "breadcrumb": p["breadcrumb"], "text": p["text"], "page_start": p["page_start"],
                            "page_end": p["page_end"], "score": max(score.get(i, float("-inf")) for i in idxs),
                            "matched_child_ids": [self.children[i]["child_id"] for i in idxs]})
        tm["total"] = time.perf_counter() - t0
        top = out_children[0] if out_children else None
        second = score[ordered[1]] if len(ordered) > 1 else 0.0
        res = {"children": out_children, "parents": parents,
               "timings_ms": {k_: v * 1000 for k_, v in tm.items()},
               "top1": {"dense_cosine": top["scores"]["dense"] if top else None,
                        "fused": top["score"] if top else None,
                        "margin": (top["score"] - second) if top else None}}
        if neighbors:
            res["neighbors"] = [{"child_id": self.children[j]["child_id"], "parent_id": self.children[j]["parent_id"],
                                 "text": self.children[j]["text"]} for j in extra]
        return res
