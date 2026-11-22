"""Qdrant-backed Collection: same public API/return shape as embed.collection.Collection.

Storage (build() DROPS and recreates all three, so rebuild is idempotent; upserts in batches of 256):
  <name>          children. Named vectors: 'dense' (cosine), sparse 'bm25' and 'bm25_stripped' (Modifier.IDF).
                  Payload = full normalized child row + 'ord' (corpus order, also the tie-break/neighbour key).
  <name>_parents  one point per parent (dummy 1-d vector), payload = parent row + 'ord'; open() needs only Qdrant.
  <name>_meta     one point (id 1) with the manifest.
Point id = first 8 bytes of sha256(id string) as uint64. Sparse index = zlib.crc32(token) (32-bit, deterministic;
a collision merges two tokens' weights: negligible for this corpus). Sparse value = BM25 tf saturation
tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)); Qdrant multiplies by IDF at query time; query tokens weigh 1.0 each.
"""
import json
import time
import warnings
import zlib
from collections import Counter

from .bm25 import TOKENIZER_FORMAT, tokenize, tokenizer_version
from .collection import (FORMAT_VERSION, Collection, MODEL_NAME, _minmax, _normalize, _parse, _read_rows,
                         _tok_stripped, sha256_lf)
from .embedder import child_text

K1, B, BATCH = 1.5, 0.75, 256
SPARSE = ("bm25", "bm25_stripped")


def point_id(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def _sparse(tokens, avgdl=None):
    """tokens -> (indices, values). avgdl None = query side (weight 1.0 per distinct token)."""
    acc = {}
    dl = len(tokens)
    for t, tf in Counter(tokens).items():
        w = 1.0 if avgdl is None else tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        i = zlib.crc32(t.encode("utf-8"))
        acc[i] = acc.get(i, 0.0) + w
    return list(acc), list(acc.values())


def _corpus_hash(children, parents):
    f = lambda rows: "".join(json.dumps({k: v for k, v in r.items() if k != "ord"}, ensure_ascii=False,
                                        sort_keys=True) + "\n" for r in rows).encode("utf-8")
    return {"children": sha256_lf(f(children)), "parents": sha256_lf(f(parents))}


def _scroll_all(client, name):
    rows, off = [], None
    while True:
        pts, off = client.scroll(name, limit=1000, offset=off, with_payload=True)
        rows += [p.payload for p in pts]
        if off is None:
            return sorted(rows, key=lambda r: r["ord"])


class QdrantCollection:
    # lazy models / close are model-only logic: reuse Collection's (need _embedder, _reranker, _owns, device, manifest)
    _get_embedder = Collection._get_embedder
    _get_reranker = Collection._get_reranker
    close = Collection.close

    def __init__(self, client, name, manifest, device=None, embedder=None, reranker=None):
        self.client, self.name, self.manifest, self.device = client, name, manifest, device
        self.tokenizer_mode = manifest["tokenizer"]
        self.n = manifest["num_children"]
        self._embedder, self._reranker = embedder, reranker
        self._owns = {"embedder": embedder is None, "reranker": reranker is None}

    @staticmethod
    def _client(client, settings):
        if client is not None:
            return client
        from qdrant_client import QdrantClient
        if settings is None:
            from store.settings import StoreSettings
            settings = StoreSettings.from_env()
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    # ---- build / open ----
    @classmethod
    def build(cls, children, parents, name=None, client=None, settings=None, device=None, batch_size=4,
              embedder=None, reranker=None, tokenizer_mode="underthesea"):
        import numpy as np
        from qdrant_client import models as m
        if name is None:
            if settings is None:
                from store.settings import StoreSettings
                settings = StoreSettings.from_env()
            name = settings.qdrant_collection
        client = cls._client(client, settings)
        rows, prows = _normalize(_parse(_read_rows(children), "children"), _parse(_read_rows(parents), "parents"))
        if embedder is None:
            from .embedder import Embedder
            embedder = Embedder(device=device)
        vectors = np.ascontiguousarray(
            embedder.encode([child_text(c) for c in rows], batch_size=batch_size), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not np.isfinite(vectors).all() \
                or np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise ValueError("Embeddings sai kích thước, NaN/Inf hoặc vector rỗng.")
        ids = [point_id(c["child_id"]) for c in rows]
        pids = [point_id(p["parent_id"]) for p in prows]
        if len(set(ids)) != len(ids) or len(set(pids)) != len(pids):
            raise ValueError("Va chạm point id (uint64); đổi child_id/parent_id.")
        toks = {"bm25": [tokenize(c["text"], tokenizer_mode) or ["_"] for c in rows],
                "bm25_stripped": [_tok_stripped(c["text"]) or ["_"] for c in rows]}
        avgdl = {k: sum(map(len, v)) / len(v) for k, v in toks.items()}

        dummy = m.VectorParams(size=1, distance=m.Distance.DOT)
        for n_ in (name, name + "_parents", name + "_meta"):
            if client.collection_exists(n_):
                client.delete_collection(n_)
        client.create_collection(
            name, vectors_config={"dense": m.VectorParams(size=int(vectors.shape[1]), distance=m.Distance.COSINE)},
            sparse_vectors_config={k: m.SparseVectorParams(modifier=m.Modifier.IDF) for k in SPARSE})
        client.create_collection(name + "_parents", vectors_config=dummy)
        client.create_collection(name + "_meta", vectors_config=dummy)
        with warnings.catch_warnings():  # local (in-memory/path) mode warns that indexes are no-ops
            warnings.simplefilter("ignore", UserWarning)
            for f, s in (("doc_id", "KEYWORD"), ("parent_id", "KEYWORD"), ("ord", "INTEGER")):
                client.create_payload_index(name, f, getattr(m.PayloadSchemaType, s))

        for s in range(0, len(rows), BATCH):
            pts = []
            for i in range(s, min(s + BATCH, len(rows))):
                vec = {"dense": vectors[i].tolist()}
                for k in SPARSE:
                    ix, vl = _sparse(toks[k][i], avgdl[k])
                    vec[k] = m.SparseVector(indices=ix, values=vl)
                pts.append(m.PointStruct(id=ids[i], vector=vec, payload={**rows[i], "ord": i}))
            client.upsert(name, points=pts, wait=True)
        for s in range(0, len(prows), BATCH):
            client.upsert(name + "_parents", wait=True, points=[
                m.PointStruct(id=pids[i], vector=[1.0], payload={**prows[i], "ord": i})
                for i in range(s, min(s + BATCH, len(prows)))])
        manifest = {"format_version": FORMAT_VERSION, "num_children": len(rows), "num_parents": len(prows),
                    "dimension": int(vectors.shape[1]), "model_name": MODEL_NAME,
                    "embedder_revision": getattr(embedder, "resolved_revision", None),
                    "tokenizer": tokenizer_mode, "tokenizer_format": TOKENIZER_FORMAT,
                    "tokenizer_version": tokenizer_version(tokenizer_mode), "avgdl": avgdl,
                    "sha256": _corpus_hash(rows, prows)}
        client.upsert(name + "_meta", points=[m.PointStruct(id=1, vector=[1.0], payload=manifest)], wait=True)
        return cls(client, name, manifest, device, embedder, reranker)

    @classmethod
    def open(cls, name=None, client=None, settings=None, device=None, embedder=None, reranker=None):
        if name is None:
            if settings is None:
                from store.settings import StoreSettings
                settings = StoreSettings.from_env()
            name = settings.qdrant_collection
        client = cls._client(client, settings)
        if not all(client.collection_exists(n_) for n_ in (name, name + "_parents", name + "_meta")):
            raise ValueError(f"{name}: thiếu collection Qdrant; không phải QdrantCollection hợp lệ.")
        got = client.retrieve(name + "_meta", ids=[1])
        if not got:
            raise ValueError(f"{name}: thiếu manifest.")
        mf = got[0].payload
        if mf.get("format_version") != FORMAT_VERSION or mf.get("tokenizer_format") != TOKENIZER_FORMAT:
            raise ValueError("Collection format không tương thích; cần rebuild.")
        if tokenizer_version(mf["tokenizer"]) != mf["tokenizer_version"]:
            raise ValueError("Phiên bản tokenizer (underthesea) đã đổi; cần rebuild Collection.")
        children, parents = _scroll_all(client, name), _scroll_all(client, name + "_parents")
        if (len(children), len(parents)) != (mf["num_children"], mf["num_parents"]):
            raise ValueError("Số lượng children/parents không khớp manifest.")
        if _corpus_hash(children, parents) != mf["sha256"]:
            raise ValueError("Nội dung children/parents (sha256) đã thay đổi so với manifest; cần rebuild.")
        return cls(client, name, mf, device, embedder, reranker)

    # ---- search ----
    def search(self, query, top_k=10, candidate_k=100, fusion="rrf", alpha=0.7, rerank=False,
               rerank_pool=50, max_parents=5, neighbors=False, filter_doc_ids=None, server_fusion=False):
        """server_fusion=True: ONE request, prefetch(dense,bm25,bm25_stripped)+Fusion.RRF on the server.
        It ignores `fusion`/`alpha` (always RRF) and per-source scores (only fused rank score is returned)."""
        import numpy as np
        from qdrant_client import models as m
        if not query.strip() or top_k < 1 or candidate_k < top_k or max_parents < 1 or rerank_pool < 1:
            raise ValueError("Query không rỗng; candidate_k >= top_k >= 1; max_parents, rerank_pool >= 1.")
        if fusion not in ("rrf", "convex") or not 0 <= alpha <= 1:
            raise ValueError("fusion phải là 'rrf'|'convex'; alpha trong [0,1].")
        t0 = time.perf_counter()
        tm = {}
        k = min(candidate_k, self.n)
        flt = m.Filter(must=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(filter_doc_ids)))]) \
            if filter_doc_ids else None
        q = np.ascontiguousarray(self._get_embedder().encode_query(query), dtype="float32")
        tm["embed"] = time.perf_counter() - t0
        pay = {}  # ord -> payload

        def hits(res):
            for p in res.points:
                pay[p.payload["ord"]] = p.payload
            return [(p.payload["ord"], float(p.score)) for p in res.points]

        queries = {"dense": q[0].tolist()}
        for k_, toks in (("bm25", tokenize(query, self.tokenizer_mode)), ("bm25_stripped", _tok_stripped(query))):
            if toks:
                ix, vl = _sparse(toks)
                queries[k_] = m.SparseVector(indices=ix, values=vl)
        t = time.perf_counter()
        lists = {k_: [] for k_ in ("dense", *SPARSE)}
        if server_fusion:
            res = self.client.query_points(
                self.name, limit=k, with_payload=True, query=m.FusionQuery(fusion=m.Fusion.RRF),
                prefetch=[m.Prefetch(query=v, using=u, limit=k, filter=flt) for u, v in queries.items()])
            fused = dict(hits(res))
            lists = {k_: [] for k_ in lists}
            tm["dense"] = tm["bm25"] = 0.0
            tm["fuse"] = time.perf_counter() - t
        else:
            for u, v in queries.items():
                h = hits(self.client.query_points(self.name, query=v, using=u, limit=k,
                                                  query_filter=flt, with_payload=True))
                if u != "dense":
                    h = [x for x in h if x[1] > 0]
                lists[u] = sorted(h, key=lambda x: (-x[1], x[0]))  # server tie order is arbitrary
            tm["dense"] = tm["bm25"] = time.perf_counter() - t
            t = time.perf_counter()
            fused = {}
            if fusion == "rrf":
                for h in lists.values():
                    for rank, (i, _) in enumerate(h, 1):
                        fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
            else:
                nd = _minmax(dict(lists["dense"]))
                nb, ns = _minmax(dict(lists["bm25"])), _minmax(dict(lists["bm25_stripped"]))
                for i in set(nd) | set(nb) | set(ns):
                    fused[i] = alpha * nd.get(i, 0.0) + (1 - alpha) * max(nb.get(i, 0.0), ns.get(i, 0.0))
            tm["fuse"] = time.perf_counter() - t
        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        score = dict(fused)
        per = {k_: dict(h) for k_, h in lists.items()}
        t = time.perf_counter()
        rr = {}
        if rerank and ordered:
            pool = ordered[:rerank_pool]
            raw = self._get_reranker().predict([(query, child_text(pay[i])) for i in pool],
                                               batch_size=16, show_progress_bar=False)
            rr = {i: float(s) for i, s in zip(pool, np.asarray(raw, dtype="float64").reshape(-1))}
            ordered = sorted(pool, key=lambda i: (-rr[i], ordered.index(i)))
            score = rr
        tm["rerank"] = time.perf_counter() - t

        top_ids = ordered[:top_k]
        missing = [i for i in top_ids if i not in per["dense"]]
        if missing:  # exact cosine for children the dense list did not return (as Collection does)
            for p in self.client.retrieve(self.name, ids=[point_id(pay[i]["child_id"]) for i in missing],
                                          with_vectors=["dense"]):
                per["dense"][p.payload["ord"]] = float(np.dot(p.vector["dense"], q[0]))

        def child_out(i):
            sc = {"dense": per["dense"].get(i), "bm25": per["bm25"].get(i), "bm25_stripped": per["bm25_stripped"].get(i)}
            if i in rr:
                sc["rerank"] = rr[i]
            return {"child_id": pay[i]["child_id"], "parent_id": pay[i]["parent_id"], "score": score[i], "scores": sc}

        out_children = [child_out(i) for i in top_ids]
        groups = {}
        for i in ordered:
            groups.setdefault(pay[i]["parent_id"], []).append(i)
        extra = []
        if neighbors and ordered:
            best = ordered[0]
            pid = pay[best]["parent_id"]
            same = m.Filter(must=[m.FieldCondition(key="parent_id", match=m.MatchValue(value=pid))])
            if self.client.count(self.name, count_filter=same, exact=True).count < 3:
                nf = m.Filter(must=[m.FieldCondition(key="doc_id", match=m.MatchValue(value=pay[best]["doc_id"])),
                                    m.FieldCondition(key="ord", match=m.MatchAny(any=[best - 1, best + 1]))])
                pts, _ = self.client.scroll(self.name, scroll_filter=nf, limit=2, with_payload=True)
                for p in sorted(pts, key=lambda p: p.payload["ord"]):
                    pay[p.payload["ord"]] = p.payload
                    extra.append(p.payload["ord"])
                groups[pid] = groups[pid] + [j for j in extra if j not in groups[pid]]
        ranked = sorted(groups.items(), key=lambda kv: (-max(score.get(i, float("-inf")) for i in kv[1]), min(kv[1])))[:max_parents]
        prow = {p.payload["parent_id"]: p.payload for p in self.client.retrieve(
            self.name + "_parents", ids=[point_id(pid) for pid, _ in ranked])}
        parents = []
        for pid, idxs in ranked:
            p = prow[pid]
            parents.append({"parent_id": pid, "doc_id": p["doc_id"], "title": p["title"],
                            "breadcrumb": p["breadcrumb"], "text": p["text"], "page_start": p["page_start"],
                            "page_end": p["page_end"], "score": max(score.get(i, float("-inf")) for i in idxs),
                            "matched_child_ids": [pay[i]["child_id"] for i in idxs]})
        tm["total"] = time.perf_counter() - t0
        top = out_children[0] if out_children else None
        second = score[ordered[1]] if len(ordered) > 1 else 0.0
        res = {"children": out_children, "parents": parents,
               "timings_ms": {k_: v * 1000 for k_, v in tm.items()},
               "top1": {"dense_cosine": top["scores"]["dense"] if top else None,
                        "fused": top["score"] if top else None,
                        "margin": (top["score"] - second) if top else None}}
        if neighbors:
            res["neighbors"] = [{"child_id": pay[j]["child_id"], "parent_id": pay[j]["parent_id"],
                                 "text": pay[j]["text"]} for j in extra]
        return res
