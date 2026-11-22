"""Run from RAG/: python -m unittest tests.test_collection (fake embedder/reranker, no downloads)."""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed.collection import Collection, strip_accents  # noqa: E402

LEGACY = Path(__file__).resolve().parents[1] / "data" / "processed"


class FakeEmbedder:
    resolved_revision = "fake"

    def encode(self, texts, batch_size=4, show_progress=True):
        out = np.zeros((len(texts), 64), dtype="float32")
        for r, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_query(self, query):
        return self.encode([query])


class FakeReranker:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return np.array([self.fn(q, t) for q, t in pairs])


def corpus(spec, doc="d"):
    """spec: list of (child_id, parent_id, text)."""
    children = [{"child_id": c, "parent_id": p, "doc_id": doc, "breadcrumb": "B", "text": t} for c, p, t in spec]
    parents = [{"parent_id": p, "doc_id": doc, "title": p, "breadcrumb": "B", "text": p}
               for p in dict.fromkeys(p for _, p, _ in spec)]
    return children, parents


class StubIndex:
    def __init__(self, sims):
        self.sims = sims

    def search(self, q, k):
        order = sorted(range(len(self.sims)), key=lambda i: (-self.sims[i], i))[:k]
        return np.array([[self.sims[i] for i in order]], dtype="float32"), np.array([order])


class StubBM25:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype="float64")

    def get_scores(self, toks):
        return self.scores


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def out(self):
        self.n += 1
        return Path(self.tmp.name) / f"c{self.n}"

    def build(self, spec, **kw):
        ch, pa = corpus(spec)
        return Collection.build(ch, pa, self.out(), embedder=FakeEmbedder(), tokenizer_mode="regex", **kw)

    def test_roundtrip(self):
        col = self.build([("c0", "p0", "Sinh viên đăng ký học phần"), ("c1", "p1", "Học phí và học bổng"),
                          ("c2", "p1", "Tốt nghiệp đại học")])
        res = col.search("đăng ký học phần", top_k=3)
        col2 = Collection.open(col.out_dir, embedder=FakeEmbedder())
        res2 = col2.search("đăng ký học phần", top_k=3)
        for r in (res, res2):
            r.pop("timings_ms")
        self.assertEqual(res, res2)
        self.assertEqual(res["children"][0]["child_id"], "c0")

    def test_accent_stripped_query(self):
        self.assertEqual(strip_accents("Đăng ký HỌC"), "dang ky hoc")
        col = self.build([("c0", "p0", "Sinh viên đăng ký học phần"), ("c1", "p1", "Quy định về học bổng"),
                          ("c2", "p2", "Điều kiện tốt nghiệp"), ("c3", "p3", "Nghỉ học tạm thời")])
        res = col.search("dang ky hoc phan", fusion="convex", alpha=0.0)  # lexical only (dense is a fake)
        top = res["children"][0]
        self.assertEqual(top["child_id"], "c0")
        self.assertIsNone(top["scores"]["bm25"])  # plain bm25 fails on unaccented query
        self.assertGreater(top["scores"]["bm25_stripped"], 0)

    def test_rrf_vs_convex(self):
        col = self.build([("c0", "p0", "a"), ("c1", "p1", "b"), ("c2", "p2", "c"), ("c3", "p3", "d")])
        col.index = StubIndex([0.90, 0.89, 0.10, 0.0])
        col.bm25 = StubBM25([1, 9, 10, 0])
        col.bm25s = StubBM25([0, 0, 0, 0])
        rrf = [c["child_id"] for c in col.search("x y", top_k=4, fusion="rrf")["children"]]
        cvx = [c["child_id"] for c in col.search("x y", top_k=4, fusion="convex", alpha=0.7)["children"]]
        self.assertEqual(rrf[:3], ["c0", "c2", "c1"])  # c0/c2 tie broken by corpus index
        self.assertEqual(cvx[:3], ["c1", "c0", "c2"])

    def test_parent_grouping_by_max(self):
        col = self.build([("c0", "p0", "alpha beta gamma"), ("c1", "p0", "alpha"), ("c2", "p1", "beta delta")])
        res = col.search("alpha beta gamma", top_k=3)
        score = {c["child_id"]: c["score"] for c in res["children"]}
        p0 = next(p for p in res["parents"] if p["parent_id"] == "p0")
        self.assertEqual(p0["score"], max(score["c0"], score["c1"]))
        self.assertEqual(set(p0["matched_child_ids"]), {"c0", "c1"})
        self.assertEqual(len({p["parent_id"] for p in res["parents"]}), len(res["parents"]))

    def test_rerank_before_grouping(self):
        spec = [(f"c{i}", f"p{i}", "alpha beta") for i in range(4)] + \
               [("c4", "p4", "alpha beta zzz gamma delta eps zeta eta theta")]
        col = self.build(spec)
        base = [p["parent_id"] for p in col.search("alpha beta", max_parents=2)["parents"]]
        self.assertNotIn("p4", base)  # precondition: p4 is outside the top max_parents parents
        col._reranker = FakeReranker(lambda q, t: 1.0 if "zzz" in t else 0.0)
        res = col.search("alpha beta", max_parents=2, rerank=True, rerank_pool=5)
        self.assertEqual(res["parents"][0]["parent_id"], "p4")
        self.assertIn("rerank", res["children"][0]["scores"])
        small = col.search("alpha beta", max_parents=2, rerank=True, rerank_pool=2)
        self.assertNotIn("p4", [p["parent_id"] for p in small["parents"]])

    def test_neighbors(self):
        col = self.build([("c0", "p0", "intro text"), ("c1", "p1", "needle haystack"), ("c2", "p2", "closing text")])
        res = col.search("needle haystack", neighbors=True)
        self.assertEqual({n["child_id"] for n in res["neighbors"]}, {"c0", "c2"})
        self.assertNotIn("neighbors", col.search("needle haystack"))
        # parent with >= 3 children gets no neighbours
        col = self.build([("c0", "p0", "x"), ("c1", "p1", "needle a"), ("c2", "p1", "b"), ("c3", "p1", "c")])
        self.assertEqual(col.search("needle", neighbors=True)["neighbors"], [])

    def test_manifest_crlf_independent(self):
        ch, pa = corpus([("c0", "p0", "xin chào"), ("c1", "p1", "tạm biệt")])
        lf = Path(self.tmp.name) / "lf"
        lf.mkdir()
        for name, rows in (("children.jsonl", ch), ("parents.jsonl", pa)):
            data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
            (lf / name).write_bytes(data)
            (lf / ("crlf_" + name)).write_bytes(data.replace(b"\n", b"\r\n"))
        a = Collection.build(lf / "children.jsonl", lf / "parents.jsonl", self.out(),
                             embedder=FakeEmbedder(), tokenizer_mode="regex")
        b = Collection.build(lf / "crlf_children.jsonl", lf / "crlf_parents.jsonl", self.out(),
                             embedder=FakeEmbedder(), tokenizer_mode="regex")
        self.assertEqual(a.manifest["sha256"], b.manifest["sha256"])
        p = a.out_dir / "children.jsonl"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        Collection.open(a.out_dir, embedder=FakeEmbedder())  # CRLF checkout still verifies
        p.write_bytes(p.read_bytes() + b'{"x": 1}\n')
        with self.assertRaisesRegex(ValueError, "children.jsonl"):
            Collection.open(a.out_dir, embedder=FakeEmbedder())

    def test_legacy_corpus_loads(self):
        col = Collection.build(LEGACY / "children.jsonl", LEGACY / "parents.jsonl", self.out(),
                               embedder=FakeEmbedder(), tokenizer_mode="regex")
        self.assertEqual((len(col.children), len(col.parents)), (225, 51))
        c = col.children[0]
        self.assertEqual((c["doc_id"], c["page_start"], c["bbox"]), ("doc", None, None))
        res = col.search("sinh viên đăng ký học phần")
        self.assertEqual(res["parents"][0]["doc_id"], "doc")

    def test_tie_break_by_index_not_id_string(self):
        col = self.build([("d2", "p0", "same words"), ("d10", "p1", "same words"), ("d1", "p2", "same words")])
        for kw in ({"fusion": "rrf"}, {"fusion": "convex"}):
            ids = [c["child_id"] for c in col.search("same words", **kw)["children"]]
            self.assertEqual(ids, ["d2", "d10", "d1"])

    def test_close_idempotent(self):
        col = self.build([("c0", "p0", "hello world")])
        col.close()
        col.close()
        self.assertEqual(col.search("hello")["children"][0]["child_id"], "c0")


if __name__ == "__main__":
    unittest.main()
