"""Run from RAG/: python -m unittest tests.test_qdrant_collection (in-memory Qdrant, fake models).
Optional live test against deploy/docker-compose Qdrant: set RAG_INTEGRATION=1."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient, models as qm  # noqa: E402

from embed.collection import Collection  # noqa: E402
from embed.qdrant_collection import BATCH, QdrantCollection, point_id  # noqa: E402
from tests.test_collection import FakeEmbedder, FakeReranker, corpus  # noqa: E402


class QdrantCollectionTest(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(location=":memory:")
        self.addCleanup(self.client.close)

    def build(self, spec, doc="d", name="t", **kw):
        ch, pa = corpus(spec, doc)
        return QdrantCollection.build(ch, pa, name=name, client=self.client, embedder=FakeEmbedder(),
                                      tokenizer_mode="regex", **kw)

    def test_roundtrip(self):
        col = self.build([("c0", "p0", "Sinh viên đăng ký học phần"), ("c1", "p1", "Học phí và học bổng"),
                          ("c2", "p1", "Tốt nghiệp đại học")])
        res = col.search("đăng ký học phần", top_k=3)
        res2 = QdrantCollection.open("t", client=self.client, embedder=FakeEmbedder()).search("đăng ký học phần", top_k=3)
        ids = lambda r: ([c["child_id"] for c in r["children"]], [p["parent_id"] for p in r["parents"]])
        self.assertEqual(ids(res), ids(res2))  # scores differ ~1e-7 between calls (float32 scoring)
        self.assertEqual(res["children"][0]["child_id"], "c0")
        self.assertEqual(res["parents"][0]["text"], "p0")

    def test_same_parents_as_faiss(self):
        spec = [("c0", "p0", "Sinh viên đăng ký học phần"), ("c1", "p1", "Quy định về học bổng"),
                ("c2", "p2", "Điều kiện tốt nghiệp đại học"), ("c3", "p2", "Nghỉ học tạm thời"),
                ("c4", "p3", "Học phí mỗi học kỳ")]
        ch, pa = corpus(spec)
        with tempfile.TemporaryDirectory() as tmp:
            faiss_col = Collection.build(ch, pa, Path(tmp) / "c", embedder=FakeEmbedder(), tokenizer_mode="regex")
            qcol = self.build(spec)
            for q in ("đăng ký học phần", "học phí", "tốt nghiệp"):
                a = [p["parent_id"] for p in faiss_col.search(q, fusion="rrf")["parents"]]
                b = [p["parent_id"] for p in qcol.search(q, fusion="rrf")["parents"]]
                self.assertEqual(a, b, q)

    def test_accent_stripped_query(self):
        col = self.build([("c0", "p0", "Sinh viên đăng ký học phần"), ("c1", "p1", "Quy định về học bổng"),
                          ("c2", "p2", "Điều kiện tốt nghiệp"), ("c3", "p3", "Nghỉ học tạm thời")])
        top = col.search("dang ky hoc phan", fusion="convex", alpha=0.0)["children"][0]
        self.assertEqual(top["child_id"], "c0")
        self.assertIsNone(top["scores"]["bm25"])
        self.assertGreater(top["scores"]["bm25_stripped"], 0)

    def test_filter_doc_ids(self):
        ch, pa = corpus([("a0", "pa", "needle haystack")], "docA")
        ch2, pa2 = corpus([("b0", "pb", "needle haystack"), ("b1", "pb2", "other text")], "docB")
        col = QdrantCollection.build(ch + ch2, pa + pa2, name="t", client=self.client, embedder=FakeEmbedder(),
                                     tokenizer_mode="regex")
        self.assertEqual({p["doc_id"] for p in col.search("needle haystack")["parents"]}, {"docA", "docB"})
        for sf in (False, True):
            res = col.search("needle haystack", filter_doc_ids=["docB"], server_fusion=sf)
            self.assertEqual({p["doc_id"] for p in res["parents"]}, {"docB"})
            self.assertEqual(res["children"][0]["child_id"], "b0")

    def test_server_fusion(self):
        col = self.build([("c0", "p0", "needle haystack"), ("c1", "p1", "other words"), ("c2", "p2", "more")])
        res = col.search("needle haystack", server_fusion=True)
        self.assertEqual(res["children"][0]["child_id"], "c0")
        self.assertEqual(res["parents"][0]["parent_id"], "p0")

    def test_rerank_before_grouping(self):
        spec = [(f"c{i}", f"p{i}", "alpha beta") for i in range(4)] + \
               [("c4", "p4", "alpha beta zzz gamma delta eps zeta eta theta")]
        col = self.build(spec)
        base = [p["parent_id"] for p in col.search("alpha beta", max_parents=2)["parents"]]
        self.assertNotIn("p4", base)
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
        col = self.build([("c0", "p0", "x"), ("c1", "p1", "needle a"), ("c2", "p1", "b"), ("c3", "p1", "c")])
        self.assertEqual(col.search("needle", neighbors=True)["neighbors"], [])

    def test_manifest_mismatch(self):
        self.build([("c0", "p0", "xin chào"), ("c1", "p1", "tạm biệt")])
        self.client.set_payload("t", payload={"text": "bị sửa"}, points=[point_id("c0")])
        with self.assertRaisesRegex(ValueError, "sha256"):
            QdrantCollection.open("t", client=self.client, embedder=FakeEmbedder())
        self.build([("c0", "p0", "xin chào")])
        self.client.set_payload("t_meta", payload={"format_version": 99}, points=[1])
        with self.assertRaisesRegex(ValueError, "format"):
            QdrantCollection.open("t", client=self.client)
        with self.assertRaisesRegex(ValueError, "thiếu collection"):
            QdrantCollection.open("nope", client=self.client)

    def test_deterministic_ids_and_rebuild(self):
        self.assertEqual(point_id("c0"), point_id("c0"))
        self.assertNotEqual(point_id("c0"), point_id("c1"))
        self.build([("c0", "p0", "one"), ("c1", "p1", "two")])
        self.build([("c0", "p0", "one")])  # rebuild recreates, no leftovers
        self.assertEqual(self.client.count("t").count, 1)
        self.assertEqual(self.client.retrieve("t", ids=[point_id("c0")])[0].payload["child_id"], "c0")

    def test_batch_upsert_over_256(self):
        n = BATCH + 44
        col = self.build([(f"c{i}", f"p{i}", f"word{i} common") for i in range(n)])
        self.assertEqual(self.client.count("t").count, n)
        self.assertEqual(col.search("word299", top_k=1)["children"][0]["child_id"], "c299")

    def test_close_idempotent(self):
        col = self.build([("c0", "p0", "hello world")])
        col.close()
        col.close()
        self.assertEqual(col.search("hello")["children"][0]["child_id"], "c0")


@unittest.skipUnless(os.environ.get("RAG_INTEGRATION") == "1", "set RAG_INTEGRATION=1 (needs Qdrant on :6333)")
class QdrantLiveTest(unittest.TestCase):
    def test_live_roundtrip(self):
        client = QdrantClient(url=os.environ.get("RAG_QDRANT_URL", "http://127.0.0.1:6333"))
        ch, pa = corpus([("c0", "p0", "needle haystack"), ("c1", "p1", "other words")])
        col = QdrantCollection.build(ch, pa, name="rag_test_tmp", client=client, embedder=FakeEmbedder(),
                                     tokenizer_mode="regex")
        try:
            QdrantCollection.open("rag_test_tmp", client=client, embedder=FakeEmbedder())
            for sf in (False, True):
                self.assertEqual(col.search("needle", server_fusion=sf)["children"][0]["child_id"], "c0")
        finally:
            for s in ("", "_parents", "_meta"):
                client.delete_collection("rag_test_tmp" + s)


if __name__ == "__main__":
    unittest.main()
