"""Run from RAG/: python -m unittest tests.test_ingest (no models, no network)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ingest as ing  # noqa: E402
from parser.chunker import ChunkingError  # noqa: E402

CT = lambda s: len(s.split())


def fake_parse(path, dest, force):
    text = Path(path).read_text(encoding="utf-8")
    if "BAD" in text:
        raise ChunkingError("bad doc")
    blocks = [{"id": f"b{i}", "type": "paragraph", "text": t, "level": 0, "location": {"page_num": 1},
               "bbox": None, "metadata": {}} for i, t in enumerate(text.split("\n\n"))]
    return {"blocks": blocks, "stats": {}}


class QdrantIndexTest(unittest.TestCase):
    def test_name_and_skip_if_unchanged(self):
        self.assertEqual(ing.qdrant_name("D:/a/Tài Liệu 1"), "rag_tai_lieu_1")
        from embed.collection import _normalize, _parse, _read_rows
        from embed.qdrant_collection import _corpus_hash
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "corpus"
            c.mkdir()
            ing.write_jsonl([{"child_id": "c1", "parent_id": "p1", "text": "a", "doc_id": "d"}], c / "children.jsonl")
            ing.write_jsonl([{"parent_id": "p1", "text": "a", "doc_id": "d"}], c / "parents.jsonl")
            sha = _corpus_hash(*_normalize(_parse(_read_rows(c / "children.jsonl"), "c"),
                                           _parse(_read_rows(c / "parents.jsonl"), "p")))

            class Client:
                def __init__(self, sha):
                    self.sha = sha

                def collection_exists(self, n):
                    return True

                def retrieve(self, n, ids):
                    return [type("P", (), {"payload": {"sha256": self.sha}})()]

            built = []

            def builder(ch_, pa_, device=None):
                built.append(1)
                return type("C", (), {"close": lambda s: None})()

            self.assertFalse(ing.rebuild_index(d, backend="qdrant", client=Client(sha), builder=builder)[0])
            self.assertEqual(built, [])
            self.assertTrue(ing.rebuild_index(d, backend="qdrant", client=Client("other"), builder=builder)[0])
            self.assertTrue(ing.rebuild_index(d, backend="qdrant", client=Client(sha), builder=builder,
                                              force=True)[0])
            self.assertEqual(len(built), 2)


class IngestTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.dir = Path(self._t.name)
        self.store = self.dir / "store"
        self.addCleanup(self._t.cleanup)

    def doc(self, name, text):
        p = self.dir / "docs" / name
        p.parent.mkdir(exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def run_ingest(self, paths, **kw):
        return ing.ingest_paths(paths, self.store, CT, parse_fn=fake_parse, out=lambda s: None, **kw)

    def rows(self, name="children.jsonl"):
        return ing._read_jsonl(self.store / "corpus" / name)

    def test_slug_and_doc_id(self):
        self.assertEqual(ing.slugify("Quyết định Đào tạo_2024"), "quyet-dinh-dao-tao-2024")
        self.assertEqual(ing.make_doc_id("Báo cáo.pdf", "abcdef123"), "bao-cao-abcdef")

    def test_idempotent_by_sha_and_force(self):
        p = self.doc("a.md", "xin chào thế giới\n\nđoạn hai")
        self.assertEqual(self.run_ingest([p])[0]["status"], "ok")
        n = len(self.rows())
        self.assertEqual(self.run_ingest([p])[0]["status"], "skip")
        self.assertEqual(len(self.rows()), n)
        self.assertEqual(self.run_ingest([p], force=True)[0]["status"], "ok")
        self.assertEqual(len(self.rows()), n)  # replaced, not duplicated

    def test_reingest_changed_file_replaces_old_doc(self):
        p = self.doc("a.md", "version một\n\nđoạn")
        old = self.run_ingest([p])[0]["doc_id"]
        p.write_text("version hai khác hẳn\n\nđoạn mới", encoding="utf-8")
        new = self.run_ingest([p])[0]
        self.assertNotEqual(old, new["doc_id"])
        self.assertEqual({r["doc_id"] for r in self.rows()}, {new["doc_id"]})
        self.assertEqual({r["doc_id"] for r in self.rows("parents.jsonl")}, {new["doc_id"]})
        self.assertEqual(list(ing.load_registry(self.store)), [new["doc_id"]])

    def test_merge_keeps_other_docs(self):
        self.run_ingest([self.doc("a.md", "alpha một"), self.doc("b.md", "beta hai")])
        self.assertEqual(len({r["doc_id"] for r in self.rows()}), 2)

    def test_directory_expansion(self):
        self.doc("a.md", "x")
        self.doc("b.xyz", "x")
        sub = self.dir / "docs" / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("x", encoding="utf-8")
        self.assertEqual([p.name for p in ing.expand_paths([self.dir / "docs"])], ["a.md", "c.txt"])

    def test_failure_isolation(self):
        good, bad = self.doc("good.md", "nội dung tốt"), self.doc("bad.md", "BAD")
        res = self.run_ingest([bad, good, self.dir / "missing.md"])
        self.assertEqual([r["status"] for r in res], ["failed", "ok", "failed"])
        self.assertEqual(len(self.rows()), 1)

    def test_rebuild_index_skips_when_current(self):
        self.run_ingest([self.doc("a.md", "alpha một")])
        calls = []

        class Col:
            def close(self):
                pass

        def builder(children, parents, out_dir, device=None):
            from embed.collection import sha256_lf
            calls.append(out_dir)
            out_dir.mkdir()
            c = self.store / "corpus"
            (out_dir / "manifest.json").write_text(json.dumps({"sha256": {
                n: sha256_lf((c / n).read_bytes()) for n in ing.CORPUS_FILES}}), encoding="utf-8")
            return Col()

        self.assertTrue(ing.rebuild_index(self.store, builder=builder)[0])
        self.assertFalse(ing.rebuild_index(self.store, builder=builder)[0])
        self.assertEqual(len(calls), 1)
        self.run_ingest([self.doc("b.md", "beta hai")])
        self.assertTrue(ing.rebuild_index(self.store, builder=builder)[0])
        self.assertFalse((self.store / "index.tmp").exists())

    def test_format_table(self):
        res = self.run_ingest([self.doc("a.md", "alpha một"), self.dir / "nope.md"])
        t = ing.format_table(res, 2.0)
        self.assertIn("LỖI", t)
        self.assertIn("max tok", t)


if __name__ == "__main__":
    unittest.main()
