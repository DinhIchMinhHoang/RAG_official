"""ingest/CLI engine wiring: make_router, close-before-index ordering, engine rows, flags + env defaults."""
import argparse
import builtins
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli  # noqa: E402
import ingest as ing  # noqa: E402


class MakeRouterTest(unittest.TestCase):
    def test_off_is_none(self):
        self.assertIsNone(ing.make_router("off", 45.0, "S"))

    def test_import_failure_is_none(self):
        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "parser.engine_router":
                raise ImportError("boom")
            return real(name, *a, **k)
        with mock.patch.object(builtins, "__import__", fake), self.assertLogs(level="WARNING"):
            self.assertIsNone(ing.make_router("auto", 45.0, "S"))

    def test_router_built(self):
        r = ing.make_router("glm", 12.0, "S")
        self.assertEqual((r.mode, r.budget_s, r.cache_dir), ("glm", 12.0, Path("S") / "engine_cache"))


class OrderTest(unittest.TestCase):
    def test_router_closed_before_index(self):
        events = []

        class R:
            mode = "auto"

            def close(self):
                events.append("close")

        def parse(path, dest, force, router=None):
            events.append("parse")
            return {"blocks": [{"id": "b", "type": "paragraph", "text": "hello", "location": {"page_num": 1}}],
                    "stats": {"pages": 1, "engines": {"engines": {"glm": {"state": "ok", "s": 1.0}}}}}

        def chunk(blocks, doc_id, ct, ceiling=400, doc_title=""):
            return [{"parent_id": "p", "doc_id": doc_id}], [{"child_id": "c", "doc_id": doc_id, "tokens": 3}]

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.txt"
            f.write_text("x", encoding="utf-8")
            args = argparse.Namespace(paths=[f], store=Path(tmp) / "S", tessdata=None, backend="faiss", ceiling=400,
                                      force=False, device=None, stack=False, engines="auto", engine_budget=9.0)
            real = ing.ingest_paths
            with mock.patch.object(ing, "make_router", lambda *a: events.append(("make", a[0], a[1])) or R()), \
                    mock.patch.object(ing, "ingest_paths", lambda *p, **k: real(*p, **{**k, "parse_fn": parse})), \
                    mock.patch.object(ing, "chunk_document", chunk), \
                    mock.patch.object(ing, "rebuild_index", lambda *a, **k: events.append("index") or (True, 0.1)), \
                    mock.patch.object(cli, "get_counter", lambda: len):
                out = []
                code = cli.cmd_ingest(args, out=mock.Mock(write=lambda s: out.append(s)))
        self.assertEqual(code, 0)
        self.assertEqual(events, [("make", "auto", 9.0), "parse", "close", "index"])
        self.assertIn("engine glm  ok", "".join(out))


class RowsTest(unittest.TestCase):
    def test_rows(self):
        eng = {"engines": {"glm": {"state": "ok", "s": 12.3, "formula": {"done": 25, "cached": 3, "rejected": 4, "skipped": 12},
                                   "table": {"done": 1, "cached": 0, "skipped": 0}},
                           "docling": {"state": "skipped", "reason": "python not found (set RAG_ENGINE_DOCLING_PYTHON)"}}}
        rows = ing.engine_rows([{"engines": eng}])
        self.assertEqual(rows[0], "engine glm  ok  formulas 28/44 (cached 3, rejected 4)  tables 1  12.3s")
        self.assertEqual(rows[1], "engine docling skipped: python not found (set RAG_ENGINE_DOCLING_PYTHON)")

    def test_off_no_rows(self):
        self.assertEqual(ing.engine_rows([{"engines": None}, {"status": "skip"}]), [])
        self.assertEqual(ing.engine_rows([{"engines": {"engines": {"glm": {"state": "off"}}}}]), [])


class FlagsTest(unittest.TestCase):
    def parse(self, *extra, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False):
            for k in ("RAG_ENGINES", "RAG_ENGINE_BUDGET"):
                if not env or k not in env:
                    os.environ.pop(k, None)
            return cli.build_parser().parse_args(["ingest", "x.pdf", "--store", "S", *extra])

    def test_defaults(self):
        a = self.parse()
        self.assertEqual((a.engines, a.engine_budget), ("auto", 45.0))

    def test_flags(self):
        a = self.parse("--engines", "glm, docling", "--engine-budget", "20")
        self.assertEqual((a.engines, a.engine_budget), ("glm,docling", 20.0))

    def test_env_defaults_and_flag_wins(self):
        env = {"RAG_ENGINES": "off", "RAG_ENGINE_BUDGET": "30"}
        a = self.parse(env=env)
        self.assertEqual((a.engines, a.engine_budget), ("off", 30.0))
        self.assertEqual(self.parse("--engines", "glm", env=env).engines, "glm")

    def test_invalid(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.parse("--engines", "bogus")


if __name__ == "__main__":
    unittest.main()
