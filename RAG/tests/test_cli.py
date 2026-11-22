"""Run from RAG/: python -m unittest tests.test_cli (fake Collection + fake LLM, no models, no network)."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli  # noqa: E402
import ingest as ing  # noqa: E402
from llm.answering import REFUSAL_TEXT  # noqa: E402
from llm.client import UsageMeter  # noqa: E402
from parser.chunker import ChunkingError  # noqa: E402

CT = lambda s: len(s.split())


class FakeCol:
    def __init__(self, cos=0.8):
        self.cos, self.closed = cos, False

    def search(self, q, **kw):
        par = [{"parent_id": "p1", "doc_id": "d1", "title": "T", "breadcrumb": "d1 > T", "text": "Nội dung một " * 30,
                "page_start": 1, "page_end": 2, "score": 0.9, "matched_child_ids": ["c1"]}]
        return {"children": [], "parents": par, "timings_ms": {"embed": 10, "total": 30, "rerank": 0},
                "top1": {"dense_cosine": self.cos, "fused": 0.1, "margin": 0.0}}

    def close(self):
        self.closed = True


class FakeLLM:
    def __init__(self, text="Đây là đáp án [S1] và [S9]."):
        self.text, self.called, self.messages = text, 0, None
        self.config = cli.LLMConfig()
        self.meter = UsageMeter()

    def stream_chat(self, messages, **kw):
        self.called += 1
        self.messages = messages
        for w in self.text.split(" "):
            yield {"type": "token", "text": w + " "}
        yield {"type": "final", "text": self.text, "truncated": True, "t_first_token_s": 0.1,
               "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.001}}


def args(*argv):
    return cli.build_parser().parse_args(list(argv))


class ParseTest(unittest.TestCase):
    def test_ingest(self):
        a = args("ingest", "a.pdf", "dir", "--store", "S", "--device", "cpu", "--ceiling", "300", "--force",
                 "--stack", "--tessdata", "T")
        self.assertEqual((len(a.paths), a.ceiling, a.device, a.force, a.stack), (2, 300, "cpu", True, True))
        self.assertEqual(args("ingest", "a", "--store", "S").ceiling, 400)

    def test_ask_defaults_and_flags(self):
        a = args("ask", "hỏi gì", "--store", "S")
        self.assertEqual((a.top, a.max_parents, a.fusion, a.alpha, a.budget_tokens, a.min_cosine),
                         (5, 3, "rrf", 0.7, 3500, None))
        a = args("ask", "q", "--store", "S", "--fusion", "convex", "--rerank", "--no-llm", "--min-cosine", "0.4",
                 "--json")
        self.assertTrue(a.rerank and a.no_llm and a.json)
        self.assertEqual(a.min_cosine, 0.4)

    def test_chat_info(self):
        self.assertEqual(args("chat", "--store", "S", "--top", "2").top, 2)
        self.assertEqual(args("info", "--store", "S").cmd, "info")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            args("ask", "q")  # --store required


class AskTest(unittest.TestCase):
    def ask(self, col, llm, *argv, history=None):
        out = io.StringIO()
        r = cli.ask_once("câu hỏi", col, llm, args("ask", "câu hỏi", "--store", "S", *argv), CT, history, out)
        return r, out.getvalue()

    def test_stream_citations_flags(self):
        r, text = self.ask(FakeCol(), FakeLLM())
        self.assertIn("Đây là đáp án [S1] và [S9].", text)
        self.assertIn("Nguồn:", text)
        self.assertIn("[S1] T (doc d1) tr.1-2 | d1 > T", text)
        self.assertEqual([c["source_id"] for c in r["citations"]], ["S1"])
        self.assertEqual(r["invalid_citations"], ["S9"])
        self.assertTrue(r["truncated"])
        self.assertIn("tokens 5+3", text)
        self.assertEqual(r["timings_ms"]["ttft"], 100.0)

    def test_refused(self):
        r, _ = self.ask(FakeCol(), FakeLLM(REFUSAL_TEXT))
        self.assertTrue(r["refused"])
        self.assertFalse(r["gated"])

    def test_min_cosine_gate_skips_llm(self):
        llm = FakeLLM()
        r, text = self.ask(FakeCol(cos=0.2), llm, "--min-cosine", "0.5")
        self.assertEqual(llm.called, 0)
        self.assertTrue(r["gated"] and r["refused"])
        self.assertIn(REFUSAL_TEXT, text)
        llm2 = FakeLLM()
        self.ask(FakeCol(cos=0.9), llm2, "--min-cosine", "0.5")
        self.assertEqual(llm2.called, 1)

    def test_no_llm_prints_passages(self):
        llm = FakeLLM()
        r, text = self.ask(FakeCol(), llm, "--no-llm")
        self.assertEqual(llm.called, 0)
        self.assertIn("#1 score=0.9000 doc=d1", text)
        self.assertEqual(len(r["passages"][0]["text"]), 200)

    def test_json_shape(self):
        out = io.StringIO()
        a = args("ask", "q", "--store", "S", "--json")
        with mock.patch.object(cli, "open_collection", return_value=FakeCol()), \
                mock.patch.object(cli, "LLMClient", return_value=FakeLLM()), \
                mock.patch.object(cli, "get_counter", return_value=CT), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            self.assertEqual(cli.cmd_ask(a, out=out), 0)
        d = json.loads(out.getvalue())  # stdout is pure JSON (no streamed tokens)
        for k in ("question", "answer", "refused", "citations", "invalid_citations", "truncated", "sources",
                  "timings_ms", "usage"):
            self.assertIn(k, d)
        self.assertEqual(d["sources"][0]["source_id"], "S1")

    def test_missing_api_key_names_variable(self):
        err = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "RAG_LLM_API_KEY_ENV")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(cli.cmd_ask(args("ask", "q", "--store", "S"), out=io.StringIO(), err=err), 2)
        self.assertIn("OPENROUTER_API_KEY", err.getvalue())
        self.assertIn("--no-llm", err.getvalue())

    def test_store_missing_suggests_ingest(self):
        err = io.StringIO()
        a = args("ask", "q", "--store", str(Path(tempfile.gettempdir()) / "no_such_store_xyz"), "--no-llm")
        self.assertEqual(cli.cmd_ask(a, out=io.StringIO(), err=err), 2)
        self.assertIn("ingest", err.getvalue())


class ChatTest(unittest.TestCase):
    def test_history_trimmed_and_commands(self):
        llm, col, out = FakeLLM("trả lời [S1]"), FakeCol(), io.StringIO()
        inputs = iter([f"câu {i}" for i in range(5)] + ["/sources", "/stats", "/quit"])
        a = args("chat", "--store", "S")
        with mock.patch.object(cli, "open_collection", return_value=col), \
                mock.patch.object(cli, "LLMClient", return_value=llm), \
                mock.patch.object(cli, "get_counter", return_value=CT), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            self.assertEqual(cli.cmd_chat(a, out=out, input_fn=lambda _: next(inputs)), 0)
        user = llm.messages[1]["content"]
        hist = user.split("LỊCH SỬ HỘI THOẠI")[1].split("CÂU HỎI")[0]
        self.assertEqual(hist.count("user:"), 3)  # 6 messages kept = last 3 exchanges
        self.assertNotIn("câu 0", hist)
        self.assertIn("câu 3", hist)
        self.assertIn("[S1] T", out.getvalue())
        self.assertIn("lượt hỏi=5", out.getvalue())
        self.assertTrue(col.closed)


class RecCol(FakeCol):
    """Records search kwargs; returns parents of two docs so the --doc post-filter has work to do."""
    def search(self, q, **kw):
        self.kw = kw
        mk = lambda pid, d, s: {"parent_id": pid, "doc_id": d, "title": f"T-{d}", "breadcrumb": "", "text": "x",
                                "page_start": None, "page_end": None, "score": s, "matched_child_ids": [pid + "c"]}
        return {"children": [{"child_id": "p1c", "parent_id": "p1", "score": 0.9, "scores": {"dense": 0.9}},
                             {"child_id": "p2c", "parent_id": "p2", "score": 0.5, "scores": {"dense": 0.5}}],
                "parents": [mk("p1", "da", 0.9), mk("p2", "db", 0.5)],
                "timings_ms": {"embed": 10, "total": 30, "rerank": 0},
                "top1": {"dense_cosine": 0.9, "fused": 0.9, "margin": 0.4}}


class FeatureTest(unittest.TestCase):
    def run_ask(self, col, *argv, n_docs=1):
        a = args("ask", "q", "--store", "S", "--no-llm", *argv)
        a.n_docs = n_docs
        out = io.StringIO()
        return cli.ask_once("q", col, None, a, None, out=out), out.getvalue()

    def test_rerank_auto(self):
        self.assertFalse(cli.resolve_rerank("auto", 1))
        self.assertTrue(cli.resolve_rerank("auto", 2))
        self.assertTrue(cli.resolve_rerank("on", 1))
        self.assertFalse(cli.resolve_rerank("off", 5))
        self.assertEqual(args("ask", "q", "--store", "S").rerank, "auto")
        self.assertEqual(args("ask", "q", "--store", "S", "--rerank", "off").rerank, "off")
        col = RecCol()
        r, text = self.run_ask(col, n_docs=1)
        self.assertFalse(col.kw["rerank"])
        self.assertIn("rerank off (auto)", text)
        r, text = self.run_ask(col, n_docs=2)
        self.assertTrue(col.kw["rerank"])
        self.assertIn("rerank on (auto)", text)
        self.run_ask(col, "--rerank", "off", n_docs=2)
        self.assertFalse(col.kw["rerank"])

    def test_doc_filter_faiss_post_filters(self):
        col = RecCol()
        r, _ = self.run_ask(col, "--doc", "db", n_docs=3)
        self.assertEqual([p["doc_id"] for p in r["passages"]], ["db"])
        self.assertEqual(r["top1"]["dense_cosine"], 0.5)
        self.assertNotIn("filter_doc_ids", col.kw)
        self.assertFalse(col.kw["rerank"])  # one restricted doc -> auto off

    def test_doc_filter_qdrant_native(self):
        col = RecCol()
        self.run_ask(col, "--backend", "qdrant", "--doc", "db", "--doc", "da")
        self.assertEqual(col.kw["filter_doc_ids"], ["da", "db"])
        self.assertTrue(col.kw["rerank"])

    def test_backend_selection_and_unreachable(self):
        self.assertEqual(args("ask", "q", "--store", "S").backend, "faiss")
        self.assertEqual(args("info", "--store", "S", "--backend", "qdrant").backend, "qdrant")
        import embed.qdrant_collection as qc
        fake_client = object()
        with mock.patch.object(ing, "connect_qdrant", return_value=fake_client), \
                mock.patch.object(qc.QdrantCollection, "open", return_value="COL") as op:
            self.assertEqual(cli.open_collection("D:/x/My Store", "cpu", "qdrant"), "COL")
        op.assert_called_once_with("rag_my_store", client=fake_client, device="cpu")
        err = io.StringIO()
        boom = ConnectionError("Không kết nối được Qdrant ... docker compose up -d ... python -m store.doctor")
        with mock.patch.object(ing, "connect_qdrant", side_effect=boom):
            a = args("ask", "q", "--store", "S", "--no-llm", "--backend", "qdrant")
            self.assertEqual(cli.cmd_ask(a, out=io.StringIO(), err=err), 2)
            self.assertEqual(cli.cmd_ingest(args("ingest", __file__, "--store", "S", "--backend", "qdrant"),
                                            out=io.StringIO(), err=err), 2)
        self.assertIn("docker compose up -d", err.getvalue())
        self.assertIn("store.doctor", err.getvalue())

    def test_hint_condition(self):
        self.assertIsNone(cli.load_hint({"timings_ms": {"embed": 2999}}))
        self.assertIn("chat", cli.load_hint({"timings_ms": {"embed": 3500}}))
        out = io.StringIO()
        slow = lambda *a, **k: {"timings_ms": {"embed": 9000}}
        with mock.patch.object(cli, "open_collection", return_value=FakeCol()), \
                mock.patch.object(cli, "ask_once", slow):
            cli.cmd_ask(args("ask", "q", "--store", "S", "--no-llm"), out=out)
        self.assertIn("giữ mô hình nóng", out.getvalue())

    def test_page_formatting_and_title(self):
        self.assertEqual(cli.fmt_pages(None, None), "không có số trang")
        self.assertEqual(cli.fmt_pages(3, 3), "tr.3")
        self.assertEqual(cli.fmt_pages(3, 5), "tr.3-5")
        line = cli._src_line({"source_id": "S1", "title": "guide-ab12cd", "doc_id": "guide-ab12cd",
                              "page_start": None, "page_end": None, "breadcrumb": "x"})
        self.assertEqual(line, "[S1] guide (doc guide-ab12cd) không có số trang | x")
        _, text = self.run_ask(RecCol())
        self.assertIn("doc=da không có số trang", text)
        self.assertNotIn("None", text)


class IngestCmdTest(unittest.TestCase):
    def test_failure_isolation_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            good, bad = d / "good.md", d / "bad.md"
            good.write_text("tốt", encoding="utf-8")
            bad.write_text("BAD", encoding="utf-8")

            def parse(path, dest, force):
                if "BAD" in Path(path).read_text(encoding="utf-8"):
                    raise ChunkingError("boom")
                return {"blocks": [{"id": "b0", "type": "paragraph", "text": "tốt lắm", "level": 0,
                                    "location": {"page_num": 1}, "bbox": None, "metadata": {}}], "stats": {}}

            a = args("ingest", str(bad), str(good), "--store", str(d / "s"))
            out = io.StringIO()
            real = ing.ingest_paths
            with mock.patch.object(cli, "get_counter", return_value=CT), \
                    mock.patch.object(ing, "ingest_paths", lambda *p, **k: real(*p, **{**k, "parse_fn": parse})), \
                    mock.patch.object(ing, "rebuild_index", return_value=(True, 1.0)) as rb:
                self.assertEqual(cli.cmd_ingest(a, out=out), 2)
            self.assertTrue(rb.called)
            self.assertIn("THẤT BẠI", out.getvalue())
            self.assertEqual(len(ing._read_jsonl(d / "s" / "corpus" / "children.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
