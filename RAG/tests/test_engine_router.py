"""engine_router tests: fake worker only (stdlib), no real models."""
import contextlib
import copy
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import engine_router as er  # noqa: E402

FAKE = Path(er.ENGINES_DIR) / "fake" / "worker.py"
PY = {"glm": (sys.executable, FAKE), "docling": (sys.executable, FAKE)}


def make_png(path, shade, w=64, h=64):
    raw = b"".join(b"\x00" + bytes([shade]) * w for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.pdf = self.d / "x.pdf"
        self.pdf.write_bytes(b"%PDF fake")
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("FAKE_")}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def formulas(self, n, text="frac ab"):
        out = []
        for i in range(n):
            make_png(self.d / f"f{i}.png", 10 + i)
            out.append({"id": f"f{i}", "type": "formula", "text": text, "location": {"page_num": i + 1},
                        "asset": str(self.d / f"f{i}.png"), "metadata": {"format": "text"}, "bbox": None})
        return out

    def router(self, mode="glm", **kw):
        return er.EngineRouter(mode=mode, python_paths=PY, **kw)

    def run_router(self, blocks, mode="glm", ema=None, **kw):
        r = self.router(mode, **kw.pop("rkw", {}))
        if ema:
            r.ema.update(ema)
        with r:
            return r.enhance(blocks, self.pdf, self.d, **kw)


class Pure(unittest.TestCase):
    def test_plausible_accept(self):
        self.assertEqual(er.plausible_latex(r"\frac{a}{b}", "frac ab"), (True, ""))
        self.assertTrue(er.plausible_latex(r"\left( x \right)", "")[0])
        self.assertTrue(er.plausible_latex(r"x", "∂")[0])  # unicode has < 4 alnum: no overlap check

    def test_plausible_reject(self):
        for bad in (r"\frac{a}{b", r"a}", r"\left( x", "", "x" * 601, "a <|im_end|>", "a\x00b",
                    "x " * 10, r"\alpha_1 + " * 5):
            self.assertFalse(er.plausible_latex(bad, "")[0], bad)
        ok, why = er.plausible_latex(r"\frac{a}{b}", "zzzz qqqq")
        self.assertFalse(ok)
        self.assertIn("unicode", why)

    def test_suspect_table(self):
        self.assertTrue(er.suspect_table({"none_ratio": 0.04, "fill": 1.0}))
        self.assertTrue(er.suspect_table({"none_ratio": 0.0, "fill": 0.79}))
        self.assertFalse(er.suspect_table({"none_ratio": 0.03, "fill": 0.9}))
        self.assertFalse(er.suspect_table({}))

    def test_group_pages(self):
        self.assertEqual(er.group_pages([5, 3, 4, 9, 3, 11, 10]), [(3, 5), (9, 11)])
        self.assertEqual(er.group_pages([]), [])

    def test_iou(self):
        self.assertAlmostEqual(er.iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(er.iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertAlmostEqual(er.iou([0, 0, 2, 1], [1, 0, 3, 1]), 1 / 3)


class Router(Base):
    def test_off_unchanged(self):
        blocks = self.formulas(4)
        before = copy.deepcopy(blocks)
        st = self.run_router(blocks, mode="off")
        self.assertEqual(blocks, before)
        self.assertEqual(st["engines"]["glm"]["state"], "off")
        self.assertEqual(st["signature"], "")

    def test_formula_accept_dict_and_dataclass(self):
        blocks = self.formulas(3)
        st = self.run_router(blocks)
        self.assertEqual(st["engines"]["glm"]["formula"]["done"], 3)
        self.assertEqual(blocks[0]["metadata"]["latex"], r"\frac{a}{b}")
        self.assertEqual(blocks[0]["metadata"]["engine"], "glm")
        self.assertEqual(blocks[0]["text"], "frac ab")
        self.assertIn("glm=", st["signature"])
        # dataclass + tiny unicode -> text replaced by latex
        from parser.parse_document import ContentBlock
        cbs = [ContentBlock(id=b["id"], type="formula", text="∂", location=b["location"], asset=b["asset"]) for b in self.formulas(3)]
        self.run_router(cbs)
        self.assertEqual(cbs[0].text, r"\frac{a}{b}")

    def test_formula_reject_keeps_unicode(self):
        blocks = self.formulas(3, text="zzzz qqqq")
        st = self.run_router(blocks)
        self.assertEqual(st["engines"]["glm"]["formula"]["rejected"], 3)
        self.assertEqual(blocks[0]["metadata"]["latex_flags"], ["rejected:unicode mismatch"])
        self.assertNotIn("latex", blocks[0]["metadata"])
        self.assertEqual(blocks[0]["text"], "zzzz qqqq")

    def test_budget_skip(self):
        log = self.d / "log"
        os.environ["FAKE_LOG"] = str(log)
        st = self.run_router(self.formulas(4), elapsed_s=100.0)
        self.assertLess(st["effective_s"], 15)
        self.assertEqual(st["engines"]["glm"]["state"], "skipped")
        self.assertFalse(log.exists())

    def test_min_candidates(self):
        log = self.d / "log"
        os.environ["FAKE_LOG"] = str(log)
        st = self.run_router(self.formulas(2))
        self.assertEqual(st["engines"]["glm"]["state"], "skipped")
        self.assertFalse(log.exists())

    def test_timeout_kills_and_continues(self):
        os.environ["FAKE_SLEEP"] = "3"
        procs, orig = [], subprocess.Popen

        def spy(*a, **k):
            procs.append(orig(*a, **k))
            return procs[-1]
        blocks = self.formulas(3)
        with mock.patch.object(er, "MIN_REQ_TIMEOUT_S", 0.5), mock.patch.object(er.subprocess, "Popen", spy):
            st = self.run_router(blocks, ema={"glm_formula": 0.05})
        g = st["engines"]["glm"]
        self.assertEqual(g["state"], "failed")
        self.assertEqual(g["formula"]["skipped"], 3)
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertNotIn("latex", blocks[0]["metadata"])

    def test_crash_keeps_earlier_results(self):
        os.environ["FAKE_CRASH_AT"] = "3"
        blocks = self.formulas(4)
        st = self.run_router(blocks)
        g = st["engines"]["glm"]
        self.assertEqual(g["state"], "failed")
        self.assertEqual(g["formula"]["done"], 2)
        self.assertEqual(g["formula"]["skipped"], 2)
        self.assertIn("latex", blocks[1]["metadata"])
        self.assertNotIn("latex", blocks[2]["metadata"])

    def test_bad_json(self):
        os.environ["FAKE_BAD_JSON"] = "2"
        blocks = self.formulas(3)
        st = self.run_router(blocks)
        g = st["engines"]["glm"]
        self.assertEqual(g["state"], "failed")
        self.assertEqual(g["formula"]["done"], 1)

    def test_missing_python_skipped_with_warning(self):
        r = er.EngineRouter(mode="glm", python_paths={"glm": str(self.d / "nope" / "python.exe")})
        err = io.StringIO()
        with contextlib.redirect_stderr(err), r:
            st = r.enhance(self.formulas(3), self.pdf, self.d)
        g = st["engines"]["glm"]
        self.assertEqual(g["state"], "skipped")
        self.assertIn("python not found", g["reason"])
        self.assertIn("[engine glm] skipped: python not found (set RAG_ENGINE_GLM_PYTHON)", err.getvalue())

    def test_cache_hit_starts_no_worker(self):
        cache = self.d / "cache"
        blocks = self.formulas(3)
        self.run_router(blocks, rkw={"cache_dir": cache})
        self.assertTrue((cache / "engine_stats.json").exists())
        log = self.d / "log"
        os.environ["FAKE_LOG"] = str(log)
        blocks2 = self.formulas(3)
        st = self.run_router(blocks2, rkw={"cache_dir": cache})
        self.assertFalse(log.exists())
        self.assertEqual(st["engines"]["glm"]["formula"]["cached"], 3)
        self.assertEqual(blocks2[2]["metadata"]["latex"], r"\frac{a}{b}")

    def test_close_leaves_no_process(self):
        procs, orig = [], subprocess.Popen

        def spy(*a, **k):
            procs.append(orig(*a, **k))
            return procs[-1]
        with mock.patch.object(er.subprocess, "Popen", spy):
            self.run_router(self.formulas(3))
        self.assertTrue(procs)
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertFalse(er._LIVE)

    def table_block(self, text):
        return {"id": "t1", "type": "table", "text": text, "location": {"page_num": 1}, "bbox": [0.0, 0.0, 1.0, 1.0],
                "metadata": {"page": 1, "none_ratio": 0.2, "fill": 0.6, "rows": [["a", "b"], ["1", "2"]]}}

    def test_docling_replaces_when_words_subset(self):
        b = self.table_block("| a | b |\n| 1 | 2 |")
        st = self.run_router([b], mode="docling")
        self.assertEqual(st["engines"]["docling"]["tables_replaced"], 1)
        self.assertEqual(b["text"], "| a | b |\n| --- | --- |\n| 1 | 2 |")
        self.assertEqual(b["metadata"]["engine"], "docling")
        self.assertEqual(b["metadata"]["parser_rows"], [["a", "b"], ["1", "2"]])

    def test_docling_guard_blocks_invented_text(self):
        b = self.table_block("| x | y |\n| 3 | 4 |")
        before = copy.deepcopy(b)
        st = self.run_router([b], mode="docling")
        self.assertEqual(st["engines"]["docling"]["tables_replaced"], 0)
        self.assertEqual(b, before)

    def test_image_table_new_block(self):
        make_png(self.d / "fig.png", 99)
        fig = {"id": "fig1", "type": "figure", "text": "", "location": {"page_num": 2}, "asset": str(self.d / "fig.png"),
               "metadata": {"kind": "raster", "ocr_text": "a 1 2\nb 3 4\nc 5 6\nd 7 8"}}
        before = copy.deepcopy(fig)
        st = self.run_router([fig])
        self.assertEqual(fig, before)
        self.assertEqual(len(st["new_blocks"]), 1)
        nb = st["new_blocks"][0]
        self.assertEqual(nb["after_id"], "fig1")
        self.assertEqual(nb["type"], "table")
        self.assertEqual(nb["metadata"]["engine"], "glm")
        self.assertEqual(nb["location"], {"page_num": 2})


if __name__ == "__main__":
    unittest.main()
