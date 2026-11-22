"""parser v3.1 engine integration: stub router (no models) + fake worker for the latex delimiter strip."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parser.parse_document as pd  # noqa: E402
from parser import engine_router as er  # noqa: E402


def make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Introduction paragraph about tables and results.", fontsize=11, fontname="helv")
    x0, y0, cw, ch = 50, 100, 100, 30
    cells = [["Name", "Score", "Rank"], ["alpha", "1", ""], ["beta", "", ""]]
    for r, row in enumerate(cells):
        for c, v in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch)
            page.draw_rect(rect, width=1)
            if v:
                page.insert_text((rect.x0 + 5, rect.y0 + 20), v, fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()


class Stub:
    def __init__(self, mode="auto", new=None, boom=False):
        self.mode, self.new, self.boom, self.calls = mode, new, boom, []

    def enhance(self, blocks, pdf_path, asset_dir, *, elapsed_s=0.0, n_pages=0):
        if self.boom:
            raise RuntimeError("kaboom")
        self.calls.append({"ids": [b.id for b in blocks], "asset_dir": asset_dir, "elapsed_s": elapsed_s,
                           "n_pages": n_pages})
        new = [dict(n, after_id=blocks[0].id) for n in (self.new or [])]
        return {"mode": self.mode, "engines": {"glm": {"state": "ok"}}, "new_blocks": new, "signature": "stub=1"}


NEW = [{"id": "x", "type": "table", "text": "| a | b |", "location": {"page_num": 1}, "metadata": {"engine": "stub"}}]


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        self.pdf = self.d / "t.pdf"
        make_pdf(self.pdf)

    def run_exec(self, dest="out", **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return pd.execute(self.pdf, self.d / dest, ocr=object(), **kw)


class ParseEnginesTest(Base):
    def test_version_and_table_metadata(self):
        p = self.run_exec()
        self.assertEqual(p["parser_version"], "3.1.0")
        t = [b for b in p["blocks"] if b["type"] == "table"]
        self.assertTrue(t)
        m = t[0]["metadata"]
        self.assertEqual(m["page"], 1)
        self.assertGreater(m["none_ratio"], 0.0)
        self.assertAlmostEqual(m["none_ratio"] + m["fill"], 1.0, places=3)

    def test_off_equals_none(self):
        a = self.run_exec("a")
        b = self.run_exec("b", router=Stub(mode="off"))
        self.assertEqual(json.dumps(a["blocks"], sort_keys=True), json.dumps(b["blocks"], sort_keys=True))
        self.assertEqual((a["stats"]["engines_mode"], b["stats"]["engines_mode"]), ("off", "off"))
        self.assertNotIn("engines", a["stats"])

    def test_enhance_args_and_splice(self):
        s = Stub(new=NEW + NEW)
        p = self.run_exec(router=s)
        base = self.run_exec("base")
        c = s.calls[0]
        self.assertTrue(all(i for i in c["ids"]))  # ids already assigned
        self.assertEqual(c["n_pages"], 1)
        self.assertGreaterEqual(c["elapsed_s"], 0.0)
        self.assertEqual(Path(c["asset_dir"]), self.d / "out")
        first = p["blocks"][0]["id"]
        self.assertEqual([b["id"] for b in p["blocks"][1:3]], [f"{first}_e1", f"{first}_e2"])
        self.assertEqual(p["blocks"][1]["location"], {"page_num": 1})
        self.assertEqual(len(p["blocks"]), len(base["blocks"]) + 2)
        self.assertEqual(len({b["id"] for b in p["blocks"]}), len(p["blocks"]))
        st = p["stats"]
        self.assertEqual((st["engines_sig"], st["engines_mode"]), ("stub=1", "auto"))
        self.assertNotIn("new_blocks", st["engines"])
        self.assertEqual(st["blocks"], len(p["blocks"]))
        self.assertIn(NEW[0]["text"], (self.d / "out" / "t.md").read_text(encoding="utf-8"))

    def test_cache_invalidated_by_mode(self):
        s1, s2, s3 = Stub("auto"), Stub("glm"), Stub("glm")
        self.run_exec(router=s1)
        self.run_exec(router=s2)
        self.assertEqual((len(s1.calls), len(s2.calls)), (1, 1))  # mode changed -> re-parse
        self.run_exec(router=s3)
        self.assertEqual(len(s3.calls), 0)  # same mode -> cache hit
        s4 = Stub("auto")
        self.run_exec(router=s4)
        self.assertEqual(len(s4.calls), 1)

    def test_router_exception_keeps_blocks(self):
        base = self.run_exec("base")
        p = self.run_exec(router=Stub(boom=True))
        self.assertEqual(p["stats"]["engines"]["state"], "failed")
        self.assertIn("kaboom", p["stats"]["engines"]["reason"])
        self.assertEqual(p["blocks"], base["blocks"])


class LatexStripTest(unittest.TestCase):
    def test_strip(self):
        s = er.strip_math_delims
        self.assertEqual(s("$$\n\\frac{a}{b}\n$$"), "\\frac{a}{b}")
        self.assertEqual(s("  \\[ x^2 \\]\n"), "x^2")
        self.assertEqual(s("x$$"), "x$$")
        self.assertEqual(s("$$"), "$$")

    def test_accept_path_stores_stripped_latex(self):
        fake = Path(er.ENGINES_DIR) / "fake" / "worker.py"
        r = er.EngineRouter(mode="glm", python_paths={"glm": (sys.executable, fake)})
        b = {"id": "f0", "type": "formula", "text": "frac ab", "metadata": {}}
        st = {"formula": {"done": 0, "cached": 0, "rejected": 0, "skipped": 0}}
        r._apply_glm("formula", b, {"latex": "$$\n\\frac{a}{b}\n$$"}, st, "done", {"new_blocks": []})
        self.assertEqual(b["metadata"]["latex"], "\\frac{a}{b}")
        self.assertEqual(st["formula"]["done"], 1)


if __name__ == "__main__":
    unittest.main()
