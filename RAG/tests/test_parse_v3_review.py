import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf as fitz

import parser.parse_document as pd


def fake_line(*spans, size=10.0):
    """spans: (font, text). Chars are 5pt wide, no gaps."""
    x, out = 0.0, []
    for font, text in spans:
        chars = []
        for ch in text:
            chars.append({"c": ch, "bbox": (x, 0, x + 5, size)})
            x += 5
        out.append({"font": font, "size": size, "flags": 0, "text": text, "chars": chars})
    return {"bbox": (0, 0, x, size), "spans": out}


def make_pdf(path, pages):
    doc = fitz.open()
    for draw in pages:
        draw(doc.new_page())
    doc.save(str(path))
    doc.close()


def text_page(lines, y=60):
    def f(page):
        for i, t in enumerate(lines):
            page.insert_text((50, y + 16 * i), t, fontsize=11, fontname="helv")
    return f


class SeedTest(unittest.TestCase):
    def seed(self, *spans):
        return pd._make_line(fake_line(*spans), 0, 0).seed

    def test_cmr_prose_not_seed(self):
        self.assertFalse(self.seed(("CMR10", "the model is trained")))
        self.assertFalse(self.seed(("CMR10", "200")))  # axis tick, no strong span

    def test_prose_with_one_math_span_stays_prose(self):
        self.assertFalse(self.seed(("CMR10", "the value of"), ("CMMI10", "x")))

    def test_formula_is_seed(self):
        self.assertTrue(self.seed(("CMMI10", "x̄"), ("CMR10", " = "), ("CMEX10", "Σ"),
                                  ("CMMI10", " x"), ("CMSY10", "/"), ("CMMI10", "n")))

    def test_size_zero_no_spaces(self):
        ln = pd._make_line(fake_line(("Helvetica", "abc"), size=0.0), 0, 0)
        self.assertEqual(ln.text, "abc")


class JoinTest(unittest.TestCase):
    def test_hyphens(self):
        self.assertEqual(pd._join_lines(["a trans-", "former"]), "a transformer")
        self.assertEqual(pd._join_lines(["the BERT-", "base model"]), "the BERT-base model")
        self.assertEqual(pd._join_lines(["state-of-the-", "art"]), "state-of-the-art")
        self.assertEqual(pd._join_lines(["a trans-", "Former"]), "a trans- Former")


class PageTest(unittest.TestCase):
    def blocks(self, pages):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.pdf"
            make_pdf(p, pages)
            out, stats = pd.parse_pdf(p, None, Path(tmp) / "assets")
        return out, stats

    def test_filled_bands_not_figure(self):
        body = [f"Line number {i} of ordinary body text on the page." for i in range(10)]

        def page(pg):
            for k in range(5):
                pg.draw_rect(fitz.Rect(40, 55 + 33 * k, 550, 55 + 33 * k + 30), color=None, fill=(.9, .9, .9))
            text_page(body)(pg)
        out, _ = self.blocks([page])
        self.assertFalse([b for b in out if b.type == "figure"])
        self.assertIn("Line number 9", " ".join(b.text for b in out))

    def test_vector_figure_detected(self):
        def page(pg):
            for k in range(8):
                pg.draw_rect(fitz.Rect(100 + 20 * k, 200, 120 + 20 * k, 200 + 15 * (k + 1)), color=(0, 0, 0))
            pg.draw_line((90, 360), (400, 360), color=(0, 0, 0))
            pg.insert_text((100, 380), "0", fontsize=8)
            text_page(["Some paragraph text that sits above the chart in the page."])(pg)
        out, _ = self.blocks([page])
        self.assertEqual(len([b for b in out if b.type == "figure"]), 1)

    def test_bold_only_heading_level(self):
        def page(pg):
            pg.insert_text((50, 60), "Short Title", fontsize=11, fontname="hebo")
            text_page([f"Body sentence number {i} that is long enough to count." for i in range(8)], y=100)(pg)
        out, _ = self.blocks([page])
        h = next(b for b in out if b.type == "heading")
        self.assertEqual(h.level, 3)

    def test_page_exception_fallback(self):
        real = pd.extract_pdf_page

        def fake(page, num, *a, **k):
            if num == 2:
                raise RuntimeError("boom")
            return real(page, num, *a, **k)
        pages = [text_page([f"Page {i} sentence one.", f"Page {i} sentence two."]) for i in (1, 2, 3)]
        with mock.patch.object(pd, "extract_pdf_page", fake):
            out, stats = self.blocks(pages)
        self.assertEqual(stats["page_errors"], 1)
        self.assertEqual({b.location["page_num"] if hasattr(b, "location") else b.page_num for b in out}, {1, 2, 3})

    def test_cache_miss_when_asset_deleted(self):
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        png = pix.tobytes("png")

        def page(pg):
            pg.insert_image(fitz.Rect(100, 300, 400, 500), stream=png)
            text_page([f"Some body text line {i} for the page." for i in range(4)])(pg)
        class NoOcr:
            def run(self, *_):
                return []
        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / "d.pdf", Path(tmp) / "out"
            make_pdf(src, [page])
            first = pd.execute(src, dest, ocr=NoOcr())
            assets = [b["asset"] for b in first["blocks"] if b.get("asset")]
            self.assertTrue(assets)
            for a in assets:
                (dest / a).unlink()
            with mock.patch.object(pd, "parse_pdf", wraps=pd.parse_pdf) as spy:
                pd.execute(src, dest, ocr=NoOcr())
            self.assertEqual(spy.call_count, 1)
            self.assertTrue(all((dest / a).exists() for a in assets))
            json.loads((dest / "d.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
