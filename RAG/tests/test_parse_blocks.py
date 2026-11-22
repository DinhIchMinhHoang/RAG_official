import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

from parser.parse_document import clean_text, extract_pdf_page, parse_markdown_blocks


def page_blocks(draw, tmp):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    draw(page)
    return extract_pdf_page(page, 1, Path(tmp) / "assets")


class ParseBlocksTest(unittest.TestCase):
    def test_glue_gap_becomes_space(self):
        def draw(p):
            # base-14 Helvetica has no Vietnamese glyphs, so ASCII stands in for "phổ biến"
            w = fitz.get_text_length("pho", fontname="helv", fontsize=11)
            p.insert_text((72, 100), "pho", fontname="helv", fontsize=11)
            p.insert_text((72 + w + 0.35 * 11, 100), "bien", fontname="helv", fontsize=11)
        with tempfile.TemporaryDirectory() as tmp:
            texts = [b.text for b in page_blocks(draw, tmp)]
        self.assertIn("pho bien", " ".join(texts))

    def test_raw_order_columns_and_label(self):
        def draw(p):
            p.insert_text((72, 100), "left one", fontsize=11)
            p.insert_text((72, 300), "left two", fontsize=11)
            p.insert_text((330, 100), "right one", fontsize=11)
            p.insert_text((72, 500), "label:", fontsize=11)
            p.insert_text((72, 520), "value", fontsize=11)
        with tempfile.TemporaryDirectory() as tmp:
            joined = " ".join(b.text for b in page_blocks(draw, tmp))
        self.assertLess(joined.index("left two"), joined.index("right one"))
        self.assertLess(joined.index("label:"), joined.index("value"))

    def test_code_keeps_indent_and_small_mono_is_not_code(self):
        def draw(p):
            p.insert_text((72, 100), "def f(x):", fontname="cour", fontsize=10)
            p.insert_text((72, 112), "    if x:", fontname="cour", fontsize=10)
            p.insert_text((72, 124), "        return 1", fontname="cour", fontsize=10)
            p.insert_text((72, 700), "logo = (tiny)", fontname="cour", fontsize=6)
        with tempfile.TemporaryDirectory() as tmp:
            blocks = page_blocks(draw, tmp)
        code = [b for b in blocks if b.type == "code"]
        self.assertEqual(len(code), 1)
        self.assertEqual(code[0].text.splitlines(), ["def f(x):", "    if x:", "        return 1"])
        self.assertFalse(any("logo" in b.text for b in code))

    def test_footer_counter_dropped(self):
        def draw(p):
            p.insert_text((72, 100), "Hello world body text", fontsize=11)
            p.insert_text((280, 820), "51 / 106", fontsize=9)
        with tempfile.TemporaryDirectory() as tmp:
            blocks = page_blocks(draw, tmp)
        self.assertFalse(any("51" in b.text for b in blocks))
        self.assertTrue(any("Hello" in b.text for b in blocks))

    def test_image_becomes_figure_with_png_asset(self):
        def draw(p):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
            pix.set_rect(pix.irect, (200, 30, 30))
            p.insert_image(fitz.Rect(100, 200, 200, 300), pixmap=pix)
            p.insert_text((72, 100), "Some text", fontsize=11)
        with tempfile.TemporaryDirectory() as tmp:
            blocks = page_blocks(draw, tmp)
            figs = [b for b in blocks if b.type == "figure"]
            self.assertEqual(len(figs), 1)
            self.assertEqual(figs[0].metadata["kind"], "raster")
            png = Path(tmp) / "assets" / Path(figs[0].asset).name
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))

    def test_clean_text(self):
        self.assertEqual(clean_text("a<mark>b</mark><br>c d"), "abc d")
        self.assertEqual(clean_text("x<!-- x -->y"), "xy")
        self.assertEqual(clean_text("¯x"), "x̄")

    def test_markdown_fence_and_math(self):
        blocks = parse_markdown_blocks("intro\n\n```python\n# comment\n  x = 1\n```\n\nafter")
        self.assertEqual([b.type for b in blocks].count("code"), 1)
        self.assertEqual([b.type for b in blocks].count("heading"), 0)
        code = next(b for b in blocks if b.type == "code")
        self.assertEqual(code.metadata["language"], "python")
        self.assertEqual(code.text, "# comment\n  x = 1")
        f = parse_markdown_blocks("$$x^2$$")
        self.assertEqual((f[0].type, f[0].metadata["format"], f[0].text), ("formula", "latex", "x^2"))


if __name__ == "__main__":
    unittest.main()
