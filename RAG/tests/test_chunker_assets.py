import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser.chunker import chunk_document  # noqa: E402

CT = lambda s: len(s.split())
_n = [0]


def blk(type_, text, asset=None, lang=None):
    _n[0] += 1
    b = {"id": f"a{_n[0]:04d}", "type": type_, "text": text, "level": 0, "location": {"page_num": 1},
         "bbox": [0, 0, 1, 1], "breadcrumb": "", "metadata": {"language": lang} if lang else {}}
    if asset:
        b["asset"] = asset
    return b


def chunk(blocks, **kw):
    return chunk_document(blocks, "doc", CT, min_tokens=5, **kw)


class AssetChunkTest(unittest.TestCase):
    def test_code_indent_and_language(self):
        code = "    x = 1\n    if x:\n        y = 2\n"
        _, c = chunk([blk("paragraph", "Ví dụ mã nguồn"), blk("code", code, lang="python")])
        self.assertIn("    x = 1\n    if x:\n        y = 2", c[0]["text"])
        self.assertEqual(c[0]["languages"], ["python"])

    def test_figure_with_text(self):
        _, c = chunk([blk("paragraph", "Mô tả"), blk("figure", "Hình 1 sơ đồ", asset="assets/b1.png")])
        self.assertEqual(c[0]["assets"][0]["asset"], "assets/b1.png")
        self.assertIn(c[0]["content_type"], ("figure", "mixed"))

    def test_figure_empty_text(self):
        f = blk("figure", "", asset="assets/e.png")
        _, c = chunk([blk("paragraph", "Dòng một"), f, blk("paragraph", "Dòng hai")])
        self.assertEqual([a["block_id"] for a in c[0]["assets"]], [f["id"]])
        self.assertIn(f["id"], c[0]["block_ids"])
        self.assertNotIn("\n\n", c[0]["text"])
        self.assertFalse(c[0]["text"].startswith("\n") or c[0]["text"].endswith("\n"))

    def test_label_and_formula_same_child(self):
        f = blk("formula", "E = mc²", asset="assets/f.png")
        _, c = chunk([blk("paragraph", "Công thức năng lượng:"), f])
        self.assertEqual(len(c), 1)
        self.assertIn("Công thức năng lượng:", c[0]["text"])
        self.assertEqual(c[0]["assets"][0]["block_id"], f["id"])
        self.assertEqual(c[0]["assets"][0]["type"], "formula")

    def test_oversize_code_splits_keep_indent(self):
        code = "\n".join("    " * (i % 3) + f"line{i} a b c" for i in range(20))
        _, c = chunk([blk("code", code, lang="python")], ceiling=20)
        self.assertGreater(len(c), 1)
        got = "\n".join(x["text"] for x in c).split("\n")
        self.assertEqual(got, code.split("\n"))
        self.assertTrue(all(x["languages"] == ["python"] for x in c))

    def test_no_assets(self):
        _, c = chunk([blk("paragraph", "Chỉ văn bản thường")])
        self.assertEqual((c[0]["assets"], c[0]["languages"]), ([], []))


if __name__ == "__main__":
    unittest.main()
