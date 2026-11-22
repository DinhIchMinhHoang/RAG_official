import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser.chunker import ChunkingError, chunk_document  # noqa: E402

CT = lambda s: len(s.split())
_n = [0]


def blk(type_, text, asset=None, level=0):
    _n[0] += 1
    b = {"id": f"e{_n[0]:04d}", "type": type_, "text": text, "level": level, "location": {"page_num": 1},
         "bbox": [0, 0, 1, 1], "breadcrumb": "", "metadata": {}}
    if asset:
        b["asset"] = asset
    return b


def chunk(blocks, **kw):
    return chunk_document(blocks, "doc", CT, min_tokens=5, **kw)


def fig():
    return blk("figure", "", asset="assets/e.png")


class EmptyChildTest(unittest.TestCase):
    def test_figure_between_paragraphs(self):
        f = fig()
        _, c = chunk([blk("paragraph", "một hai ba bốn năm sáu"), f, blk("paragraph", "bảy tám chín mười mười một")])
        self.assertTrue(all(x["text"].strip() for x in c))
        self.assertEqual([a["block_id"] for x in c for a in x["assets"]], [f["id"]])

    def test_section_only_empty_figure(self):
        f = fig()
        blocks = [blk("heading", "Mục A", level=1), f,
                  blk("heading", "Mục B", level=1), blk("paragraph", "nội dung mục B có nhiều chữ hơn năm")]
        p, c = chunk(blocks)
        self.assertEqual(len(p), 2)
        self.assertEqual(len(c), 1)
        self.assertNotIn(f["id"], c[0]["block_ids"])
        self.assertEqual(c[0]["text"], "nội dung mục B có nhiều chữ hơn năm")

    def test_ids_sequential(self):
        bl = []
        for i in range(3):
            bl += [blk("heading", f"Mục {i}", level=1), fig() if i == 1 else blk("paragraph", "a b c d e f g")]
        _, c = chunk(bl)
        self.assertEqual([x["child_id"] for x in c], [f"doc-c{i:04d}" for i in range(len(c))])

    def test_only_figures_raises(self):
        with self.assertRaises(ChunkingError):
            chunk([fig(), fig()])


if __name__ == "__main__":
    unittest.main()
