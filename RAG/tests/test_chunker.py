import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser.chunker import ChunkingError, chunk_document, last_stats, load_blocks, write_jsonl  # noqa: E402

CT = lambda s: len(s.split())
_n = [0]


def blk(type_, text, level=0, page=1, bbox=None):
    _n[0] += 1
    return {"id": f"b{_n[0]:05d}", "type": type_, "text": text, "level": level,
            "location": {"page_num": page}, "bbox": bbox, "metadata": {}}


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


class ChunkerTest(unittest.TestCase):
    def run_chunk(self, blocks, **kw):
        return chunk_document(blocks, "doc", CT, **kw)

    def test_heading_tree_levels(self):
        bl = [blk("heading", "A", 1), blk("heading", "A1", 2), blk("paragraph", words(30)),
              blk("heading", "A2", 2), blk("paragraph", words(30)), blk("heading", "B", 1),
              blk("paragraph", words(30))]
        p, c = self.run_chunk(bl, min_tokens=5)
        self.assertEqual([x["breadcrumb"] for x in p], ["doc > A > A1", "doc > A > A2", "doc > B"])
        self.assertEqual(len(c), 3)
        self.assertEqual(c[0]["parent_id"], "doc-p0000")
        self.assertNotIn("article", p[0])

    def test_numbering_fallback(self):
        bl = [blk("paragraph", "1. Gioi thieu"), blk("paragraph", words(30)),
              blk("paragraph", "1.1 Muc tieu"), blk("paragraph", words(30)),
              blk("paragraph", "2. Ket luan"), blk("paragraph", words(30))]
        p, _ = self.run_chunk(bl, min_tokens=5)
        self.assertEqual([x["breadcrumb"] for x in p],
                         ["doc > 1. Gioi thieu", "doc > 1. Gioi thieu > 1.1 Muc tieu", "doc > 2. Ket luan"])

    def test_legal_gap_keeps_all_articles(self):
        bl = []
        for n in (1, 2, 5, 6):  # thieu 3, 4
            bl += [blk("heading", f"Điều {n}. Tieu de {n}", 2), blk("paragraph", words(20))]
        bl.insert(0, blk("heading", "Chương I", 1))
        with self.assertLogs("parser.chunker", level="WARNING"):
            p, c = self.run_chunk(bl, min_tokens=5)
        self.assertEqual([x["article"] for x in p], [1, 2, 5, 6])
        self.assertEqual(p[0]["chapter"], "I")
        self.assertEqual(len(c), 4)

    def test_no_headings_window_parents(self):
        bl = [blk("paragraph", words(100, f"p{i}x")) for i in range(40)]  # 4000 tokens
        p, c = self.run_chunk(bl)
        self.assertGreaterEqual(len(p), 3)
        self.assertTrue(all(x["breadcrumb"] == "doc" for x in p))
        self.assertTrue(all(len(x["text"].split()) <= 2000 for x in p))

    def test_table_atomic_and_oversize_split(self):
        small = "| h1 | h2 |\n| --- | --- |\n" + "\n".join(f"| a{i} | b{i} |" for i in range(10))
        big = "| h1 | h2 |\n| --- | --- |\n" + "\n".join(f"| r{i} x | y{i} z |" for i in range(300))
        _, c = self.run_chunk([blk("table", small)], ceiling=200)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["text"], small)
        self.assertEqual(c[0]["content_type"], "table")
        _, c = self.run_chunk([blk("table", big)], ceiling=200)
        self.assertGreater(len(c), 1)
        for ch in c:
            self.assertTrue(ch["text"].startswith("| h1 | h2 |\n| --- | --- |"))
            self.assertLessEqual(ch["tokens"], 200)

    def test_code_atomic(self):
        code = "```\n" + "\n".join(f"x{i} = {i}" for i in range(30)) + "\n```"
        bl = [blk("paragraph", words(150)), blk("code", code), blk("paragraph", words(150))]
        _, c = self.run_chunk(bl, ceiling=200, min_tokens=10)
        self.assertEqual([ch["content_type"] for ch in c].count("code") + sum(
            ch["content_type"] == "mixed" for ch in c) >= 1, True)
        self.assertEqual(sum(code in ch["text"] for ch in c), 1)

    def test_formula_block(self):
        _, c = self.run_chunk([blk("formula", "E = m c^2")], min_tokens=1)
        self.assertEqual(c[0]["content_type"], "formula")
        self.assertEqual(c[0]["text"], "E = m c^2")

    def test_min_tokens_merge(self):
        bl = [blk("paragraph", words(10)), blk("paragraph", words(150)), blk("paragraph", words(10))]
        _, c = self.run_chunk(bl, ceiling=200, min_tokens=80)
        self.assertEqual(len(c), 1)
        self.assertEqual(len(c[0]["block_ids"]), 3)

    def test_ceiling_respected_random(self):
        rnd = random.Random(7)
        bl = [blk("paragraph", ". ".join(words(rnd.randint(3, 25)) for _ in range(rnd.randint(1, 30))))
              for _ in range(200)]
        _, c = self.run_chunk(bl, ceiling=60, min_tokens=10)
        self.assertTrue(c)
        self.assertLessEqual(max(ch["tokens"] for ch in c), 60)

    def test_single_paragraph_cut_carries_sentence(self):
        text = " ".join(f"{words(9)[:-1]}." for _ in range(10))  # 10 cau x 9 tu
        _, c = self.run_chunk([blk("paragraph", text)], ceiling=30, min_tokens=1)
        self.assertGreater(len(c), 1)
        first_last = c[0]["text"].split(". ")[-1] if ". " in c[0]["text"] else c[0]["text"]
        self.assertIn(first_last, c[1]["text"])

    def test_zero_children_raises(self):
        with self.assertRaises(ChunkingError):
            self.run_chunk([blk("heading", "Only title", 1), blk("paragraph", "   ")])

    def test_deterministic_ids(self):
        bl = [blk("heading", "A", 1), blk("paragraph", words(500)), blk("paragraph", words(500))]
        self.assertEqual(self.run_chunk(bl), self.run_chunk(bl))
        _, c = self.run_chunk(bl)
        self.assertEqual(c[0]["child_id"], "doc-c0000")

    def test_skip_sections_and_stats(self):
        bl = [blk("heading", "Lich su", 2), blk("paragraph", words(30)),
              blk("heading", "Chú thích", 2), blk("paragraph", words(30, "ref")),
              blk("heading", "Con cua chu thich", 3), blk("paragraph", words(30, "sub")),
              blk("heading", "Tiep", 2), blk("paragraph", words(30, "keep"))]
        p, c = self.run_chunk(bl, min_tokens=5)
        self.assertEqual([x["breadcrumb"] for x in p], ["doc > Lich su", "doc > Tiep"])
        self.assertEqual(len(c), 2)
        self.assertEqual(last_stats()["skipped_sections"], ["Chú thích", "Con cua chu thich"])
        # khong dau / hoa thuong van khop; tat bang skip_sections=()
        bl2 = [blk("heading", "A", 1), blk("paragraph", words(30)), blk("heading", "THAM KHAO", 1),
               blk("paragraph", words(30, "r"))]
        self.assertEqual(len(self.run_chunk(bl2, min_tokens=5)[1]), 1)
        self.assertEqual(len(self.run_chunk(bl2, min_tokens=5, skip_sections=())[1]), 2)

    def test_legal_other_section_title_not_dropped(self):
        bl = [blk("heading", "Chương I", 1)]
        for n in (1, 2, 3):
            bl += [blk("heading", f"Điều {n}. Tham khảo ý kiến", 2), blk("paragraph", words(20))]
        _, c = self.run_chunk(bl, min_tokens=5)
        self.assertEqual(len(c), 3)  # tieu de chi chua 'Tham khảo', khong khop ^...$

    def test_noise_filter(self):
        iface = "\n".join(f'- [Lang{i}](https://k{i}.wikipedia.org/wiki/X "X")' for i in range(5))
        bl = [blk("paragraph", words(30)), blk("list", iface), blk("list", "- [↑](#cite_ref-1)"),
              blk("list", "- mot\n- hai\n- ba " + words(5)),
              blk("table", "| a |\n| --- |\n" + "\n".join(f'| [x](https://k.wikipedia.org/w) |' for _ in range(5)))]
        _, c = self.run_chunk(bl, min_tokens=1)
        txt = "\n".join(x["text"] for x in c)
        self.assertNotIn("wikipedia.org/wiki/X", txt)
        self.assertNotIn("cite_ref", txt)
        self.assertIn("- mot", txt)  # danh sach thuong con
        self.assertIn("| a |", txt)  # bang khong bao gio bi loc
        self.assertEqual(last_stats()["noise_blocks"], 2)

    def test_ocr_heading_block_split(self):
        bl = [blk("heading", "Chương III"), blk("heading", "TO CHỨC ĐÀO TẠO"),
              blk("heading", "Điều 16. Học kỳ\n" + words(20) + "\nĐiều 17. Khóa học\n" + words(20, "k")),
              blk("heading", "Điều 18. Khác"), blk("paragraph", words(20, "z")),
              blk("heading", "Điều 19. Nữa"), blk("paragraph", words(20, "y"))]
        p, c = self.run_chunk(bl, min_tokens=5)
        crumbs = [x["breadcrumb"] for x in p]
        self.assertEqual(crumbs[0], "doc > Chương III TO CHỨC ĐÀO TẠO > Điều 16. Học kỳ")
        self.assertEqual(crumbs[1], "doc > Chương III TO CHỨC ĐÀO TẠO > Điều 17. Khóa học")
        self.assertEqual([x["article"] for x in p][:2], [16, 17])
        self.assertTrue(all("\n" not in x["breadcrumb"] for x in p))
        self.assertTrue(c[0]["text"].startswith("w0"))
        # heading dai: breadcrumb cat con 120 ky tu
        _, c2 = self.run_chunk([blk("heading", "T" * 300, 1), blk("paragraph", words(30))], min_tokens=5)
        self.assertEqual(len(c2[0]["breadcrumb"]), len("doc > ") + 120)

    def test_jsonl_round_trip(self):
        _, c = self.run_chunk([blk("paragraph", "Xin chào Việt Nam " * 20)], min_tokens=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.jsonl"
            write_jsonl(c, path)
            import json
            back = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(back, c)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            p.write_text(json.dumps({"blocks": [blk("paragraph", "x")]}), encoding="utf-8")
            self.assertEqual(len(load_blocks(p)), 1)


if __name__ == "__main__":
    unittest.main()
