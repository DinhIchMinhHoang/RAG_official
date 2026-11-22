import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.answering import (REFUSAL_TEXT, SYSTEM_PROMPT, assemble_contexts, build_messages,  # noqa: E402
                           parse_answer, quote_support)

count = lambda s: len(s.split())  # noqa: E731


def P(i, text, **kw):
    return {"parent_id": f"p{i}", "doc_id": "d", "title": "T", "breadcrumb": f"B{i}",
            "page_start": i, "page_end": i, "text": text, **kw}


class AssembleTests(unittest.TestCase):
    def test_fit_and_numbering(self):
        r = assemble_contexts([P(1, "a b c"), P(2, "d e")], count, 10)
        self.assertEqual([c["source_id"] for c in r], ["S1", "S2"])
        self.assertEqual(r[1]["parent_id"], "p2")
        self.assertFalse(any(c["truncated"] for c in r))

    def test_drops_whole_items(self):
        r = assemble_contexts([P(1, "a b c"), P(2, "x " * 10), P(3, "d e")], count, 6)
        self.assertEqual([c["parent_id"] for c in r], ["p1", "p3"])
        self.assertEqual([c["source_id"] for c in r], ["S1", "S2"])
        self.assertEqual(r[0]["text"], "a b c")

    def test_first_too_big_truncated_at_paragraph(self):
        text = "a b c\n\nd e f\n\ng h i j k"
        r = assemble_contexts([P(1, text), P(2, "z")], count, 7)
        self.assertTrue(r[0]["truncated"])
        self.assertEqual(r[0]["text"], "a b c\nd e f")
        self.assertEqual(r[1]["parent_id"], "p2")

    def test_history_trimmed_to_six(self):
        h = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        msgs = build_messages("Q?", assemble_contexts([P(1, "alpha")], count), h)
        u = msgs[1]["content"]
        self.assertNotIn("m3", u)
        self.assertIn("m4", u)
        self.assertIn("m9", u)
        self.assertIn("[S1]", u)
        self.assertEqual(msgs[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(len(msgs), 2)

    def test_empty_question_and_no_contexts(self):
        with self.assertRaises(ValueError):
            build_messages(" ", [])
        self.assertIn("Không có nguồn", build_messages("Q", [])[1]["content"])

    def test_prompt_is_generic(self):
        self.assertNotIn("Quy chế", SYSTEM_PROMPT)
        self.assertIn(REFUSAL_TEXT, SYSTEM_PROMPT)


class ParseTests(unittest.TestCase):
    ctx = assemble_contexts([P(1, "sinh viên phải đóng học phí đúng hạn mỗi học kỳ"), P(2, "điểm D là điểm đạt")], count)

    def test_citations_invalid_and_groups(self):
        r = parse_answer("Phải đóng đúng hạn [S1]. Điểm D đạt [S2, S9] [S1].", self.ctx)
        self.assertEqual([c["source_id"] for c in r["citations"]], ["S1", "S2"])
        self.assertEqual(r["citations"][0]["page_start"], 1)
        self.assertEqual(r["invalid_citations"], ["S9"])
        self.assertTrue(r["has_citation"])
        self.assertFalse(r["refused"])

    def test_refusal(self):
        for t in (REFUSAL_TEXT, "Văn bản không đề cập đến vấn đề này.", "Nguồn không có thông tin về học bổng."):
            r = parse_answer(t, self.ctx)
            self.assertTrue(r["refused"], t)
            self.assertFalse(r["has_citation"])

    def test_quote_support(self):
        ctx = [{"source_id": "S1", "text": "sinh viên phải đóng học phí đúng hạn mỗi học kỳ"},
               {"source_id": "S2", "text": "hoàn toàn khác biệt nội dung ở đây"}]
        # 5 words -> 2 shingles, both in S1
        self.assertEqual(quote_support("Sinh viên phải đóng học phí [S1]", ctx), 1.0)
        # cited S2 only: nothing matches
        self.assertEqual(quote_support("Sinh viên phải đóng học phí [S2]", ctx), 0.0)
        # 6 words -> 3 shingles, first 2 match
        self.assertAlmostEqual(quote_support("sinh viên phải đóng học xyz [S1]", ctx), 2 / 3)
        self.assertIsNone(quote_support("quá ngắn [S1]", ctx))


if __name__ == "__main__":
    unittest.main()
