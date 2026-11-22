import tempfile
import unittest
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from parser.parse_document import parse_pptx


def run(prs):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.pptx"
        prs.save(str(p))
        return parse_pptx(p, None)[0]  # ocr unused without pictures


class PptxFixesTest(unittest.TestCase):
    def test_title_and_chart(self):
        prs = Presentation()
        s = prs.slides.add_slide(prs.slide_layouts[5])  # title only
        s.shapes.title.text = "Quarterly"
        cd = CategoryChartData()
        cd.categories = ["1st Qtr", "2nd Qtr"]
        cd.add_series("Sales", (8.2, 3.2))
        s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(2), Inches(5), Inches(3), cd)
        blocks = run(prs)
        title = next(b for b in blocks if b.text == "Quarterly")
        self.assertEqual((title.type, title.level), ("heading", 1))
        fig = next(b for b in blocks if b.type == "figure")
        self.assertIn("8.2", fig.text)
        self.assertNotIn("8.200000000000001", fig.text)
        chart = fig.metadata["chart"]
        self.assertEqual(chart["series"][0]["values"], [8.2, 3.2])
        self.assertEqual(chart["categories"], ["1st Qtr", "2nd Qtr"])
        self.assertEqual(chart["series"][0]["name"], "Sales")
        self.assertIsInstance(chart["type"], str)

    def test_float_noise_value(self):
        prs = Presentation()
        s = prs.slides.add_slide(prs.slide_layouts[6])
        cd = CategoryChartData()
        cd.categories = ["a"]
        cd.add_series("S", (8.200000000000001,))
        s.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(1), Inches(1), Inches(4), Inches(3), cd)
        fig = next(b for b in run(prs) if b.type == "figure")
        self.assertEqual(fig.text, "S: a=8.2")

    def test_empty_table_skipped(self):
        prs = Presentation()
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1))
        t = s.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(4), Inches(1))
        t.table.cell(1, 1).text = "x"
        tables = [b for b in run(prs) if b.type == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("x", tables[0].text)

    def test_placeholder_without_position_no_crash(self):
        prs = Presentation()
        s = prs.slides.add_slide(prs.slide_layouts[1])  # title + body, inherit position
        s.shapes.title.text = "T"
        s.placeholders[1].text = "body"
        s.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text_frame.text = "box"
        texts = [b.text for b in run(prs)]
        self.assertIn("T", texts)
        self.assertIn("body", texts)


if __name__ == "__main__":
    unittest.main()
