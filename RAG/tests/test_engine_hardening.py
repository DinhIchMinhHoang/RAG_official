"""Engine hardening: cold-start deadline, docling opt-in, image-table plausibility. Fake worker only."""
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engine_router import Base, make_png  # noqa: E402
from parser import engine_router as er  # noqa: E402


class Hardening(Base):
    def test_cold_start_cannot_pass_budget_plus_5(self):
        os.environ["FAKE_BOOT"] = "40"  # worker would boot far past the budget
        t = time.perf_counter()
        st = self.run_router(self.formulas(3), ema={"load_glm": 1.0}, rkw={"budget_s": 15.0})
        wall = time.perf_counter() - t
        self.assertLessEqual(wall, 15.0 + er.OVERRUN_S + 1.0)
        self.assertLessEqual(st["used_s"], 15.0 + er.OVERRUN_S + 1.0)
        self.assertEqual(st["engines"]["glm"]["state"], "failed")

    def test_auto_is_glm_only(self):
        self.assertEqual(self.router("auto")._wanted(), ["glm"])
        self.assertEqual(self.router("glm,docling")._wanted(), ["docling", "glm"])
        self.assertEqual(self.router("docling")._wanted(), ["docling"])

    def test_plausible_table(self):
        md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        self.assertTrue(er.plausible_table(md, "a b 1 2")[0])
        self.assertTrue(er.plausible_table(md, "")[0])
        self.assertFalse(er.plausible_table(md, "x y z w q r")[0])
        self.assertFalse(er.plausible_table("| a |\n| 1 |")[0])
        self.assertFalse(er.plausible_table("| a | b |")[0])

    def test_html_table_to_markdown(self):
        h = '<table><tr><td></td><td colspan="2">P</td></tr><tr><td>1</td><td>2.7</td><td>a|b</td></tr></table>'
        md = er.table_to_markdown(h)
        self.assertEqual(md, "|  | P |  |\n| --- | --- | --- |\n| 1 | 2.7 | a/b |")
        self.assertTrue(er.plausible_table(md, "P 1 2.7")[0])
        self.assertEqual(er.table_to_markdown("| a | b |"), "| a | b |")

    def test_image_table_not_requested_when_budget_low(self):
        make_png(self.d / "fig.png", 99)
        fig = {"id": "fig1", "type": "figure", "text": "", "location": {"page_num": 2}, "asset": str(self.d / "fig.png"),
               "metadata": {"kind": "raster", "ocr_text": "a 1 2\nb 3 4\nc 5 6"}}
        st = self.run_router([fig], rkw={"budget_s": 20.0}, ema={"load_glm": 1.0})
        self.assertEqual(st["new_blocks"], [])

    def test_image_table_cap(self):
        figs = []
        for i in range(5):
            make_png(self.d / f"g{i}.png", 50 + i)
            figs.append({"id": f"g{i}", "type": "figure", "text": "", "location": {"page_num": 1}, "asset": str(self.d / f"g{i}.png"),
                         "metadata": {"kind": "raster", "ocr_text": "a 1 2\nb 3 4\nc 5 6"}})
        st = self.run_router(figs)
        self.assertEqual(len(st["new_blocks"]), 3)


if __name__ == "__main__":
    unittest.main()
