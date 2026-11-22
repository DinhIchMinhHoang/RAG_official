"""Engine budget behaviour: fake worker only."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engine_router import Base, make_png  # noqa: E402
from parser import engine_router as er  # noqa: E402


def table(none_ratio, fill, rows=2, cols=2):
    return {"id": "t1", "type": "table", "text": "| a | b |\n| 1 | 2 |", "location": {"page_num": 1}, "bbox": [0.0, 0.0, 1.0, 1.0],
            "metadata": {"page": 1, "none_ratio": none_ratio, "fill": fill, "n_rows": rows, "n_cols": cols}}


class Budget(Base):
    def test_target_default_120(self):
        self.assertEqual(er.TARGET_TOTAL_S, 120.0)
        import ingest  # class default stays 110 (older test pins it); the ingest path must use 120
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ingest.make_router("auto", 45.0, d).target_total_s, 120.0)

    def test_clean_tables_never_launch_docling(self):
        with mock.patch.object(er.subprocess, "Popen", side_effect=AssertionError("spawned")):
            st = self.run_router([table(0.05, 0.95, 4, 5)], mode="docling")  # 1 blank cell of 20
        self.assertEqual(st["engines"]["docling"]["state"], "skipped")
        st = self.run_router([table(0.0, 1.0)], mode="docling")
        self.assertEqual(st["engines"]["docling"]["reason"], "no suspect tables")

    def test_broken_table_still_launches_docling(self):
        st = self.run_router([table(0.3, 0.7, 4, 5)], mode="docling")
        self.assertEqual(st["engines"]["docling"]["state"], "ok")

    def test_spawn_time_counts_toward_used(self):
        os.environ["FAKE_BOOT"] = "1.5"
        st = self.run_router(self.formulas(3), ema={"load_glm": 1.5})
        self.assertGreaterEqual(st["used_s"], 1.5)
        self.assertGreaterEqual(st["engines"]["glm"]["s"], 1.5)

    def test_slow_spawn_skips_by_measured_cost(self):
        st = self.run_router(self.formulas(3), ema={"load_glm": 100.0})
        self.assertEqual(st["engines"]["glm"]["state"], "skipped")

    def test_trivial_crops_skipped(self):
        bs = self.formulas(3, "x") + self.formulas(3, "frac ab")
        for i, b in enumerate(bs[3:]):
            b["id"] = f"g{i}"
        self.assertEqual(len(er.EngineRouter._formula_cands(bs, self.d)), 3)
        make_png(self.d / "tiny.png", 5, 20, 20)  # 20 px at 200 dpi = 7 pt
        bs[3]["asset"] = str(self.d / "tiny.png")
        self.assertEqual(len(er.EngineRouter._formula_cands(bs, self.d)), 0)  # only 2 worthwhile left (< 3)

    def test_batching_reduces_requests(self):
        log = self.d / "log.txt"
        os.environ.update(FAKE_LOG=str(log), FAKE_MAX_BATCH="4")
        st = self.run_router(self.formulas(8))
        self.assertEqual(st["engines"]["glm"]["formula"]["done"], 8)
        self.assertEqual(len(log.read_text().splitlines()), 2)

    def test_hard_deadline(self):
        os.environ["FAKE_SLEEP"] = "30"
        blocks = self.formulas(3)
        import time
        t = time.perf_counter()
        st = self.run_router(blocks, rkw={"budget_s": 16.0})
        self.assertLess(time.perf_counter() - t, 16.0 + er.OVERRUN_S)
        self.assertEqual(st["engines"]["glm"]["formula"]["done"], 0)


if __name__ == "__main__":
    unittest.main()
