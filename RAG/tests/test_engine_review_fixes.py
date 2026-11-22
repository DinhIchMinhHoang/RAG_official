"""Reviewer fixes for the engine router: html spans, stats file, budget, asset path, spawn failure, fd 1 redirect."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import engine_router as er  # noqa: E402
from test_engine_router import FAKE, make_png  # noqa: E402


class Html(unittest.TestCase):
    def md(self, span):
        return er.table_to_markdown(f"<table><tr><td colspan='{span}'>a</td><td>b</td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>")

    def test_unicode_digit_span(self):
        self.assertIn("| a | b |", self.md("²"))

    def test_huge_span_capped(self):
        out = self.md("999999999")
        self.assertLess(len(out), 2000)
        self.assertEqual(self.md("2").splitlines()[0].count("|"), 4)

    def test_malformed(self):
        for s in ("<table><tr><td>a<td colspan=", "<table></td></tr><tr><td", "<table"):
            self.assertIsInstance(er.table_to_markdown(s), str)
        with mock.patch.object(er._Html, "feed", side_effect=RuntimeError):
            self.assertEqual(er.table_to_markdown("<table><tr><td>a</td></tr></table>"), "")


class Stats(unittest.TestCase):
    def ema(self, content):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "engine_stats.json").write_bytes(content)
            return er.EngineRouter(mode="glm", cache_dir=d).ema

    def test_bad_files_use_seeds(self):
        for c in (b"[1,2]", b'{"glm_formula":"x"}', b'{"glm_formula":NaN}', b'{"glm_formula":Infinity}',
                  b'{"glm_formula":-1}', b'{"glm_formula":1e9}', b'{"glm_formula":true}', b"\xff\xfe\x00garbage"):
            self.assertEqual(self.ema(c), er.SEED, c)

    def test_good_value_loaded(self):
        e = self.ema(b'{"glm_formula":1.5,"bogus":3}')
        self.assertEqual(e["glm_formula"], 1.5)
        self.assertNotIn("bogus", e)

    def test_atomic_save(self):
        with tempfile.TemporaryDirectory() as d:
            r = er.EngineRouter(mode="glm", cache_dir=d)
            r._save_ema()
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["engine_stats.json"])


class Budget(unittest.TestCase):
    def test_bad_budget_is_zero(self):
        for b in (float("nan"), -5.0, 0.0):
            self.assertEqual(er.EngineRouter(budget_s=b).budget_s, 0.0)
        self.assertEqual(er.EngineRouter(budget_s=30).budget_s, 30)

    def test_zero_budget_skips(self):
        with tempfile.TemporaryDirectory() as d:
            r = er.EngineRouter(mode="glm", budget_s=float("nan"), python_paths={"glm": (sys.executable, FAKE)})
            st = r.enhance([], Path(d) / "x.pdf", d)
            self.assertEqual(st["engines"]["glm"]["state"], "skipped")

    def test_cli_arg(self):
        import argparse
        import cli
        self.assertEqual(cli.budget_arg("12.5"), 12.5)
        self.assertEqual(cli.budget_arg("0"), 0.0)
        for bad in ("nan", "-1", "inf", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                cli.budget_arg(bad)

    def test_ingest_make_router(self):
        import ingest
        with tempfile.TemporaryDirectory() as d:
            for b in (float("nan"), -1.0, float("inf")):
                self.assertEqual(ingest.make_router("glm", b, d).budget_s, 0.0)


class Assets(unittest.TestCase):
    def test_relative_asset_not_cwd(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cwd:
            d, cwd = Path(d), Path(cwd)
            (d / "assets").mkdir()
            make_png(d / "assets" / "a.png", 1)
            make_png(cwd / "assets_a.png", 2)
            old = os.getcwd()
            os.chdir(cwd)
            try:
                p = er._asset_path({"asset": "assets/a.png"}, d)
                self.assertEqual(p, (d / "assets" / "a.png").resolve())
                self.assertIsNone(er._asset_path({"asset": "assets_a.png"}, d / "nope"))
            finally:
                os.chdir(old)


class Worker(unittest.TestCase):
    def test_popen_failure_cleans_up(self):
        w = er._Worker("glm", sys.executable, FAKE)
        with mock.patch.object(er.subprocess, "Popen", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                w.start(5)
        self.assertIsNone(w.log)
        self.assertNotIn(w, er._LIVE)

    def test_stray_fd1_write_does_not_corrupt_protocol(self):
        w = er._Worker("glm", sys.executable, FAKE)
        try:
            with mock.patch.dict(os.environ, {"FAKE_STRAY_FD1": "1"}):
                self.assertTrue(w.start(20).get("ready"))
            self.assertEqual(w.request("formula", 10, image="x")["latex"], "\\frac{a}{b}")
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
