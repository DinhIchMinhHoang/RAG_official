import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parser"))
import parse_document as pd  # noqa: E402


def fake_pt(langs=("eng", "vie")):
    m = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=None),
        get_languages=lambda: list(langs),
    )
    return m


def make(env, which=None, win_exists=False, langs=("eng", "vie")):
    pt = fake_pt(langs)
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.dict(sys.modules, {"pytesseract": pt}), \
         mock.patch("shutil.which", return_value=which), \
         mock.patch("os.path.exists", return_value=win_exists):
        return pd.OCR(), pt


class OcrConfigTest(unittest.TestCase):
    def test_env_wins(self):
        _, pt = make({"TESSERACT_CMD": "/env/t"}, which="/path/t", win_exists=True)
        self.assertEqual(pt.pytesseract.tesseract_cmd, "/env/t")

    def test_which_then_windows(self):
        _, pt = make({}, which="/path/t", win_exists=True)
        self.assertEqual(pt.pytesseract.tesseract_cmd, "/path/t")
        _, pt = make({}, which=None, win_exists=True)
        self.assertEqual(pt.pytesseract.tesseract_cmd, pd._WIN_TESSERACT)

    def test_no_command(self):
        with self.assertRaisesRegex(RuntimeError, "TESSERACT_CMD"):
            make({}, which=None, win_exists=False)

    def test_missing_language(self):
        with self.assertRaises(RuntimeError) as c:
            make({"TESSERACT_CMD": "t"}, langs=("eng",))
        self.assertIn("vie", str(c.exception))
        self.assertIn("TESSDATA_PREFIX", str(c.exception))

    def test_ocr_lang_override(self):
        ocr, _ = make({"TESSERACT_CMD": "t", "OCR_LANG": "eng"}, langs=("eng",))
        self.assertEqual(ocr.lang, "eng")
        ocr, _ = make({"TESSERACT_CMD": "t"})
        self.assertEqual(ocr.lang, "vie+eng")


if __name__ == "__main__":
    unittest.main()
