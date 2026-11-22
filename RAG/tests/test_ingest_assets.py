"""Run from RAG/: python -m unittest tests.test_ingest_assets (fake MinIO + mocked Postgres, no network)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ingest as ing  # noqa: E402
from store import objects, pg  # noqa: E402
from store.settings import StoreSettings  # noqa: E402
from tests.test_store_objects import FakeMinio  # noqa: E402


def mirror(result):
    mc, out = FakeMinio(), []
    with mock.patch.object(ing, "stack_ready", return_value=True), mock.patch.object(objects, "client",
            return_value=mc), mock.patch.object(pg, "connect"), mock.patch.object(pg, "delete_document"), \
            mock.patch.object(pg, "upsert_document"), mock.patch.object(pg, "replace_chunks"), \
            mock.patch.object(pg, "set_status"):
        ok = ing.mirror_stack([result], settings=StoreSettings.from_env(), out=out.append)
    return ok, mc, out


class MirrorAssetsTest(unittest.TestCase):
    def result(self, d, parsed):
        src = Path(d) / "doc.txt"
        src.write_text("x", encoding="utf-8")
        return {"status": "ok", "doc_id": "doc-abc123", "path": str(src), "pages": 1, "children": 1,
                "_parents": [], "_children": [], "_payload": {"blocks": []}, "_parsed_dir": str(parsed)}

    def test_uploads_assets(self):
        with tempfile.TemporaryDirectory() as d:
            parsed = Path(d) / "parsed" / "doc-abc123"
            (parsed / "assets").mkdir(parents=True)
            (parsed / "assets" / "p0001_b00002.png").write_bytes(b"png")
            ok, mc, out = mirror(self.result(d, parsed))
        self.assertTrue(ok)
        keys = [c[2] for c in mc.calls if c[0] == "fput"]
        self.assertIn("assets/doc-abc123/p0001_b00002.png", keys)
        self.assertEqual(len(keys), 2)  # raw file + 1 asset
        self.assertFalse(any("CẢNH BÁO" in o for o in out))

    def test_no_assets_dir(self):
        with tempfile.TemporaryDirectory() as d:
            parsed = Path(d) / "parsed" / "doc-abc123"
            parsed.mkdir(parents=True)
            ok, mc, out = mirror(self.result(d, parsed))
        self.assertTrue(ok)
        self.assertEqual(len([c for c in mc.calls if c[0] == "fput"]), 1)
        self.assertFalse(any("CẢNH BÁO" in o for o in out))


if __name__ == "__main__":
    unittest.main()
