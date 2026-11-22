import os
import tempfile
import unittest

from store import doctor, objects
from store.settings import StoreSettings


class FakeMinio:
    def __init__(self):
        self.calls, self.buckets, self.objs = [], set(), {}

    def bucket_exists(self, b):
        return b in self.buckets

    def make_bucket(self, b):
        self.buckets.add(b)

    def fput_object(self, b, k, path, content_type=None):
        self.calls.append(("fput", b, k, content_type))

    def put_object(self, b, k, stream, n, content_type=None):
        self.objs[k] = stream.read()
        self.calls.append(("put", b, k, n))

    def fget_object(self, b, k, path):
        self.calls.append(("fget", b, k))

    def presigned_get_object(self, b, k, expires):
        return f"http://x/{b}/{k}?e={int(expires.total_seconds())}"

    def list_objects(self, b, prefix=None, recursive=False):
        return [type("O", (), {"object_name": k}) for k in self.objs if k.startswith(prefix)]

    def remove_object(self, b, k):
        self.objs.pop(k)


class KeyTest(unittest.TestCase):
    def test_no_traversal_or_absolute(self):
        for k in ("../../etc/passwd", "/abs/x", "a/../b", "..\\..\\win", "a//b/./c"):
            out = objects.safe_key(k)
            self.assertNotIn("..", out)
            self.assertFalse(out.startswith("/"))
            self.assertNotIn("//", out)
        self.assertEqual(objects.safe_key("a/../b"), "a/b")

    def test_filename_cannot_escape(self):
        k = objects.raw_key("abc", "../../evil.pdf")
        self.assertEqual(k, "raw/abc/evil.pdf")
        self.assertNotIn("..", objects.raw_key("abc", "a..b.pdf"))

    def test_unicode_normalised(self):
        nfd = "Quyắt.pdf"  # decomposed
        k = objects.raw_key("abc", nfd)
        import unicodedata
        self.assertTrue(unicodedata.is_normalized("NFC", k))
        self.assertEqual(k, objects.raw_key("abc", unicodedata.normalize("NFC", nfd)))

    def test_long_name_capped_keeps_ext(self):
        k = objects.raw_key("abc", "a" * 500 + ".pdf")
        seg = k.split("/")[-1]
        self.assertLessEqual(len(seg), objects.MAX_SEG)
        self.assertTrue(seg.endswith(".pdf"))

    def test_page_and_blocks(self):
        self.assertEqual(objects.page_key("d1", 3), "pages/d1/00003.png")
        self.assertEqual(objects.blocks_key("../d1"), "blocks/d1.json")


class OpsTest(unittest.TestCase):
    def test_ops(self):
        c = FakeMinio()
        self.assertTrue(objects.ensure_bucket(c, "rag"))
        self.assertFalse(objects.ensure_bucket(c, "rag"))
        objects.put_bytes(c, "rag", b"hi", "/../a/b.txt")
        self.assertIn("a/b.txt", c.objs)
        with tempfile.TemporaryDirectory() as d:
            objects.put_file(c, "rag", os.path.join(d, "f"), "../f.bin")
            objects.get_to_file(c, "rag", "../f.bin", os.path.join(d, "g"))
        self.assertEqual(c.calls[1][2], "f.bin")
        self.assertIn("e=60", objects.presigned_get(c, "rag", "a/b.txt", 60))
        self.assertEqual(objects.delete_prefix(c, "rag", "a"), 1)
        with self.assertRaises(ValueError):
            objects.delete_prefix(c, "rag", "../")


class DoctorTest(unittest.TestCase):
    def test_down_services_exit_1(self):
        s = StoreSettings.from_env({"RAG_PG_PORT": "1", "RAG_MINIO_ENDPOINT": "127.0.0.1:1",
                                    "RAG_QDRANT_URL": "http://127.0.0.1:1", "RAG_PG_PASSWORD": "pw-SECRET"})
        lines = []
        self.assertEqual(doctor.run(s, out=lines.append), 1)
        self.assertEqual(len(lines), 3)
        for line, name in zip(lines, ("postgres", "minio", "qdrant")):
            self.assertTrue(line.startswith(f"FAIL {name}: "), line)
            self.assertNotIn("pw-SECRET", line)


@unittest.skipUnless(os.environ.get("RAG_INTEGRATION") == "1", "needs RAG_INTEGRATION=1 and running MinIO")
class MinioIntegration(unittest.TestCase):
    def test_roundtrip(self):
        s = StoreSettings.from_env()
        c = objects.client(s)
        objects.ensure_bucket(c, s.minio_bucket)
        objects.put_bytes(c, s.minio_bucket, b"x", "it/test.bin")
        self.assertEqual(objects.delete_prefix(c, s.minio_bucket, "it"), 1)


if __name__ == "__main__":
    unittest.main()
