import unittest

from store.settings import StoreSettings


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        s = StoreSettings.from_env({})
        self.assertEqual((s.pg_host, s.pg_port, s.pg_db), ("127.0.0.1", 5432, "rag"))
        self.assertEqual(s.minio_endpoint, "127.0.0.1:9000")
        self.assertFalse(s.minio_secure)
        self.assertEqual(s.qdrant_url, "http://127.0.0.1:6333")
        self.assertEqual(s.qdrant_collection, "rag_children")
        self.assertEqual(s.qdrant_api_key, "")

    def test_override(self):
        s = StoreSettings.from_env({"RAG_PG_PORT": "5544", "RAG_MINIO_SECURE": "true", "RAG_MINIO_BUCKET": "b",
                                    "RAG_QDRANT_API_KEY": "k"})
        self.assertEqual((s.pg_port, s.minio_secure, s.minio_bucket, s.qdrant_api_key), (5544, True, "b", "k"))

    def test_redacted_and_repr_hide_secrets(self):
        s = StoreSettings.from_env({"RAG_PG_PASSWORD": "pw-XYZ", "RAG_MINIO_SECRET_KEY": "sk-XYZ",
                                    "RAG_MINIO_ACCESS_KEY": "ak-XYZ", "RAG_QDRANT_API_KEY": "qk-XYZ"})
        text = str(s.redacted()) + repr(s) + str(s)
        for secret in ("pw-XYZ", "sk-XYZ", "ak-XYZ", "qk-XYZ"):
            self.assertNotIn(secret, text)
        self.assertEqual(s.redacted()["pg_host"], "127.0.0.1")

    def test_frozen(self):
        with self.assertRaises(Exception):
            StoreSettings().pg_host = "x"


if __name__ == "__main__":
    unittest.main()
