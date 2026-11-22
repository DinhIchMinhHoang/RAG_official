import os
import unittest
from contextlib import contextmanager

from store import pg
from store.settings import StoreSettings


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.calls.append(("execute", sql, params, self.conn.depth))

    def executemany(self, sql, rows):
        self.conn.calls.append(("executemany", sql, list(rows), self.conn.depth))

    def fetchall(self):
        return self.conn.rows

    def fetchone(self):
        return self.conn.rows[0] if self.conn.rows else None

    rowcount = 1


class FakeConn:
    def __init__(self, rows=None):
        self.calls, self.rows, self.depth, self.tx_count = [], rows or [], 0, 0

    def cursor(self):
        return FakeCursor(self)

    @contextmanager
    def transaction(self):
        self.depth += 1
        self.tx_count += 1
        try:
            yield
        finally:
            self.depth -= 1


EVIL = "x'; DROP TABLE documents; --"


class PgTest(unittest.TestCase):
    def test_schema_has_tables(self):
        sql = " ".join(pg.SCHEMA_SQL)
        for t in ("documents", "parents", "children", "ingest_events", "api_usage"):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {t}", sql)

    def test_init_schema_idempotent_sql(self):
        c = FakeConn()
        pg.init_schema(c)
        self.assertTrue(all("IF NOT EXISTS" in q[1] or "ON CONFLICT" in q[1] for q in c.calls))
        self.assertEqual(c.tx_count, 1)

    def test_values_never_in_sql(self):
        c = FakeConn()
        pg.upsert_document(c, {"doc_id": EVIL, "title": EVIL, "sha256": EVIL})
        pg.set_status(c, EVIL, "failed", error=EVIL)
        pg.find_by_sha256(c, EVIL)
        pg.get_parents(c, [EVIL])
        pg.delete_document(c, EVIL)
        pg.record_event(c, EVIL, EVIL, 1.0, {"a": EVIL})
        pg.record_usage(c, EVIL, EVIL, 1, 2, 0.1)
        pg.load_corpus(c, [EVIL])
        for _, sql, params, _ in c.calls:
            self.assertNotIn("DROP", sql)
            self.assertNotIn(EVIL, sql)
            self.assertIsNotNone(params)

    def test_bad_status_rejected(self):
        with self.assertRaises(ValueError):
            pg.set_status(FakeConn(), "d", "bogus")
        with self.assertRaises(ValueError):
            pg.upsert_document(FakeConn(), {"doc_id": "d", "status": "bogus"})

    def test_replace_chunks_one_transaction(self):
        c = FakeConn()
        parents = [{"parent_id": "p1", "text": "t"}]
        children = [{"child_id": "c1", "parent_id": "p1", "text": "t", "bbox": [1, 2], "block_ids": None}]
        pg.replace_chunks(c, "d1", parents, children)
        self.assertEqual(c.tx_count, 1)
        self.assertTrue(all(call[3] == 1 for call in c.calls))
        kinds = [(k, s.split()[0:3]) for k, s, _, _ in c.calls]
        self.assertEqual([s[0] for _, s in kinds], ["DELETE", "DELETE", "INSERT", "INSERT"])
        self.assertEqual(c.calls[2][2][0][1], "d1")  # doc_id forced onto parent rows
        self.assertEqual(c.calls[3][2][0][1], "p1")

    def test_get_parents_preserves_order(self):
        c = FakeConn(rows=[{"parent_id": "a"}, {"parent_id": "b"}, {"parent_id": "c"}])
        got = pg.get_parents(c, ["c", "zz", "a", "b"])
        self.assertEqual([r["parent_id"] for r in got], ["c", "a", "b"])
        self.assertEqual(pg.get_children(FakeConn(), []), [])

    def test_load_corpus_params(self):
        c = FakeConn()
        pg.load_corpus(c)
        self.assertEqual([x[2] for x in c.calls], [(), ()])
        c = FakeConn()
        pg.load_corpus(c, ["d"])
        self.assertEqual([x[2] for x in c.calls], [(["d"],), (["d"],)])


@unittest.skipUnless(os.environ.get("RAG_INTEGRATION") == "1", "needs RAG_INTEGRATION=1 and running Postgres")
class PgIntegration(unittest.TestCase):
    def test_roundtrip(self):
        with pg.connect(StoreSettings.from_env()) as conn:
            pg.init_schema(conn)
            pg.init_schema(conn)
            pg.upsert_document(conn, {"doc_id": "it_doc", "sha256": "it_sha"})
            pg.replace_chunks(conn, "it_doc", [{"parent_id": "it_p", "text": "t"}],
                              [{"child_id": "it_c", "parent_id": "it_p", "text": "t", "bbox": [1]}])
            kids, pars = pg.load_corpus(conn, ["it_doc"])
            self.assertEqual((len(kids), len(pars)), (1, 1))
            pg.delete_document(conn, "it_doc")
            conn.commit()


if __name__ == "__main__":
    unittest.main()
