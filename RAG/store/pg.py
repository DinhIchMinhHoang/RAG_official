"""Postgres layer (psycopg 3). Only parameterized SQL; identifiers are constants."""
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA_VERSION = 1
STATUSES = ("queued", "parsing", "ocr", "chunking", "indexing", "ready", "failed")

SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS schema_version (version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_id text PRIMARY KEY, title text, filename text, mime text,
        sha256 text UNIQUE, size_bytes bigint, pages int,
        status text NOT NULL DEFAULT 'queued'
            CHECK (status IN ('queued','parsing','ocr','chunking','indexing','ready','failed')),
        progress_done int, progress_total int, error text, object_key text,
        created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS parents (
        parent_id text PRIMARY KEY, doc_id text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        ord int, title text, breadcrumb text, text text, page_start int, page_end int, chapter text, article text)""",
    """CREATE TABLE IF NOT EXISTS children (
        child_id text PRIMARY KEY, parent_id text NOT NULL REFERENCES parents(parent_id) ON DELETE CASCADE,
        doc_id text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        ord int, breadcrumb text, text text, page_start int, page_end int,
        bbox jsonb, block_ids jsonb, tokens int, content_type text)""",
    """CREATE TABLE IF NOT EXISTS ingest_events (
        id bigserial PRIMARY KEY, doc_id text, stage text, seconds double precision, details jsonb,
        created_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS api_usage (
        id bigserial PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), stage text, model text,
        prompt_tokens int, completion_tokens int, cost numeric)""",
    "CREATE INDEX IF NOT EXISTS idx_parents_doc ON parents(doc_id, ord)",
    "CREATE INDEX IF NOT EXISTS idx_children_doc ON children(doc_id, ord)",
    "CREATE INDEX IF NOT EXISTS idx_children_parent ON children(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_doc ON ingest_events(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_usage_ts ON api_usage(ts)",
)

PARENT_COLS = ("parent_id", "doc_id", "ord", "title", "breadcrumb", "text", "page_start", "page_end", "chapter", "article")
CHILD_COLS = ("child_id", "parent_id", "doc_id", "ord", "breadcrumb", "text", "page_start", "page_end",
              "bbox", "block_ids", "tokens", "content_type")
JSON_COLS = ("bbox", "block_ids")
DOC_COLS = ("doc_id", "title", "filename", "mime", "sha256", "size_bytes", "pages", "status",
            "progress_done", "progress_total", "error", "object_key")


def connect(s, connect_timeout=5):
    return psycopg.connect(host=s.pg_host, port=s.pg_port, dbname=s.pg_db, user=s.pg_user,
                           password=s.pg_password, row_factory=dict_row, connect_timeout=connect_timeout)


def init_schema(conn):
    with conn.transaction(), conn.cursor() as cur:
        for sql in SCHEMA_SQL:
            cur.execute(sql)
        cur.execute("INSERT INTO schema_version (version) VALUES (%s) ON CONFLICT DO NOTHING", (SCHEMA_VERSION,))


def _check_status(status):
    if status not in STATUSES:
        raise ValueError(f"status invalid: {status}")


def upsert_document(conn, doc):
    """doc: dict with doc_id + any DOC_COLS; on update only the given keys change."""
    status = doc.get("status", "queued")
    _check_status(status)
    vals = {c: doc.get(c) for c in DOC_COLS}
    vals["status"] = status
    present = [c for c in DOC_COLS if c != "doc_id" and (c == "status" or c in doc)]
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in present)
    sql = (f"INSERT INTO documents ({', '.join(DOC_COLS)}) VALUES ({', '.join(['%s'] * len(DOC_COLS))}) "
           f"ON CONFLICT (doc_id) DO UPDATE SET {sets}, updated_at = now()")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(sql, tuple(vals[c] for c in DOC_COLS))


def set_status(conn, doc_id, status, done=None, total=None, error=None):
    _check_status(status)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = %s, progress_done = COALESCE(%s, progress_done), "
            "progress_total = COALESCE(%s, progress_total), error = %s, updated_at = now() WHERE doc_id = %s",
            (status, done, total, error, doc_id))


def _row(cols, d):
    return tuple(Jsonb(d.get(c)) if c in JSON_COLS and d.get(c) is not None else d.get(c) for c in cols)


def replace_chunks(conn, doc_id, parents, children):
    """Delete old rows of the doc, insert new ones, all in one transaction."""
    marks = lambda cols: ", ".join(["%s"] * len(cols))
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM children WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM parents WHERE doc_id = %s", (doc_id,))
        pr = [_row(PARENT_COLS, {**p, "doc_id": doc_id}) for p in parents]
        cr = [_row(CHILD_COLS, {**c, "doc_id": doc_id}) for c in children]
        if pr:
            cur.executemany(f"INSERT INTO parents ({', '.join(PARENT_COLS)}) VALUES ({marks(PARENT_COLS)})", pr)
        if cr:
            cur.executemany(f"INSERT INTO children ({', '.join(CHILD_COLS)}) VALUES ({marks(CHILD_COLS)})", cr)


def _by_ids(conn, sql, key, ids):
    ids = list(ids)
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(sql, (ids,))
        rows = {r[key]: r for r in cur.fetchall()}
    return [rows[i] for i in ids if i in rows]


def get_parents(conn, ids):
    return _by_ids(conn, "SELECT * FROM parents WHERE parent_id = ANY(%s)", "parent_id", ids)


def get_children(conn, ids):
    return _by_ids(conn, "SELECT * FROM children WHERE child_id = ANY(%s)", "child_id", ids)


def list_documents(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return cur.fetchall()


def find_by_sha256(conn, sha):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE sha256 = %s", (sha,))
        return cur.fetchone()


def delete_document(conn, doc_id):
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        return cur.rowcount


def record_event(conn, doc_id, stage, seconds=None, details=None):
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("INSERT INTO ingest_events (doc_id, stage, seconds, details) VALUES (%s, %s, %s, %s)",
                    (doc_id, stage, seconds, Jsonb(details) if details is not None else None))


def record_usage(conn, stage, model, prompt_tokens=None, completion_tokens=None, cost=None):
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("INSERT INTO api_usage (stage, model, prompt_tokens, completion_tokens, cost) "
                    "VALUES (%s, %s, %s, %s, %s)", (stage, model, prompt_tokens, completion_tokens, cost))


def load_corpus(conn, doc_ids=None):
    """-> (children, parents) dicts for the retrieval collection; children inherit chapter/article from parent."""
    if doc_ids is None:
        psql = "SELECT * FROM parents ORDER BY doc_id, ord"
        csql = ("SELECT c.*, p.chapter, p.article FROM children c JOIN parents p USING (parent_id) "
                "ORDER BY c.doc_id, c.ord")
        params = ()
    else:
        psql = "SELECT * FROM parents WHERE doc_id = ANY(%s) ORDER BY doc_id, ord"
        csql = ("SELECT c.*, p.chapter, p.article FROM children c JOIN parents p USING (parent_id) "
                "WHERE c.doc_id = ANY(%s) ORDER BY c.doc_id, c.ord")
        params = (list(doc_ids),)
    with conn.cursor() as cur:
        cur.execute(psql, params)
        parents = cur.fetchall()
        cur.execute(csql, params)
        children = cur.fetchall()
    return children, parents
