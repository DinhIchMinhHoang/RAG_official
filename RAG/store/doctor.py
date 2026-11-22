"""python -m store.doctor [--init]: check Postgres, MinIO, Qdrant. Never prints secrets."""
import argparse
import sys
import urllib.request

from . import objects, pg
from .settings import StoreSettings


def _pg(s, init):
    with pg.connect(s, connect_timeout=2) as conn:
        if init:
            pg.init_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT version() AS v")
            v = cur.fetchone()["v"]
    return v.split(",")[0] + (" (schema initialised)" if init else "")


def _minio(s, init):
    import urllib3
    c = objects.client(s)
    c._http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=2, read=2), retries=False)
    if init:
        return "bucket created" if objects.ensure_bucket(c, s.minio_bucket) else "bucket exists"
    if not c.bucket_exists(s.minio_bucket):
        raise RuntimeError(f"bucket '{s.minio_bucket}' missing (run --init)")
    return f"bucket '{s.minio_bucket}' exists"


def _qdrant(s, init):
    req = urllib.request.Request(s.qdrant_url.rstrip("/") + "/collections")
    if s.qdrant_api_key:
        req.add_header("api-key", s.qdrant_api_key)
    with urllib.request.urlopen(req, timeout=2) as r:
        return f"HTTP {r.status}"


def _reason(e):
    # keep first line only; connection errors from libs do not include credentials
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0][:120] if lines else 'unreachable'}"


def run(s, init=False, out=print):
    ok = True
    for name, fn in (("postgres", _pg), ("minio", _minio), ("qdrant", _qdrant)):
        try:
            out(f"OK   {name}: {fn(s, init)}")
        except Exception as e:
            ok = False
            out(f"FAIL {name}: {_reason(e)}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="store.doctor")
    ap.add_argument("--init", action="store_true", help="create schema and bucket")
    return run(StoreSettings.from_env(), ap.parse_args(argv).init)


if __name__ == "__main__":
    sys.exit(main())
