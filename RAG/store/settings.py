"""Store settings from env. Defaults are LOCAL DEV ONLY (127.0.0.1, throwaway creds): override in any shared env."""
import os
from dataclasses import dataclass, fields

_SECRET = ("password", "secret", "api_key", "access_key")


@dataclass(frozen=True)
class StoreSettings:
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_db: str = "rag"
    pg_user: str = "rag"
    pg_password: str = "rag"
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = "ragdev"
    minio_secret_key: str = "ragdev-secret"
    minio_bucket: str = "rag"
    minio_secure: bool = False
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_children"

    @classmethod
    def from_env(cls, env=None):
        e = os.environ if env is None else env
        d = cls()
        g = lambda k, default: e[k] if e.get(k, "") != "" else default
        return cls(
            pg_host=g("RAG_PG_HOST", d.pg_host), pg_port=int(g("RAG_PG_PORT", d.pg_port)),
            pg_db=g("RAG_PG_DB", d.pg_db), pg_user=g("RAG_PG_USER", d.pg_user),
            pg_password=g("RAG_PG_PASSWORD", d.pg_password),
            minio_endpoint=g("RAG_MINIO_ENDPOINT", d.minio_endpoint),
            minio_access_key=g("RAG_MINIO_ACCESS_KEY", d.minio_access_key),
            minio_secret_key=g("RAG_MINIO_SECRET_KEY", d.minio_secret_key),
            minio_bucket=g("RAG_MINIO_BUCKET", d.minio_bucket),
            minio_secure=str(g("RAG_MINIO_SECURE", "false")).strip().lower() in ("1", "true", "yes", "on"),
            qdrant_url=g("RAG_QDRANT_URL", d.qdrant_url), qdrant_api_key=g("RAG_QDRANT_API_KEY", ""),
            qdrant_collection=g("RAG_QDRANT_COLLECTION", d.qdrant_collection),
        )

    def redacted(self):
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = ("***" if v else "") if any(s in f.name for s in _SECRET) else v
        return out

    def __repr__(self):
        return f"StoreSettings({self.redacted()})"
