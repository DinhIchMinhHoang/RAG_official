"""MinIO object layer. Keys are sanitised: no '..', no leading '/', NFC unicode, length-capped."""
import io
import re
import unicodedata
from datetime import timedelta

from minio import Minio

MAX_SEG = 120


def _seg(s):
    s = unicodedata.normalize("NFC", str(s)).replace("\\", "/").split("/")[-1]
    s = re.sub(r"[\x00-\x1f]", "", s).replace("..", "_").strip(" .") or "_"
    if len(s) > MAX_SEG:
        stem, dot, ext = s.rpartition(".")
        ext = ("." + ext)[:12] if dot else ""
        s = s[:MAX_SEG - len(ext)] + ext
    return s


def safe_key(key):
    k = unicodedata.normalize("NFC", str(key)).replace("\\", "/")
    return "/".join(_seg(p) for p in k.split("/") if p not in ("", ".", ".."))


def raw_key(sha256, filename):
    return safe_key(f"raw/{_seg(sha256)}/{_seg(filename)}")


def page_key(doc_id, page):
    return safe_key(f"pages/{_seg(doc_id)}/{int(page):05d}.png")


def blocks_key(doc_id):
    return safe_key(f"blocks/{_seg(doc_id)}.json")


def client(s):
    return Minio(s.minio_endpoint, access_key=s.minio_access_key, secret_key=s.minio_secret_key, secure=s.minio_secure)


def ensure_bucket(c, bucket):
    """-> True if created."""
    if c.bucket_exists(bucket):
        return False
    c.make_bucket(bucket)
    return True


def put_file(c, bucket, local_path, key, content_type=None):
    c.fput_object(bucket, safe_key(key), str(local_path), content_type=content_type or "application/octet-stream")


def put_bytes(c, bucket, data, key, content_type=None):
    c.put_object(bucket, safe_key(key), io.BytesIO(data), len(data),
                 content_type=content_type or "application/octet-stream")


def get_to_file(c, bucket, key, local_path):
    c.fget_object(bucket, safe_key(key), str(local_path))


def presigned_get(c, bucket, key, expires_s=3600):
    return c.presigned_get_object(bucket, safe_key(key), expires=timedelta(seconds=expires_s))


def delete_prefix(c, bucket, prefix):
    prefix = safe_key(prefix)
    if not prefix:
        raise ValueError("empty prefix refused")
    names = [o.object_name for o in c.list_objects(bucket, prefix=prefix.rstrip("/") + "/", recursive=True)]
    for n in names:
        c.remove_object(bucket, n)
    return len(names)
