"""Engine router: out-of-process OCR engine workers (glm, docling) used to enhance parsed blocks.

Each engine is a separate worker process (engines/<name>/worker.py, JSON lines over stdin/stdout), lazily started,
kept warm for one enhance() run and closed before embedding. Optional: missing python/weights -> state 'skipped'.
Blocks may be ContentBlock dataclasses or plain dicts (same keys). Text-layer text is never overwritten by GLM text.
Test hook: python_paths values may be a python path or a (python, worker_script) pair.
"""
from __future__ import annotations

import atexit
import hashlib
import io
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ENGINES_DIR = Path(__file__).resolve().parent.parent / "engines"
DEV_ROOT = ENGINES_DIR.parent.parent / "RAG_lab"
ENGINE_VERSION = {"glm": "zai-org/GLM-OCR@v1", "docling": "2.151.1"}
DEV_VENV = {"glm": ".venv-glmocr", "docling": ".venv-docling"}
# load_* = wall of spawn+import+weights (measured 10.8 s glm, 6.6 s docling); docling = s/page incl. lazy first-call init
SEED = {"glm_formula": 0.6, "glm_table": 8.0, "docling": 6.0, "load_glm": 11.0, "load_docling": 7.0}
TARGET_TOTAL_S = 120.0  # policy: ingest <= 90-120 s per document (used by ingest.make_router; class default kept at 110)
MIN_EFFECTIVE_S = 15.0
CLOSE_RESERVE_S = 1.5  # worker shutdown is inside used_s: keep room for it
OVERRUN_S = 5.0  # hard deadline = effective_s + OVERRUN_S, enforced on every request timeout
DOCLING_MAX_SHARE = 0.4  # skip docling when its expected cost exceeds this share of the remaining budget
PLAIN_PUNCT = set(".,;:()[]{}'\"!?")  # anything else (operators, sums, roots, symbols, PUA) makes a crop non-trivial
READY_TIMEOUT_S = 120.0
MIN_REQ_TIMEOUT_S = 10.0
MAX_IMAGE_TABLES = 3
MIN_TABLE_REMAINING_S = 25.0  # an image-table request (up to ~15 s measured) is not even sent with less budget left
_LIVE: set = set()
TRACE = bool(os.environ.get("RAG_ENGINE_TRACE"))


def _trace(msg):
    if TRACE:
        print(f"[trace {time.time() % 10000:8.2f}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- pure functions
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def group_pages(pages) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for p in sorted(set(pages)):
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [(a, b) for a, b in out]


def suspect_table(meta: dict) -> bool:
    return meta.get("none_ratio", 0.0) >= 0.04 or meta.get("fill", 1.0) < 0.8


def worth_docling(meta: dict) -> bool:
    """suspect_table, minus tiny tables whose only 'defect' is 1-2 blank cells (legit blanks, not a broken extraction)."""
    if not suspect_table(meta):
        return False
    empties = meta.get("none_ratio", 0.0) * meta.get("n_rows", 0) * meta.get("n_cols", 0)
    return meta.get("fill", 1.0) < 0.8 or empties >= 2.5 or "n_rows" not in meta

class _Html(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cell, self.span = [], None, 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th") and self.rows:
            self.cell = []
            m = re.fullmatch(r"[0-9]{1,2}", str(dict(attrs).get("colspan") or ""))  # ASCII only, model output is untrusted
            self.span = min(int(m[0]), 50) if m else 1

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.rows[-1] += [" ".join("".join(self.cell).split()).replace("|", "/")] + [""] * (self.span - 1)
            self.cell = None


def table_to_markdown(s: str) -> str:
    """GLM 'Table Recognition' answers with an HTML table (measured on slide p27/p85/p96); other text is returned unchanged."""
    if "<table" not in (s or "").lower():
        return s or ""
    p = _Html()
    try:
        p.feed(s)
        p.close()
    except Exception:  # malformed model output: treated as a rejected table
        return ""
    rows = [r for r in p.rows if r]
    if not rows:
        return ""
    n = max(map(len, rows))
    rows = [r + [""] * (n - len(r)) for r in rows]
    fmt = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([fmt(rows[0]), fmt(["---"] * n)] + [fmt(r) for r in rows[1:]])


def plausible_table(md: str, ocr_text: str = "") -> tuple[bool, str]:
    """>= 2 rows and >= 2 cols of markdown; when the figure has OCR text, >= 30% of its words/numbers must appear in the table."""
    rows = [r for r in (md or "").splitlines() if r.strip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", r)]
    if len(rows) < 2 or min(len(r.strip().strip("|").split("|")) for r in rows) < 2:
        return False, "shape"
    ocr = set(_words(ocr_text))
    if ocr and len(ocr & set(_words(md))) < 0.3 * len(ocr):
        return False, "ocr mismatch"
    return True, ""


def _alnum(s: str) -> Counter:
    return Counter(c for c in s.casefold() if c.isalnum())


def strip_math_delims(s: str) -> str:
    s = (s or "").strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):-len(b)].strip()
    return s


def plausible_latex(latex: str, unicode_text: str = "") -> tuple[bool, str]:
    if not latex or not latex.strip():
        return False, "empty"
    if len(latex) > 600:
        return False, "too long"
    if "<|" in latex or any(ord(c) < 32 and c not in "\n\t\r" for c in latex):
        return False, "junk"
    depth, i = 0, 0
    while i < len(latex):
        c = latex[i]
        if c == "\\":
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        if depth < 0:
            break
        i += 1
    if depth != 0:
        return False, "unbalanced braces"
    if len(re.findall(r"\\left(?![a-zA-Z])", latex)) != len(re.findall(r"\\right(?![a-zA-Z])", latex)):
        return False, "unbalanced left/right"
    run, prev = 0, None
    for tok in re.findall(r"\\[a-zA-Z]+|\S", latex):
        run = run + 1 if tok == prev else 1
        prev = tok
        if run >= 8:
            return False, "loop"
    if re.search(r"(.{6,40}?)\1{3,}", latex, re.S):
        return False, "loop"
    uni = _alnum(unicode_text)
    n = sum(uni.values())
    if n >= 4 and sum((uni & _alnum(latex)).values()) < 0.5 * n:
        return False, "unicode mismatch"
    return True, ""


# ---------------------------------------------------------------- block helpers (dataclass or dict)
def _get(b, k, default=None):
    return b.get(k, default) if isinstance(b, dict) else getattr(b, k, default)


def _set(b, k, v):
    if isinstance(b, dict):
        b[k] = v
    else:
        setattr(b, k, v)


def _meta(b) -> dict:
    m = _get(b, "metadata")
    if m is None:
        m = {}
        _set(b, "metadata", m)
    return m


def _page(b):
    return _meta(b).get("page") or (_get(b, "location") or {}).get("page_num")


def _asset_path(b, asset_dir) -> Path | None:
    a = _get(b, "asset")
    if not a:
        return None
    cands = [Path(a)] if Path(a).is_absolute() else [Path(asset_dir) / a, Path(asset_dir).parent / a]  # never relative to CWD
    for c in cands + [Path(asset_dir) / Path(a).name]:
        if c.is_file():
            return c.resolve()
    return None


def _png_pt(path: Path, dpi=200.0):
    try:
        with open(path, "rb") as f:
            h = f.read(24)
        if h[:8] == b"\x89PNG\r\n\x1a\n":
            w, hh = struct.unpack(">II", h[16:24])
            return w * 72 / dpi, hh * 72 / dpi
    except OSError:
        pass
    return None


def _ugly(text: str) -> bool:
    return (any(c == "\ufffd" or "\ue000" <= c <= "\uf8ff" for c in text)
            or sum(c.isalnum() for c in text) < 3)


def _trivial(text: str) -> bool:
    """Short plain unicode (<= 3 alnum, no operator/fraction/sum glyph, no garbage chars): stays unicode, not worth a GLM call."""
    return bool(text.strip()) and sum(c.isalnum() for c in text) <= 3 and all(c.isalnum() or c.isspace() or c in PLAIN_PUNCT for c in text)


def _is_numeric_table(b) -> bool:
    m = _meta(b)
    if m.get("approximate_table"):
        return True
    if _get(b, "type") != "figure" or m.get("kind") != "raster":
        return False
    rows = 0
    for line in (m.get("ocr_text") or "").splitlines():
        if sum(bool(re.fullmatch(r"[-+(]?\d[\d.,%)]*", t)) for t in line.split()) >= 2:
            rows += 1
    return rows >= 3


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.casefold())


# ---------------------------------------------------------------- worker process
class _Fatal(Exception):
    pass


class _Soft(Exception):
    pass


class _Worker:
    def __init__(self, name, py, script):
        self.name, self.py, self.script = name, str(py), str(script)
        self.proc = None
        self.q: queue.Queue = queue.Queue()
        self.log = None
        self.nid = 0
        self.ready: dict = {}

    def start(self, timeout: float) -> dict:
        env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1")
        self.log = tempfile.TemporaryFile(prefix=f"engine_{self.name}_", suffix=".log")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen([self.py, "-X", "utf8", self.script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log, bufsize=0, env=env, creationflags=flags)
        except OSError:
            self.log.close()
            self.log = None
            _LIVE.discard(self)
            raise
        self.proc.stdout = io.BufferedReader(self.proc.stdout)  # stdin stays unbuffered; big replies are not read byte-by-byte
        _LIVE.add(self)
        threading.Thread(target=self._read, args=(self.proc, self.q), daemon=True).start()
        try:
            self.ready = self._recv(timeout)
        except Exception:
            self.kill()
            raise
        if not self.ready.get("ready"):
            self.kill()
        return self.ready

    @staticmethod
    def _read(proc, q):
        try:
            for line in iter(proc.stdout.readline, b""):
                q.put(line)
        except (OSError, ValueError):
            pass
        q.put(None)

    def _recv(self, timeout: float) -> dict:
        try:
            line = self.q.get(timeout=max(0.01, timeout))
        except queue.Empty:
            raise _Fatal("timeout")
        if line is None:
            raise _Fatal("worker exited")
        try:
            obj = json.loads(line.decode("utf-8"))
        except ValueError:
            raise _Fatal("bad json")
        if not isinstance(obj, dict):
            raise _Fatal("bad json")
        return obj

    def request(self, op: str, timeout: float, **kw):
        self.nid += 1
        msg = dict(kw, id=self.nid, op=op)
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        except OSError:
            raise _Fatal("worker exited")
        try:
            r = self._recv(timeout)
        except _Fatal:
            self.kill()
            raise
        if r.get("id") != self.nid:
            self.kill()
            raise _Fatal("bad reply id")
        if not r.get("ok"):
            raise _Soft(str(r.get("error", ""))[:200])
        return r.get("result")

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self._finish()

    def _finish(self):
        if self.proc:
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except OSError:
                    pass
        if self.log:
            self.log.close()
            self.log = None
        _LIVE.discard(self)

    def close(self):
        _trace(f"close {self.name} begin")
        # worst-case overshoot is bounded: 2 s graceful wait + 2 s after kill (_finish)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"id":0,"op":"shutdown"}\n')
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.kill()


@atexit.register
def _kill_all():
    for w in list(_LIVE):
        w.kill()


# ---------------------------------------------------------------- router
class EngineRouter:
    def __init__(self, mode="auto", budget_s=45.0, cache_dir=None, python_paths=None, target_total_s=110.0):
        self.mode = (mode or "off").strip().lower()
        if not (budget_s > 0):  # NaN / negative / 0 -> 0.0: engines are skipped (effective budget < MIN_EFFECTIVE_S)
            budget_s = 0.0
        self.budget_s, self.target_total_s = budget_s, target_total_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.python_paths = python_paths
        self.workers: dict[str, _Worker] = {}
        self.doc_start: float | None = None
        self.ema = dict(SEED)
        if self.cache_dir:
            try:
                d = json.loads((self.cache_dir / "engine_stats.json").read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.ema.update({k: float(v) for k, v in d.items() if k in SEED and isinstance(v, (int, float))
                                     and not isinstance(v, bool) and math.isfinite(v) and 0 < v < 1e4})
            except Exception:  # corrupt stats file: keep seeds
                pass

    @classmethod
    def from_env(cls, mode="auto", budget_s=45.0, cache_dir=None):
        return cls(mode=mode, budget_s=budget_s, cache_dir=cache_dir)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for w in list(self.workers.values()):
            w.close()
        self.workers.clear()

    # ---- helpers
    def _wanted(self):
        m = self.mode
        if m in ("off", "", "none"):
            return []
        if m == "auto":
            return ["glm"]  # docling is opt-in (--engines docling | glm,docling): benefit unproven
        return [e for e in ("docling", "glm") if e in m.replace(" ", "").split(",")]

    def _locate(self, name):
        if self.python_paths is not None:
            v = self.python_paths.get(name)
            v = tuple(v) if isinstance(v, (tuple, list)) else (v, ENGINES_DIR / name / "worker.py")
            return v if v[0] and Path(v[0]).is_file() else None
        cands = [os.environ.get(f"RAG_ENGINE_{name.upper()}_PYTHON"),
                 ENGINES_DIR / name / ".venv" / "Scripts" / "python.exe",
                 DEV_ROOT / DEV_VENV[name] / "Scripts" / "python.exe"]
        for c in cands:
            if c and Path(c).is_file():
                return c, ENGINES_DIR / name / "worker.py"
        return None

    def _key(self, engine, op, params, data: bytes):
        return hashlib.sha256(f"{op}|{engine}|{ENGINE_VERSION[engine]}|{json.dumps(params, sort_keys=True)}".encode() + data).hexdigest()

    def _cget(self, engine, key):
        if not self.cache_dir:
            return None
        try:
            return json.loads((self.cache_dir / engine / key[:2] / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _cput(self, engine, key, val):
        if not self.cache_dir:
            return
        p = self.cache_dir / engine / key[:2] / f"{key}.json"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _start(self, name, st, remaining) -> _Worker | None:
        loc = self._locate(name)
        if not loc:
            st.update(state="skipped", reason=f"python not found (set RAG_ENGINE_{name.upper()}_PYTHON)")
            print(f"[engine {name}] skipped: {st['reason']}", file=sys.stderr)
            return None
        w = _Worker(name, *loc)
        t = time.perf_counter()
        _trace(f"spawn {name}")
        try:
            r = w.start(min(READY_TIMEOUT_S, remaining + OVERRUN_S))
            _trace(f"ready {name} wall={time.perf_counter() - t:.1f} worker_load={r.get('load_s')}")
        except (_Fatal, OSError) as e:
            st.update(state="failed", reason=f"start: {e}")
            return None
        if not r.get("ready"):
            st.update(state="skipped", reason=f"not ready: {r.get('error', '')}"[:200])
            print(f"[engine {name}] skipped: {st['reason']}", file=sys.stderr)
            return None
        st["load_s"] = r.get("load_s", time.perf_counter() - t)
        self.ema[f"load_{name}"] = 0.7 * self.ema[f"load_{name}"] + 0.3 * (time.perf_counter() - t)
        self.workers[name] = w
        return w

    def _req(self, w, st, t0, eff, ekey, units, op, timeout, **kw):
        """-> result or None (budget exhausted, soft error or worker dead; caller counts skipped)."""
        if w.proc is None or w.proc.poll() is not None or st["state"] == "failed":
            return None
        now = time.perf_counter()
        if now - t0 + 1.5 * self.ema[ekey] * units > eff - CLOSE_RESERVE_S:
            return None
        t = now
        timeout = max(0.5, min(timeout, t0 + eff + OVERRUN_S - CLOSE_RESERVE_S - now))  # hard deadline
        try:
            res = w.request(op, timeout, **kw)
        except _Soft:
            return None
        except _Fatal as e:
            st.update(state="failed", reason=str(e))
            return None
        _trace(f"req {op} {time.perf_counter() - t:.2f}s")
        self.ema[ekey] = 0.7 * self.ema[ekey] + 0.3 * (time.perf_counter() - t) / units
        return res

    # ---- main
    def enhance(self, blocks, pdf_path, asset_dir, *, elapsed_s=0.0, n_pages=0) -> dict:
        t0 = time.perf_counter()
        if self.doc_start is not None:
            elapsed_s = max(elapsed_s, t0 - self.doc_start)
        _trace(f"enhance begin elapsed_s={elapsed_s:.1f} pages={n_pages}")
        wanted = self._wanted()
        stats = {"mode": self.mode, "budget_s": self.budget_s, "effective_s": 0.0, "used_s": 0.0,
                 "engines": {n: {"state": "off", "reason": ""} for n in ("glm", "docling")}, "new_blocks": [], "signature": ""}
        if not wanted:
            return stats
        eff = min(self.budget_s, self.target_total_s - elapsed_s - 0.28 * n_pages)
        stats["effective_s"] = eff
        for n in wanted:
            stats["engines"][n] = {"state": "ok", "reason": "", "s": 0.0}
        if self.mode == "auto":
            stats["engines"]["docling"]["reason"] = "opt-in"
        if eff < MIN_EFFECTIVE_S:
            for n in wanted:
                stats["engines"][n].update(state="skipped", reason=f"budget {eff:.1f}s < {MIN_EFFECTIVE_S:.0f}s")
            return stats
        if "glm" in wanted and "docling" in wanted and len(self._formula_cands(blocks, asset_dir)) < 3:
            wanted = ["docling", "glm"]  # value order: glm first only with >= 3 worthwhile formulas, else docling's tables
        elif "glm" in wanted and "docling" in wanted:
            wanted = ["glm", "docling"]
        try:
            for n in wanted:
                st = stats["engines"][n]
                ts = time.perf_counter()
                try:
                    (self._docling if n == "docling" else self._glm)(blocks, pdf_path, asset_dir, st, stats, t0, eff)
                finally:
                    if n in self.workers:
                        self.workers.pop(n).close()  # sequential phases: free VRAM before the next engine
                st["s"] = round(time.perf_counter() - ts, 2)
        finally:
            self.close()
            self._save_ema()
        stats["used_s"] = round(time.perf_counter() - t0, 2)
        _trace(f"enhance end used_s={stats['used_s']}")
        stats["signature"] = ";".join(f"{n}={ENGINE_VERSION[n]}" for n in wanted if stats["engines"][n].get("used"))
        for n in wanted:
            stats["engines"][n].pop("used", None)
        return stats

    def _save_ema(self):
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_dir / f"engine_stats.{os.getpid()}.tmp"
                tmp.write_text(json.dumps(self.ema), encoding="utf-8")
                os.replace(tmp, self.cache_dir / "engine_stats.json")
            except OSError:
                pass

    # ---- docling: digital-PDF tables
    def _docling(self, blocks, pdf_path, asset_dir, st, stats, t0, eff):
        st.update(pages=0, tables_replaced=0)
        sus = [b for b in blocks if _get(b, "type") == "table" and worth_docling(_meta(b)) and _page(b)]
        for b in sus:
            m = _meta(b)
            _trace(f"suspect table p{_page(b)} none_ratio={m.get('none_ratio')} fill={m.get('fill')}")
        if not sus:
            st.update(state="skipped", reason="no suspect tables")
            return
        sha = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
        tables: list[dict] = []
        w = None
        for first, last in group_pages({_page(b) for b in sus}):
            n = last - first + 1
            key = hashlib.sha256(f"{sha}|{first}|{last}|{ENGINE_VERSION['docling']}".encode()).hexdigest()
            res = self._cget("docling", key)
            if res is None:
                if w is None and "docling" not in self.workers and st["state"] == "ok":
                    rem = eff - (time.perf_counter() - t0)
                    est = self.ema["load_docling"] + self.ema["docling"] * n
                    if rem < est + CLOSE_RESERVE_S or est > DOCLING_MAX_SHARE * rem:
                        st.update(state="skipped", reason=f"budget (expected {est:.0f}s of {rem:.0f}s left)")
                        continue
                    w = self._start("docling", st, rem)
                if w is None:
                    continue
                res = self._req(w, st, t0, eff, "docling", n, "tables", max(MIN_REQ_TIMEOUT_S, 5 * n + 10), pdf=str(Path(pdf_path).resolve()), first=first, last=last)
                if res is None:
                    continue
                self._cput("docling", key, res)
            st["pages"] += n
            st["used"] = True
            tables += res
        for b in sus:
            if self._replace_table(b, tables):
                st["tables_replaced"] += 1

    @staticmethod
    def _replace_table(b, tables) -> bool:
        pg, bb, m = _page(b), _get(b, "bbox"), _meta(b)
        if not bb:
            return False
        orig = set(_words(_get(b, "text") or ""))
        for t in tables:
            if t.get("page") != pg or t.get("n_rows", 0) < 2 or t.get("n_cols", 0) < 2:
                continue
            l, bt, r, tp = t["bbox"]
            w, h = t["page_w"], t["page_h"]
            if iou(bb, [l / w, 1 - tp / h, r / w, 1 - bt / h]) < 0.5:
                continue
            cw = [x for row in t.get("rows", []) for c in row for x in _words(c)]
            if not cw or sum(x in orig for x in cw) < 0.9 * len(cw):
                continue
            old = m.get("rows")
            if old is not None and len(json.dumps(old, ensure_ascii=False)) <= 4000:
                m["parser_rows"] = old
            _set(b, "text", t["markdown"])
            m.update(rows=t["rows"], n_rows=t["n_rows"], n_cols=t["n_cols"], engine="docling")
            return True
        return False

    # ---- glm: formulas and image tables
    @staticmethod
    def _formula_cands(blocks, asset_dir):
        """Worthwhile formula crops, most valuable first (ugly unicode, then long/multi-line, then page order); [] if < 3."""
        forms = []
        for b in blocks:
            if _get(b, "type") == "formula" and (p := _asset_path(b, asset_dir)):
                sz, text = _png_pt(p), _get(b, "text") or ""
                if (sz and min(sz) < 12) or _trivial(text):
                    continue
                forms.append((b, p))

        def key(bp):
            t = _get(bp[0], "text") or ""
            return (not _ugly(t), not (len(t) >= 25 or "\n" in t), _page(bp[0]) or 0)
        forms.sort(key=key)
        return forms if len(forms) >= 3 else []

    def _glm(self, blocks, pdf_path, asset_dir, st, stats, t0, eff):
        st.update(formula={"done": 0, "cached": 0, "rejected": 0, "skipped": 0}, table={"done": 0, "cached": 0, "skipped": 0})
        fs, ts = st["formula"], st["table"]
        forms = self._formula_cands(blocks, asset_dir)
        tabs = [(b, p) for b in blocks if _is_numeric_table(b) and (p := _asset_path(b, asset_dir))][:MAX_IMAGE_TABLES]
        if not forms and not tabs:
            st.update(state="skipped", reason="no candidates (need >= 3 formulas or an image table)")
            return
        jobs = [("formula", b, p) for b, p in forms] + [("table", b, p) for b, p in tabs]
        w, pending = None, []
        for op, b, p in jobs:  # cache first: all-cached never starts a worker
            data = p.read_bytes()
            key = self._key("glm", op, {"op": op, "max_new": 256 if op == "formula" else 1024}, data)
            res = self._cget("glm", key)
            if res is None:
                pending.append((op, b, p, key))
            else:
                self._apply_glm(op, b, res, st, "cached", stats)
        nf = sum(op == "formula" for op, *_ in pending)
        if pending:
            unit = self.ema["glm_formula"] if nf >= 3 else self.ema["glm_table"]
            rem = eff - (time.perf_counter() - t0)
            if rem < self.ema["load_glm"] + 3 * unit + CLOSE_RESERVE_S or (nf == 0 and rem < MIN_TABLE_REMAINING_S + self.ema["load_glm"]):
                st.update(state="skipped" if not st.get("used") else st["state"], reason="budget")
            else:
                w = self._start("glm", st, rem)
        nb = max(1, int(w.ready.get("max_batch", 1))) if w is not None else 1
        units, fl = [], [x for x in pending if x[0] == "formula"]
        units += [fl[i:i + nb] for i in range(0, len(fl), nb)] + [[x] for x in pending if x[0] != "formula"]
        for unit in units:
            op, n = unit[0][0], len(unit)
            ekey = "glm_formula" if op == "formula" else "glm_table"
            res = None
            if w is not None and (op == "formula" or eff - (time.perf_counter() - t0) >= MIN_TABLE_REMAINING_S):
                if n > 1:
                    res = self._req(w, st, t0, eff, ekey, n, "formulas", max(MIN_REQ_TIMEOUT_S, 6 * self.ema[ekey] * n), images=[str(p) for _, _, p, _ in unit])
                    res = res and res.get("latex")
                    res = [{"latex": x} for x in res] if isinstance(res, list) and len(res) == n else None
                else:
                    res = self._req(w, st, t0, eff, ekey, 1, op, max(MIN_REQ_TIMEOUT_S, 6 * self.ema[ekey]) if op == "formula" else 40, image=str(unit[0][2]))
                    res = [res] if res is not None else None
            if res is None:
                (fs if op == "formula" else ts)["skipped"] += n
                continue
            for (o, b, p, key), r in zip(unit, res):
                self._cput("glm", key, r)
                self._apply_glm(o, b, r, st, "done", stats)

    def _apply_glm(self, op, b, res, st, how, stats):
        if op == "formula":
            fs, m = st["formula"], _meta(b)
            latex, uni = strip_math_delims((res or {}).get("latex", "")), _get(b, "text") or ""
            ok, why = plausible_latex(latex, uni)
            if not ok:
                fs["rejected"] += 1
                m["latex_flags"] = [f"rejected:{why}"]
                return
            m.update(latex=latex, engine="glm")
            m.setdefault("format", "text")
            if sum(c.isalnum() for c in uni) < 3:
                _set(b, "text", latex)
            fs[how] += 1
        else:
            md = table_to_markdown((res or {}).get("markdown", ""))
            ok, why = plausible_table(md, _meta(b).get("ocr_text") or "")
            _trace(f"image table {_get(b, 'id')}: {'accepted' if ok else 'rejected ' + why} {md[:160]!r}")
            if not ok:
                st["table"]["skipped"] += 1
                st["table"]["rejected"] = st["table"].get("rejected", 0) + 1
                return
            st["table"][how] += 1
            meta = dict(engine="glm", text_source="glm", diacritics_risk=True)
            loc = dict(_get(b, "location") or {})
            bid = f"{_get(b, 'id')}_glmtbl"
            if isinstance(b, dict):
                nb = {"id": bid, "type": "table", "text": md, "location": loc, "metadata": meta, "bbox": _get(b, "bbox"),
                      "asset": None, "after_id": _get(b, "id")}
            else:
                nb = type(b)(id=bid, type="table", text=md, location=loc, bbox=_get(b, "bbox"), metadata=meta)
                nb.after_id = _get(b, "id")
            stats["new_blocks"].append(nb)
        st["used"] = True
