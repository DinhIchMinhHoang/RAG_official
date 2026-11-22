"""Chunker theo khoi, khong phu thuoc Dieu/Khoan: JSON khoi cua parse_document -> parents/children.

Chay (cwd RAG):  python -m parser.chunker <blocks.json> --doc-id ID --out-dir DIR [--ceiling N]
"""
import argparse
import functools
import json
import logging
import re
import statistics
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

ATOMIC = {"table", "code", "formula"}
CHAPTER_RE = re.compile(r"^(?:Chương|CHƯƠNG)\s+([IVXLC]+)\b\s*[.:\-]?\s*(.*)$")
ARTICLE_RE = re.compile(r"^(?:Điều|ĐIỀU)\s+(\d+)\s*[.:]\s*(.*)$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
ROMAN_RE = re.compile(r"^[IVX]+\.\s+\S")
SENT_RE = re.compile(r"(?<=[.!?;:])\s+|\n")
SKIP_SECTIONS = (r"^(Chú thích|Ghi chú|Tham khảo|Nguồn tham khảo|Liên kết ngoài|Xem thêm|Thư mục|References|Notes|"
                 r"External links|See also|Bibliography|Footnotes)$",)
BACKREF_RE = re.compile(r"^[-*]?\s*\[[↑^]\]\(#cite_ref[^)]*\)\s*$")
WIKI_LINK_RE = re.compile(r"^[-*]\s*\[[^\]]*\]\(\s*(?:https?:)?//[^/\s)]*wikipedia\.org[^)]*\)\s*$")
_stats = {}
PARENT_MAX, PARENT_WINDOW = 2000, 1500  # section dai hon PARENT_MAX bi cat thanh cua so ~PARENT_WINDOW


class ChunkingError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tok.model_max_length = 10 ** 9  # chi dem, khong cat
    return tok


def default_token_counter():
    tok = _tokenizer()
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def last_stats():
    """Thong ke lan chunk_document gan nhat: skipped_sections, skipped_blocks, noise_blocks."""
    return dict(_stats)


def _fold(s):
    s = unicodedata.normalize("NFD", s or "").replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _is_noise(b):
    """Khoi chi gom link wikipedia (lien ngon ngu) hoac back-ref [↑](#cite_ref-..); khong bao gio la bang/code."""
    if b.get("type") in ATOMIC:
        return False
    lines = [x.strip() for x in (b.get("text") or "").split("\n") if x.strip()]
    hit = sum(1 for x in lines if BACKREF_RE.match(x) or WIKI_LINK_RE.match(x))
    return bool(lines) and hit >= 0.7 * len(lines)


def _split_heading_blocks(blocks):
    """Khoi heading nhieu dong (OCR gop dong): tach dong tieu de / Dieu-Chuong nhung / doan van."""
    out = []
    for b in blocks:
        lines = [x.strip() for x in (b.get("text") or "").strip().split("\n") if x.strip()]
        if b.get("type") != "heading" or len(lines) < 2:
            out.append(b)
            continue
        head, rest = lines[0], lines[1:]
        m, r0 = CHAPTER_RE.match(_clean(head)), _clean(rest[0])
        if m and not m.group(2) and len(r0) <= 100 and r0[-1] not in ".;,:" and not (
                ARTICLE_RE.match(r0) or CHAPTER_RE.match(r0)):
            head, rest = head + " " + rest[0], rest[1:]  # ten chuong o dong ke tiep
        pieces = [("heading", [head])]
        for ln in rest:
            if ARTICLE_RE.match(_clean(ln)) or CHAPTER_RE.match(_clean(ln)):
                pieces.append(("heading", [ln]))
            elif pieces[-1][0] == "paragraph":
                pieces[-1][1].append(ln)
            else:
                pieces.append(("paragraph", [ln]))
        for k, (typ, ls) in enumerate(pieces):
            out.append({**b, "type": typ, "text": "\n".join(ls), "id": b["id"] if k == 0 else f"{b['id']}#{k}"})
    return out


def load_blocks(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["blocks"] if isinstance(d, dict) else d


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _clean(s):
    s = unicodedata.normalize("NFC", s or "").replace(" ", " ")
    s = re.sub(r"^\s*#+\s*", "", s.strip())
    s = re.sub(r"\*\*|__", "", s)
    return re.sub(r"^[-*]\s+", "", s).strip()


# ---------- cay tieu de ----------

def _legal_kind(b):
    if b.get("type") in ATOMIC:
        return None
    t = _clean(b.get("text"))
    if len(t) > 300:
        return None
    if CHAPTER_RE.match(t):
        return "chapter"
    return "article" if ARTICLE_RE.match(t) else None


def _num_level(b):
    t = _clean(b.get("text"))
    if b.get("type") in ATOMIC or not t or len(t) > 100 or t[-1] in ".;,:":
        return None
    if CHAPTER_RE.match(t) or ROMAN_RE.match(t):
        return 1
    if ARTICLE_RE.match(t):
        return 2
    m = NUM_RE.match(t)
    return m.group(1).count(".") + 1 if m else None


def _typo_levels(blocks):
    sizes = [b["metadata"]["font_size"] for b in blocks if b.get("metadata", {}).get("font_size")]
    if not sizes:
        return {}
    med = statistics.median(sizes)
    out = {}
    for i, b in enumerate(blocks):
        m, t = b.get("metadata", {}), _clean(b.get("text"))
        if not t or len(t) > 120 or b.get("type") in ATOMIC:
            continue
        if m.get("font_size", 0) >= 1.2 * med:
            out[i] = 1
        elif m.get("bold"):
            out[i] = 2
    return out


def _detect_headings(blocks, legal):
    """-> {index: (level, kind)}; kind in chapter|article|h|num."""
    typed = any(b.get("type") == "heading" and _clean(b.get("text")) for b in blocks)
    out = {}
    if typed:
        for i, b in enumerate(blocks):
            if b.get("type") == "heading" and _clean(b.get("text")):
                out[i] = (3 if legal else max(1, int(b.get("level") or 1)), "h")
    else:
        nums = {i: _num_level(b) for i, b in enumerate(blocks)}
        nums = {i: lv for i, lv in nums.items() if lv}
        if len(nums) >= 2:
            out = {i: (lv, "num") for i, lv in nums.items()}
        else:
            out = {i: (lv, "h") for i, lv in _typo_levels(blocks).items()}
    if legal:
        for i, b in enumerate(blocks):
            k = _legal_kind(b)
            if k:
                out[i] = (1 if k == "chapter" else 2, k)
    return out


def _warn_gaps(nums):
    s = sorted(set(nums))
    miss = [n for a, b in zip(s, s[1:]) for n in range(a + 1, b)]
    if miss or len(s) != len(nums):
        log.warning("Dieu numbering irregular: missing=%s duplicates=%d (nothing dropped)",
                    miss[:20], len(nums) - len(s))


def _sections(blocks, legal):
    heads = _detect_headings(blocks, legal)
    sections, stack, cur = [], [], []
    state = {"chapter": None, "chapter_title": "", "article": None}
    art_nos = []

    def emit(path, bl):
        sections.append({"path": path, "blocks": bl, **state})

    def flush():
        if cur:
            emit([f["title"] for f in stack], list(cur))
            if stack:
                stack[-1]["content"] = True
            cur.clear()

    def pop_to(level):
        while stack and stack[-1]["level"] >= level:
            f = stack[-1]
            if not f["content"] and not f["child"]:
                emit([x["title"] for x in stack], [{**f["block"], "type": "paragraph", "text": f["title"]}])
            stack.pop()

    for i, b in enumerate(blocks):
        text = _clean(b.get("text"))
        if i not in heads:
            if text:
                cur.append(b)
            continue
        text = text.split("\n")[0][:120].strip()  # thanh phan breadcrumb: chi dong dau
        level, kind = heads[i]
        flush()
        top = stack[-1] if stack else None
        if top and not top["content"] and not top["child"] and kind in ("h",) and (
                top["kind"] == "chapter" or top["level"] == level):
            top["title"] += " " + text  # tieu de nhieu dong / ten chuong
            if top["kind"] == "chapter":
                state["chapter_title"] = top["title"]
            continue
        pop_to(level)
        if stack:
            stack[-1]["child"] = True
        if kind == "chapter":
            state.update(chapter=CHAPTER_RE.match(text).group(1), chapter_title=text, article=None)
        elif kind == "article":
            state["article"] = int(ARTICLE_RE.match(text).group(1))
            art_nos.append(state["article"])
        stack.append({"level": level, "title": text, "kind": kind, "block": b, "content": False, "child": False})
    flush()
    pop_to(0)
    if legal:
        _warn_gaps(art_nos)
    return sections, {blocks[i]["id"] for i in heads}


# ---------- cat nho ----------

def _cut_words(text, limit, ct):
    words, out, i = text.split(), [], 0
    while i < len(words):
        n = min(len(words) - i, limit)
        while n > 1 and ct(" ".join(words[i:i + n])) > limit:
            n = min(n - 1, max(1, int(n * 0.9)))
        piece = " ".join(words[i:i + n])
        t = ct(piece)
        if n == 1 and t > limit:  # mot "tu" qua dai (base64...): cat theo ky tu
            size = max(1, int(len(piece) * limit / t * 0.9))
            out += [piece[k:k + size] for k in range(0, len(piece), size)]
        else:
            out.append(piece)
        i += n
    return out


def _split_text(text, limit, ct):
    """Cat 1 doan theo cau, gac cau cuoi sang manh sau."""
    sents = []
    for s in filter(None, (x.strip() for x in SENT_RE.split(text))):
        sents += [s] if ct(s) <= limit else _cut_words(s, limit, ct)
    pieces, cur, cur_t = [], [], 0
    for s in sents:
        t = ct(s)
        if cur and cur_t + t > limit:
            pieces.append(" ".join(cur))
            carry_t = ct(cur[-1])
            cur, cur_t = ([cur[-1]], carry_t) if carry_t + t <= limit else ([], 0)
        cur.append(s)
        cur_t += t
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_lines(text, ctype, limit, ct):
    """Cat khoi nguyen tu theo dong/hang; bang lap lai dong tieu de."""
    lines = text.split("\n")
    header = []
    if ctype == "table" and len(lines) > 2 and re.match(r"^\|?\s*:?-{3,}", lines[1]):
        header, lines = lines[:2], lines[2:]
        if ct("\n".join(header)) > limit // 3:
            header, lines = [], text.split("\n")
    ht = ct("\n".join(header)) if header else 0
    room = max(1, limit - ht)
    pieces, cur, cur_t = [], [], 0

    def flush():
        if cur:
            pieces.append("\n".join(header + cur))
            cur.clear()

    for ln in lines:
        for part in ([ln] if ct(ln) <= room else _cut_words(ln, room, ct)):
            t = ct(part)
            if cur and cur_t + t > room:
                flush()
                cur_t = 0
            cur.append(part)
            cur_t += t
    flush()
    return pieces


def _units(b, ceiling, ct):
    text, ctype = b["text"].strip(), b.get("type", "paragraph")
    t = ct(text)
    if t <= ceiling:
        parts = [text]
    elif ctype in ATOMIC:
        parts = _split_lines(text, ctype, ceiling, ct)
    else:
        parts = _split_text(text, ceiling, ct)
    page = (b.get("location") or {}).get("page_num")
    return [{"text": p, "tok": ct(p), "id": b["id"], "page": page, "bbox": b.get("bbox"),
             "type": ctype if ctype in ATOMIC else "text"} for p in parts]


def _pack(units, ceiling, min_tokens):
    groups, cur, cur_t = [], [], 0
    for u in units:
        if cur and cur_t + u["tok"] > ceiling:
            groups.append(cur)
            cur, cur_t = [], 0
        cur.append(u)
        cur_t += u["tok"]
    if cur:
        groups.append(cur)
    tok = lambda g: sum(u["tok"] for u in g)
    i = 0
    while i < len(groups):
        if tok(groups[i]) < min_tokens and len(groups) > 1:
            cands = [j for j in (i - 1, i + 1) if 0 <= j < len(groups) and tok(groups[j]) + tok(groups[i]) <= ceiling]
            if cands:
                j = min(cands, key=lambda j: tok(groups[j]))
                a, c = min(i, j), max(i, j)
                groups[a] = groups[a] + groups[c]
                del groups[c]
                i = a
                continue
        i += 1
    return groups


def _page_info(items):
    pages = [x for x in items if x is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def chunk_document(blocks, doc_id, count_tokens, ceiling=400, hard_max=1000, min_tokens=80,
                   profile="auto", doc_title=None, skip_sections=SKIP_SECTIONS):
    blocks = _split_heading_blocks(blocks)
    ct = functools.lru_cache(maxsize=None)(count_tokens)
    doc_title = doc_title or doc_id
    legal = profile == "legal" or (profile == "auto" and sum(
        1 for b in blocks if _legal_kind(b) == "article") >= 3)
    sections, head_ids = _sections(blocks, legal)
    content = [b for b in blocks if _clean(b.get("text")) and b["id"] not in head_ids]

    skip_rx = [re.compile(_fold(p), re.I) for p in skip_sections or ()]
    parents, children, assigned = [], [], set()
    skipped, n_skipped, noise = [], 0, 0
    for sec in sections:
        bl = [b for b in sec["blocks"] if not _is_noise(b)]
        noise += len(sec["blocks"]) - len(bl)
        assigned.update(b["id"] for b in sec["blocks"])  # nhieu + muc bi bo qua tinh la da gan
        if any(rx.match(_fold(t)) for t in sec["path"] for rx in skip_rx):
            skipped.append(sec["path"][-1])
            n_skipped += len(bl)
            continue
        if not bl:
            continue
        tot = sum(ct(b["text"].strip()) for b in bl)
        windows = [bl]
        if not sec["path"] or tot > PARENT_MAX:
            windows, cur, cur_t = [], [], 0
            for b in bl:
                t = ct(b["text"].strip())
                if cur and cur_t + t > PARENT_WINDOW:
                    windows.append(cur)
                    cur, cur_t = [], 0
                cur.append(b)
                cur_t += t
            windows.append(cur)
        title = sec["path"][-1] if sec["path"] else ""
        crumb = " > ".join([doc_title] + sec["path"])
        for w in windows:
            pid = f"{doc_id}-p{len(parents):04d}"
            ps, pe = _page_info([(b.get("location") or {}).get("page_num") for b in w])
            extra = {"chapter": sec["chapter"], "chapter_title": sec["chapter_title"],
                     "article": sec["article"]} if legal else {}
            parents.append({"parent_id": pid, "doc_id": doc_id, "title": title, "breadcrumb": crumb,
                            "text": "\n".join(([title] if title else []) + [b["text"].strip() for b in w]),
                            "page_start": ps, "page_end": pe, **extra})
            units = [u for b in w for u in _units(b, ceiling, ct)]
            for g in _pack(units, ceiling, min_tokens):
                text = "\n".join(u["text"] for u in g)
                ps, pe = _page_info([u["page"] for u in g])
                boxes = [u["bbox"] for u in g if u["bbox"] and u["page"] == ps]
                types = {u["type"] for u in g}
                ids = list(dict.fromkeys(u["id"] for u in g))
                assigned.update(ids)
                children.append({
                    "child_id": f"{doc_id}-c{len(children):04d}", "parent_id": pid, "doc_id": doc_id,
                    "breadcrumb": crumb, "text": text, "page_start": ps, "page_end": pe,
                    "bbox": [min(x[0] for x in boxes), min(x[1] for x in boxes),
                             max(x[2] for x in boxes), max(x[3] for x in boxes)] if boxes else None,
                    "block_ids": ids, "tokens": ct(text),
                    "content_type": types.pop() if len(types) == 1 else "mixed", **extra})
    _stats.clear()
    _stats.update(skipped_sections=skipped, skipped_blocks=n_skipped, noise_blocks=noise)
    if not children:
        raise ChunkingError(f"{doc_id}: 0 children from {len(blocks)} blocks")
    big = [(c["child_id"], c["tokens"]) for c in children if c["tokens"] > hard_max]
    if big:
        raise ChunkingError(f"{doc_id}: children over hard_max={hard_max}: {big[:5]}")
    cov = sum(1 for b in content if b["id"] in assigned) / max(1, len(content))
    if cov < 0.9:
        raise ChunkingError(f"{doc_id}: only {cov:.0%} of non-empty blocks assigned to a child")
    return parents, children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ceiling", type=int, default=400)
    a = ap.parse_args()
    parents, children = chunk_document(load_blocks(a.blocks), a.doc_id, default_token_counter(), ceiling=a.ceiling)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(parents, out / "parents.jsonl")
    write_jsonl(children, out / "children.jsonl")
    print(f"{len(parents)} parents, {len(children)} children -> {out}")


if __name__ == "__main__":
    main()
