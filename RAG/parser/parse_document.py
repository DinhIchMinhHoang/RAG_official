"""Unified, block-level document parser for Vietnamese RAG.

Supported: PDF, DOCX, PPTX, HTML, MD, TXT, PNG, JPG, JPEG.
Dependencies: see requirements_parser.txt.

Examples:
  python parser/parse_document.py data/Quy-che-DTDH-3626.pdf
  python parser/parse_document.py data/Quy-che-DTDH-3626.pdf --force
  python parser/parse_document.py data/sample.docx --out-dir data/processed
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

PARSER_VERSION = "3.0.0"
MIN_PDF_TEXT = 20
MAX_DRAWINGS = 5000  # per page; above this vector-figure detection is skipped
MAX_OCR_SIDE = 2400
SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".png", ".jpg", ".jpeg"}
CHAPTER = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\b\s*[.:-]?\s*(.*)$", re.I)
ARTICLE = re.compile(r"^Điều\s+(\d+)\s*[.:-]\s*(.*)$", re.I)
CLAUSE = re.compile(r"^(\d{1,2})[.)]\s+(.+)$")
POINT = re.compile(r"^([a-zđ])\)\s+(.+)$", re.I)
BULLET = re.compile(r"^[-*+]\s+(.*)$")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
PAGE_NUMBER = re.compile(r"^(?:trang\s*)?\d{1,3}(?:\s*/\s*\d{1,3})?$", re.I)


_CTRL = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u200b-\u200d\ufeff\u00ad]")
_LIGA = re.compile("[\ufb00-\ufb06]")
_TAGS = re.compile(r"</?(mark|br|sup|sub|u)\b[^>]*>|<!--.*?-->", re.I | re.S)


def clean_text(s: str) -> str:
    """Cleanup for NON-code text (never collapses newlines; never use on code)."""
    s = unicodedata.normalize("NFC", s or "")
    s = _CTRL.sub("", s)
    s = _LIGA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    s = _TAGS.sub("", s.replace("\u00a0", " "))
    s = re.sub("\u00af(\\w)", "\\1\u0304", s)
    return re.sub(r"[ \t]+", " ", s)


def clean(s: str) -> str:
    s = clean_text((s or "").replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ContentBlock:
    id: str
    type: str  # heading | paragraph | table | list | code | formula | figure | image_ocr
    text: str
    level: int = 0
    location: dict[str, int] = field(default_factory=dict)
    breadcrumb: str = ""
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    asset: str | None = None


def make_block(kind: str, text: str, *, page_num: int | None = None,
               slide_num: int | None = None, level: int = 0,
               bbox: list[float] | None = None, verbatim: bool = False,
               asset: str | None = None, **metadata: Any) -> ContentBlock:
    location = {"page_num": page_num} if page_num is not None else (
        {"slide_num": slide_num} if slide_num is not None else {})
    # code is verbatim; table cells are already cleaned cell by cell (keeps <br>).
    body = text.strip("\n").rstrip() if verbatim or kind == "table" else clean(text)
    return ContentBlock(id="", type=kind, text=body, level=min(max(level, 0), 3),
                        location=location, bbox=bbox, metadata=metadata, asset=asset)


def read_text_file(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    from charset_normalizer import from_bytes
    best = from_bytes(raw).best()
    encoding = best.encoding if best and best.encoding else "utf-8"
    try:
        return raw.decode(encoding), encoding
    except (UnicodeError, LookupError):
        return raw.decode("utf-8", errors="replace"), "utf-8 (replace)"


def detect_type(text: str, initial: str = "paragraph", level: int = 0) -> tuple[str, int, str]:
    s = clean(text)
    if not s:
        return initial, level, s
    mh = MD_HEADING.match(s)
    if mh:
        s = mh.group(2).strip()
        level = min(len(mh.group(1)), 3)
        initial = "heading"
    if CHAPTER.match(s):
        return "heading", 1, s
    if ARTICLE.match(s):
        return "heading", 2, s
    if initial == "heading":
        return initial, level or 3, s
    if CLAUSE.match(s) or POINT.match(s) or BULLET.match(s):
        return "list", 0, s
    return initial, level, s


def parse_markdown_blocks(md: str, *, page_num: int | None = None,
                          slide_num: int | None = None,
                          source: str = "text") -> list[ContentBlock]:
    """Parse headings, paragraphs, list items, tables, fenced code and $$/\\[ math in source order."""
    out: list[ContentBlock] = []
    # Fences/math are extracted from the RAW text first so cleaning never touches code.
    raw = (md or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    text_lines: list[str] = []

    def flush_text():
        if text_lines:
            out.extend(_parse_md_text("\n".join(text_lines), page_num, slide_num, source))
            text_lines.clear()

    i = 0
    while i < len(raw):
        line = raw[i]
        s = line.strip()
        fm = re.match(r"^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)", line)
        if fm:
            fence, lang = fm.group(1), fm.group(2)
            body, i = [], i + 1
            while i < len(raw) and not re.fullmatch(r"\s{0,3}" + re.escape(fence[0]) + "{%d,}\\s*" % len(fence), raw[i]):
                body.append(raw[i])
                i += 1
            i += 1  # closing fence (or EOF)
            flush_text()
            if "".join(body).strip():
                out.append(make_block("code", "\n".join(body), page_num=page_num, slide_num=slide_num,
                                      verbatim=True, language=lang, source=source))
            continue
        opener, closer = ("$$", "$$") if s.startswith("$$") else (("\\[", "\\]") if s.startswith("\\[") else (None, None))
        if opener:
            rest, j, body = s[2:], i, []
            while True:
                if closer in rest:
                    body.append(rest.split(closer, 1)[0])
                    break
                body.append(rest)
                j += 1
                if j >= len(raw):
                    body = None
                    break
                rest = raw[j].strip()
            if body is not None and "\n".join(body).strip():
                flush_text()
                out.append(make_block("formula", "\n".join(body).strip(), page_num=page_num,
                                      slide_num=slide_num, verbatim=True, format="latex", source=source))
                i = j + 1
                continue
        text_lines.append(line)
        i += 1
    flush_text()
    return out


def _parse_md_text(md: str, page_num, slide_num, source) -> list[ContentBlock]:
    lines = clean(md).splitlines()
    out: list[ContentBlock] = []
    buffer: list[str] = []
    table: list[str] = []

    def emit(kind: str, raw: str, level: int = 0):
        k, lvl, txt = detect_type(raw, kind, level)
        if txt:
            out.append(make_block(k, txt, page_num=page_num, slide_num=slide_num,
                                  level=lvl, source=source))

    def flush():
        if buffer:
            emit("paragraph", "\n".join(buffer))
            buffer.clear()
        if table:
            emit("table", "\n".join(table))
            table.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            continue
        if s.startswith("|") and "|" in s[1:]:
            if buffer:
                flush()
            table.append(s)
            continue
        if table:
            flush()
        if MD_HEADING.match(s) or CHAPTER.match(s) or ARTICLE.match(s) or CLAUSE.match(s) or POINT.match(s) or BULLET.match(s):
            flush()
            emit("paragraph", s)
        else:
            buffer.append(s)
    flush()
    return out


def image_to_pil(data: bytes | str | Path | Any):
    from PIL import Image, ImageOps
    if isinstance(data, (bytes, bytearray)):
        img = Image.open(BytesIO(data))
    elif isinstance(data, (str, Path)):
        img = Image.open(data)
    else:
        img = data
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_OCR_SIDE:
        img.thumbnail((MAX_OCR_SIDE, MAX_OCR_SIDE), Image.Resampling.LANCZOS)
    return img


OCR_ENGINE_NAME = "Tesseract"
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# OCR env vars: TESSERACT_CMD (path to tesseract binary; else PATH; else the
# default Windows install path), TESSDATA_PREFIX (folder with *.traineddata),
# OCR_LANG (tesseract languages, default "vie+eng").
class OCR:
    def __init__(self):
        import os
        import shutil

        import pytesseract

        cmd = (
            os.environ.get("TESSERACT_CMD")
            or shutil.which("tesseract")
            or (_WIN_TESSERACT if os.path.exists(_WIN_TESSERACT) else None)
        )
        if not cmd:
            raise RuntimeError(
                "Tesseract not found. Set env TESSERACT_CMD to the tesseract binary, "
                "add tesseract to PATH, or install it at " + _WIN_TESSERACT
            )
        self.engine = pytesseract
        self.engine.pytesseract.tesseract_cmd = cmd
        self.lang = os.environ.get("OCR_LANG") or "vie+eng"
        available = set(self.engine.get_languages())
        missing = [l for l in self.lang.split("+") if l not in available]
        if missing:
            raise RuntimeError(
                f"Tesseract language(s) missing: {', '.join(missing)} (available: "
                f"{', '.join(sorted(available))}). Download <lang>.traineddata (e.g. "
                "vie.traineddata) into the tessdata folder, point env TESSDATA_PREFIX "
                "at it, or set OCR_LANG to installed languages."
            )

    def run(self, data):
        from pytesseract import Output

        img = image_to_pil(data)
        w, h = img.size

        result = self.engine.image_to_data(
            img,
            lang=self.lang,
            config="--oem 1 --psm 6",
            output_type=Output.DICT
        )

        # Gom cac tu thuoc cung mot dong
        grouped = {}

        for i, word in enumerate(result["text"]):
            word = word.strip()
            conf = float(result["conf"][i])

            if not word or conf < 0:
                continue

            key = (
                result["page_num"][i],
                result["block_num"][i],
                result["par_num"][i],
                result["line_num"][i]
            )

            x = result["left"][i]
            y = result["top"][i]
            x2 = x + result["width"][i]
            y2 = y + result["height"][i]

            if key not in grouped:
                grouped[key] = {
                    "words": [],
                    "coords": [],
                    "scores": []
                }

            grouped[key]["words"].append(word)
            grouped[key]["coords"].append((x, y, x2, y2))
            grouped[key]["scores"].append(conf)

        lines = []

        for group in grouped.values():
            coords = group["coords"]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            x2s = [p[2] for p in coords]
            y2s = [p[3] for p in coords]

            lines.append({
                "text": clean(" ".join(group["words"])),
                "bbox": [
                    min(xs) / w,
                    min(ys) / h,
                    max(x2s) / w,
                    max(y2s) / h
                ],
                "confidence": sum(group["scores"]) / (
                    len(group["scores"]) * 100
                )
            })

        lines.sort(
            key=lambda x: (
                round(x["bbox"][1], 3),
                x["bbox"][0]
            )
        )

        return lines   


def ocr_lines_to_blocks(lines: list[dict[str, Any]], *, page_num: int | None = None,
                        slide_num: int | None = None) -> list[ContentBlock]:
    """Join OCR lines into paragraphs while keeping legal headings and numbered items."""
    result: list[ContentBlock] = []
    acc: list[dict] = []

    # Detect a visually aligned run of rows before paragraph reconstruction.
    # This is a heuristic, not a full table structure recognizer.
    rows: list[list[dict]] = []
    for line in lines:
        cy = (line["bbox"][1] + line["bbox"][3]) / 2
        if rows:
            last = rows[-1]
            pc = sum((x["bbox"][1]+x["bbox"][3])/2 for x in last)/len(last)
            height = max(x["bbox"][3]-x["bbox"][1] for x in last)
            if abs(cy-pc) <= max(0.008, height*0.45):
                last.append(line)
                continue
        rows.append([line])

    def looks_like_table_row(row: list[dict]) -> bool:
        if len(row) < 2:
            return False
        ordered = sorted(row, key=lambda x: x["bbox"][0])
        return (ordered[-1]["bbox"][0] - ordered[0]["bbox"][0]) > .18

    table_groups: dict[int, tuple[int, ContentBlock]] = {}
    j = 0
    while j < len(rows):
        if not looks_like_table_row(rows[j]):
            j += 1
            continue
        end = j
        while end < len(rows) and looks_like_table_row(rows[end]):
            end += 1
        if end-j >= 3:
            batch = [sorted(row, key=lambda x: x["bbox"][0]) for row in rows[j:end]]
            # Require compatible rough column counts and column alignment.
            sizes = Counter(len(row) for row in batch)
            frequent, nfreq = sizes.most_common(1)[0]
            if frequent >= 2 and nfreq >= 3:
                coords = [x["bbox"] for row in batch for x in row]
                bbox = [min(x[0] for x in coords), min(x[1] for x in coords),
                        max(x[2] for x in coords), max(x[3] for x in coords)]
                md = table_to_markdown([[x["text"] for x in row] for row in batch])
                item = make_block("table", md, page_num=page_num, slide_num=slide_num,
                                  bbox=bbox, source="rapidocr", ocr=True,
                                  approximate_table=True)
                table_groups[j] = (end, item)
                j = end
                continue
        j += 1

    def flush():
        if not acc:
            return
        text = "\n".join(x["text"] for x in acc)
        k, level, normalized = detect_type(text, "image_ocr")
        # The default OCR provenance is preserved even if a block is classified as heading/list.
        bbox = [min(x["bbox"][0] for x in acc), min(x["bbox"][1] for x in acc),
                max(x["bbox"][2] for x in acc), max(x["bbox"][3] for x in acc)]
        b = make_block(k, normalized, page_num=page_num, slide_num=slide_num,
                       level=level, bbox=[round(x, 5) for x in bbox],
                       source="rapidocr", ocr=True,
                       ocr_confidence=round(sum(x["confidence"] for x in acc)/len(acc), 4))
        result.append(b)
        acc.clear()

    row_idx = 0
    while row_idx < len(rows):
        if row_idx in table_groups:
            flush()
            end, block = table_groups[row_idx]
            result.append(block)
            row_idx = end
            continue
        for line in sorted(rows[row_idx], key=lambda x: x["bbox"][0]):
            if not acc:
                acc.append(line)
                continue
            prev = acc[-1]
            last_height = max(0.005, prev["bbox"][3] - prev["bbox"][1])
            gap = line["bbox"][1] - prev["bbox"][3]
            curr = line["text"]
            is_new_structure = bool(CHAPTER.match(curr) or ARTICLE.match(curr) or
                                    CLAUSE.match(curr) or POINT.match(curr) or BULLET.match(curr))
            new_indent = (line["bbox"][0] - prev["bbox"][0]) > 0.055
            if is_new_structure or gap > max(0.012, last_height * 0.85) or (new_indent and gap > 0):
                flush()
            acc.append(line)
        row_idx += 1
    flush()
    return result


def get_pdf_text_blocks(page: Any, page_num: int, ocr: Any = None,
                        asset_dir: Path | str | None = None, stats: dict | None = None) -> list[ContentBlock]:
    """Born-digital page -> blocks in MuPDF raw order (see extract_pdf_page)."""
    import tempfile
    return extract_pdf_page(page, page_num, asset_dir or tempfile.mkdtemp(prefix="rag_assets_"), ocr, stats)


_MONO = re.compile(r"mono|courier|consolas|menlo|typewriter|^(SF|CM|LM|VN)TT|mn-regular", re.I)
_STRONG_MATH = re.compile(r"^(CM(MI|SY|EX|MIB|BSY)|MSAM|MSBM|EUFM|LMMath|PazoMath|.*Math)", re.I)
_MATHISH = re.compile(r"^(CMSS|CMSSI|CMR)", re.I)
_MATH_CHARS = "=+−<>≈≠≤≥∈∑∏±×÷·¯∞/^_"
_BOLD = re.compile(r"bold|black|heavy|medi\b|demi|-bd\b", re.I)
_LIST_MARK = re.compile(r"^\s*([-–•*·]|\d+[.)])\s+")
_FOOTERS = (re.compile(r"^\s*(trang\s+|page\s+)?\d{1,4}\s*(/|of|trên)\s*\d{1,4}\s*$", re.I),
            re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$"))
_CAPTION = re.compile(r"^(Figure|Fig\.|Hình|Biểu đồ|Chart)\s*\d+", re.I)
_CODE_CHARS = "=(){};$>#"
_NO_PRUNE = {"heading", "code", "formula", "figure", "table"}


class _Ln:
    """One text line: spans keep raw chars; text has spaces re-inserted where a real gap exists."""


def _make_line(l: dict, midx: int, pos: int):
    spans, prev = [], None
    for s in l.get("spans", []):
        chars = s.get("chars") or []
        if not chars:
            continue
        parts = []
        for c in chars:
            ch = c["c"]
            # PDFs with a missing space glyph: a real gap >= 0.20em means a space (also across spans).
            if prev and ch.strip() and prev[0].strip() and s["size"] > 0 and c["bbox"][0] - prev[1] >= 0.2 * s["size"]:
                parts.append(" ")
            parts.append(ch)
            prev = (ch, c["bbox"][2])
        spans.append({"font": re.sub(r"^[A-Z]{6}\+", "", s.get("font", "")), "size": s["size"],
                      "flags": s["flags"], "text": "".join(parts), "chars": chars})
    nb = [s for s in spans if s["text"].strip()]
    if not nb:
        return None
    ln = _Ln()
    ln.spans, ln.nb, ln.midx, ln.pos = spans, nb, midx, pos
    ln.x0, ln.y0, ln.x1, ln.y1 = l["bbox"]
    ln.cx, ln.cy = (ln.x0 + ln.x1) / 2, (ln.y0 + ln.y1) / 2
    weight = Counter()
    for s in nb:
        weight[round(s["size"], 1)] += len(s["text"].strip())
    ln.size = weight.most_common(1)[0][0]
    ln.bold = all(s["flags"] & 16 or _BOLD.search(s["font"]) for s in nb)
    ln.text = "".join(s["text"] for s in spans).strip()
    ln.raw = "".join(c["c"] for s in spans for c in s["chars"])
    ln.code = False
    strong = [s for s in nb if _STRONG_MATH.match(s["font"])]
    # text fonts (CMR/CMSS) count as prose; without a strong span a line seeds only if it has a math
    # symbol or is a 1-2 letter variable (never pure digits: axis ticks).
    prose = " ".join(s["text"] for s in nb if s not in strong)
    weak = any(_MATHISH.match(s["font"]) for s in nb) and (
        any(c in _MATH_CHARS for c in ln.text) or (ln.text.isalpha() and len(ln.text) <= 2))
    # shortcut: lines with a strong math span AND a prose word >=4 letters stay text (inline math).
    # (>=4-letter word, or two >=3-letter words, marks prose; spec says only the 4-letter rule)
    ln.seed = bool(strong or weak) and not (re.search(r"[^\W\d_]{4,}", prose)
                                               or len(re.findall(r"[^\W\d_]{3,}", prose)) >= 2)
    return ln


def _hit(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _within(a, b, tol=1.0) -> bool:
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def _doc_font_stats(doc) -> tuple[float, list[float]]:
    """(body size, heading sizes desc) from a sample of pages; cached on the doc object."""
    import pymupdf as fitz
    cached = getattr(doc, "_rag_font_stats", None)
    if cached:
        return cached
    n = doc.page_count
    pages = range(n) if n <= 40 else sorted({int(i * (n - 1) / 39) for i in range(40)})
    weight = Counter()
    for i in pages:
        for b in doc[i].get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    t = len(s["text"].strip())
                    if t:
                        weight[round(s["size"] * 2) / 2] += t
    body = weight.most_common(1)[0][0] if weight else 10.0
    stats = (body, sorted({k for k in weight if k >= 1.2 * body}, reverse=True))
    try:
        doc._rag_font_stats = stats
    except Exception:
        pass
    return stats


def _join_lines(texts: list[str]) -> str:
    out = ""
    for t in texts:
        frag = out.rsplit(None, 1)[-1][:-1] if out.endswith("-") else ""
        if frag and out[-2].isalpha() and t[:1].islower():
            # drop the hyphen only after a plain lowercase fragment; 'BERT-' / 'state-of-the-' keep it, no space
            out = (out[:-1] if len(frag) >= 2 and frag.isalpha() and frag.islower() else out) + t
        else:
            out = f"{out} {t}" if out else t
    return out


def _code_text(grp: list) -> tuple[str, str]:
    widths = sorted(c["bbox"][2] - c["bbox"][0] for l in grp for s in l.nb for c in s["chars"] if c["c"].strip())
    cw = widths[len(widths) // 2] if widths else grp[0].size * 0.6
    rows = []
    for l in grp:
        raw = l.raw.replace(" ", " ").expandtabs(4).rstrip()
        rows.append((l, raw.lstrip(" "), len(raw) - len(raw.lstrip(" "))))
    m = min((l.x0 for l, _, lead in rows if lead == 0), default=None)
    inds = [lead if lead or m is None else max(0, round((l.x0 - m) / max(cw, 0.1))) for l, _, lead in rows]
    cut = min(inds)
    text = "\n".join(" " * (i - cut) + body for (_, body, _), i in zip(rows, inds))
    first = rows[0][1]
    return text, "python" if first.startswith(">>> ") else ("shell" if first.startswith("$ ") else "")


def extract_pdf_page(page: Any, page_num: int, asset_dir: Path | str, ocr: Any = None,
                     stats: dict | None = None) -> list[ContentBlock]:
    """Blocks of a born-digital page in MuPDF RAW order (never sorted: sort breaks 2-column pages).

    Assets are written as tmp_p<page>_<k>.png in asset_dir; execute() renames them to <block_id>.png.
    """
    import pymupdf as fitz
    asset_dir = Path(asset_dir)
    W, H = page.rect.width, page.rect.height
    parea = W * H
    body, heads = _doc_font_stats(page.parent) if page.parent else (10.0, [])
    raw = page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXT_PRESERVE_WHITESPACE
                        | fitz.TEXT_PRESERVE_IMAGES)
    lines, images, pos = [], [], 0
    for bi, b in enumerate(raw["blocks"]):
        if b["type"] == 1:
            images.append((pos, tuple(b["bbox"]), b.get("image")))
            pos += 1
            continue
        for l in b.get("lines", []):
            ln = _make_line(l, bi, pos)
            pos += 1
            if ln:
                lines.append(ln)
    # Footers: page counters in the bottom 10% only (slide titles live at the top, so no top rule).
    lines = [l for l in lines if not (l.cy > 0.9 * H and any(p.match(clean_text(l.text)) for p in _FOOTERS))]

    items: list[tuple[float, ContentBlock, tuple]] = []  # (anchor, block, rect)
    counter = [0]

    def norm(r):
        return [round(r[0] / W, 5), round(r[1] / H, 5), round(r[2] / W, 5), round(r[3] / H, 5)]

    def save(rect, dpi):
        counter[0] += 1
        name = f"tmp_p{page_num:04d}_{counter[0]}.png"
        clip = fitz.Rect(rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3) & page.rect
        asset_dir.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(clip=clip, dpi=dpi).save(str(asset_dir / name))
        return name

    def blk(btype, text, rect, anchor, name=None, **meta):
        b = make_block(btype, text, page_num=page_num, bbox=norm(rect), asset=f"assets/{name}" if name else None,
                       source="pymupdf")
        b.metadata.update(meta)  # not via make_block(**): metadata key 'kind' clashes with its arg
        b._tmp_asset = name
        items.append((anchor, b, tuple(rect)))
        return b

    def take(rect):
        inside = [l for l in lines if rect[0] <= l.cx <= rect[2] and rect[1] <= l.cy <= rect[3]]
        ids = {id(l) for l in inside}
        lines[:] = [l for l in lines if id(l) not in ids]
        return inside

    def anchor_for(inside, y0):
        return inside[0].pos if inside else next((l.pos - 0.5 for l in lines if l.y0 >= y0), 1e9)

    # tables
    table_rects = []
    try:
        found = page.find_tables().tables
    except Exception:
        found = []
    for t in found:
        if t.row_count < 2 or t.col_count < 2:
            continue
        rows = [["" if c is None else str(c) for c in r] for r in t.extract()]
        # find_tables drops spaces ('Petersetal'): rebuild each cell from our gap-aware line text.
        for ri, trow in enumerate(t.rows[:len(rows)]):
            for ci, cb in enumerate(trow.cells[:len(rows[ri])]):
                if cb is None:
                    continue
                got = [l for l in lines if cb[0] <= l.cx <= cb[2] and cb[1] <= l.cy <= cb[3]]
                if got:
                    rows[ri][ci] = " ".join(clean_text(l.text) for l in got)
        if not _has_cells(rows):
            continue
        rect = tuple(t.bbox)
        table_rects.append(rect)
        inside = take(rect)
        blk("table", table_to_markdown(rows), rect, anchor_for(inside, rect[1]),
            rows=rows, n_rows=len(rows), n_cols=max(len(r) for r in rows))

    # vector figures (before formulas, so axis ticks / labels inside a figure never become formulas)
    def _flat_fill(d):  # fill-only highlight: ~one text line tall/wide, or a wide flat band
        w, h = d["rect"].width, d["rect"].height
        return d.get("type") == "f" and not any(it[0] == "c" for it in d["items"]) and (
            min(w, h) <= 2.5 * body or (w > .5 * W and h < .15 * H))
    try:
        alld = page.get_drawings()
        if len(alld) > MAX_DRAWINGS:
            dr, vclusters = [], []
            if stats is not None:
                stats["drawing_cap_pages"] = stats.get("drawing_cap_pages", 0) + 1
        else:
            dr = [d for d in alld if d["rect"].width * d["rect"].height <= .6 * parea
                  and d["rect"].width <= .95 * W and not _flat_fill(d)]
            vclusters = page.cluster_drawings(drawings=dr) if dr else []
    except Exception:
        dr, vclusters = [], []
    for r in vclusters:
        r = tuple(r)
        area = (r[2] - r[0]) * (r[3] - r[1])
        mine = [d for d in dr if _within(tuple(d["rect"]), r)]
        if area < .03 * parea or len(mine) < 5:
            continue
        # a figure needs a stroke or curve; plain filled boxes are page decoration
        if not any(d.get("type") in ("s", "fs") or any(it[0] == "c" for it in d["items"]) for d in mine):
            continue
        if any(_hit(r, t) and (min(r[2], t[2]) - max(r[0], t[0])) * (min(r[3], t[3]) - max(r[1], t[1])) > .5 * area
               for t in table_rects):
            continue
        txt = [l for l in lines if r[0] <= l.cx <= r[2] and r[1] <= l.cy <= r[3]]
        nch = sum(len(l.text) for l in txt)
        if len(txt) > 5 and nch > 300 and nch >= 20 * len(txt):
            continue  # body text over a background (long lines); short diagram labels stay in the figure
        inside = take(r)
        blk("figure", "", r, anchor_for(inside, r[1]), name=save(r, 150), caption="",
            ocr_text="\n".join(clean_text(l.text) for l in inside), kind="vector")

    # formulas: seeds clustered by union-find on inflated bboxes
    seeds = [l for l in lines if l.seed and not _MONO_LINE(l)]
    par = list(range(len(seeds)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    infl = [(l.x0 - .5 * l.size, l.y0 - .4 * l.size, l.x1 + .5 * l.size, l.y1 + .4 * l.size) for l in seeds]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            if _hit(infl[i], infl[j]):
                par[find(j)] = find(i)
    clusters: dict[int, list] = {}
    for i, l in enumerate(seeds):
        clusters.setdefault(find(i), []).append(l)
    for members in list(clusters.values())[:20]:
        chars = "".join(m.text for m in members).replace(" ", "")
        alone = len(members) == 1 and not any(l.midx == members[0].midx for l in lines if l is not members[0])
        # inline math / ticks: keep as text unless 2+ lines, symbol-heavy, or alone in its block
        if len(members) < 2 and not alone and sum(not c.isalnum() and c not in ".,;:" for c in chars) < .25 * len(chars):
            continue
        rect = (min(m.x0 for m in members), min(m.y0 for m in members),
                max(m.x1 for m in members), max(m.y1 for m in members))
        ids = {id(m) for m in members}
        lines[:] = [l for l in lines if id(l) not in ids]
        blk("formula", " ".join(m.text for m in members), rect, members[0].pos,
            name=save(rect, 200), format="text")

    # raster figures
    for ipos, r, data in images:
        area = (r[2] - r[0]) * (r[3] - r[1])
        if area < .008 * parea:  # full-page scans never get here (page-level OCR path)
            continue
        text = ""
        if ocr is not None and data:
            try:
                text = "\n".join(x["text"] for x in ocr.run(data)).strip()
            except Exception:
                text = ""
        ok = len(re.findall(r"[^\W_]|\s", text)) >= .6 * len(text) and len(re.findall(r"[^\W\d_]{3,}", text)) >= 2
        blk("figure", "", r, ipos, name=save(r, 150), caption="", ocr_text=text if ok else "", kind="raster")

    # code runs: mono lines, big enough, outside page bands, 2+ close lines or one with code chars
    prev, runs = None, {}
    for l in lines:
        l.code = _MONO_LINE(l) and l.size >= 7 and .05 * H <= l.cy <= .92 * H
        if l.code:
            if prev is not None and prev.code and l.y0 - prev.y1 < 1.0 * (l.y1 - l.y0):
                l.run = prev.run
            else:
                l.run = len(runs)
            runs.setdefault(l.run, []).append(l)
        prev = l
    for run in runs.values():
        if len(run) < 2 and not any(c in l.raw for l in run for c in _CODE_CHARS):
            for l in run:
                l.code = False

    slide = W > H
    first_text = True
    i = 0
    while i < len(lines):
        l0 = lines[i]
        key = ("c", l0.run) if l0.code else ("t", l0.midx)
        grp = [l0]
        i += 1
        while i < len(lines) and (("c", lines[i].run) if lines[i].code else ("t", lines[i].midx)) == key:
            grp.append(lines[i])
            i += 1
        rect = (min(l.x0 for l in grp), min(l.y0 for l in grp), max(l.x1 for l in grp), max(l.y1 for l in grp))
        if l0.code:
            text, lang = _code_text(grp)
            b = make_block("code", text, page_num=page_num, bbox=norm(rect), verbatim=True,
                           language=lang, source="pymupdf")
            b._tmp_asset = None
            items.append((l0.pos, b, rect))
            first_text = False
            continue
        texts = [clean_text(l.text) for l in grp]
        joined = _join_lines(texts)
        size, bold = grp[0].size, all(l.bold for l in grp)
        is_head = (len(joined) <= 120 and not joined.endswith(".") and (
            (size >= 1.2 * body and len(grp) <= 3) or (bold and len(grp) == 1)))
        slide_title = slide and first_text and rect[1] < .2 * H and size >= 1.1 * body and len(joined) <= 120
        first_text = False
        if is_head or slide_title:
            rank = next((k for k, h in enumerate(heads) if abs(h - size) <= .5), len(heads))
            level = 1 if slide_title else (3 if rank == len(heads) else min(rank + 1, 3))
            kind, level, txt = detect_type(joined, "heading", level)
            kind_meta = {"font_size": size, "bold": bold}
        elif _LIST_MARK.match(texts[0]):
            items_l: list[str] = []
            for t in texts:
                if _LIST_MARK.match(t) or not items_l:
                    items_l.append(t)
                else:
                    items_l[-1] += " " + t  # hanging line joins the previous item
            kind, level, txt, kind_meta = "list", 0, "\n".join(items_l), {}
        else:
            kind, level, txt = detect_type(joined, "paragraph")
            kind_meta = {"font_size": size, "bold": bold} if kind == "heading" else {}
        if txt:
            b = make_block(kind, txt, page_num=page_num, level=level, bbox=norm(rect), source="pymupdf", **kind_meta)
            b._tmp_asset = None
            items.append((l0.pos, b, rect))

    # captions: nearest paragraph within 40pt, moved into the figure
    for _, fig, fr in [x for x in items if x[1].type == "figure"]:
        best, bd = None, 41.0
        for it in items:
            if it[1].type == "paragraph" and _CAPTION.match(it[1].text):
                r = it[2]
                d = math.hypot(max(0, r[0] - fr[2], fr[0] - r[2]), max(0, r[1] - fr[3], fr[1] - r[3]))
                if d <= bd:
                    best, bd = it, d
        if best:
            items.remove(best)
            fig.metadata["caption"] = best[1].text
        fig.text = clean(f"{fig.metadata['caption']}\n{fig.metadata['ocr_text']}")
    items.sort(key=lambda x: x[0])
    return [b for _, b, _ in items]


def _MONO_LINE(l) -> bool:
    return all(_MONO.search(s["font"]) or s["flags"] & 8 for s in l.nb)


def page_has_full_scan(page: Any, text: str) -> bool:
    """Detect scanned page even if a short digital signature text is overlaid."""
    if len(text.strip()) < MIN_PDF_TEXT:
        return True
    p_area = max(page.rect.width * page.rect.height, 1)
    large_image = any(
        ((x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1])) / p_area > 0.6
        for x in page.get_image_info(xrefs=False)
    )
    return large_image and len(text.strip()) < 250


def parse_pdf(path: Path, ocr: OCR, asset_dir: Path | None = None) -> tuple[list[ContentBlock], dict[str, Any]]:
    import pymupdf as fitz
    out: list[ContentBlock] = []
    counts = Counter()
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF có mật khẩu, cần mở khóa trước khi xử lý")
        for p in doc:
            idx = p.number + 1
            native = p.get_text("text", sort=True).strip()
            scanned = page_has_full_scan(p, native)
            if scanned:
                # Never generate a giant 300 DPI bitmap then downsize it.
                # Render close to 1600px on the longest side in the first place.
                zoom = min(300 / 72, MAX_OCR_SIDE / max(p.rect.width, p.rect.height))
                pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                   colorspace=fitz.csRGB, alpha=False)
                from PIL import Image
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                b = ocr_lines_to_blocks(ocr.run(image), page_num=idx)
                counts["ocr_pages"] += 1
            else:
                try:
                    b = get_pdf_text_blocks(p, idx, ocr, asset_dir, counts)
                except Exception as e:  # noqa: BLE001 - one bad page must not lose the document
                    logging.warning("parse_pdf: page %d failed (%s); falling back to plain text", idx, e)
                    counts["page_errors"] += 1
                    b = [make_block("paragraph", clean_text(ln), page_num=idx, source="pymupdf")
                         for ln in p.get_text("text").splitlines() if clean_text(ln)]
                    for x in b:
                        x._tmp_asset = None
                counts["digital_pages"] += 1
                counts["figures"] += sum(1 for x in b if x.type == "figure")
            if not b:
                counts["empty_pages"] += 1
            out.extend(b)
            print(f"[parse] Trang {idx}/{len(doc)}: {OCR_ENGINE_NAME if scanned else 'PyMuPDF'} - {len(b)} blocks", flush=True)
    return out, dict(counts)


def docx_paragraph_blocks(paragraph: Any, ocr: OCR) -> list[ContentBlock]:
    """Reads individual run/drawing XML nodes to keep image OCR in exact inline order."""
    from docx.oxml.ns import qn
    results: list[ContentBlock] = []
    style = (paragraph.style.name or "").lower() if paragraph.style else ""
    match = re.search(r"(?:heading|tiêu đề|title)\s*(\d+)", style)
    h_level = min(int(match.group(1)),3) if match else (1 if "title" in style else 0)
    buffer: list[str] = []

    def flush():
        text = clean("".join(buffer))
        buffer.clear()
        if text:
            k, lvl, s = detect_type(text, "heading" if h_level else "paragraph", h_level)
            results.append(make_block(k, s, level=lvl, source="python-docx"))

    # Paragraph XML preserves the order of runs and inline images.
    for child in paragraph._p:
        local = child.tag.rsplit("}",1)[-1]
        if local not in ("r", "hyperlink", "sdt", "ins"):
            continue
        for el in child.iter():
            if el.tag == qn("w:t") and el.text:
                buffer.append(el.text)
            elif el.tag == qn("w:tab"):
                buffer.append(" ")
            elif el.tag == qn("w:br"):
                buffer.append("\n")
            elif el.tag == qn("a:blip"):
                rid = el.get(qn("r:embed"))
                if not rid:
                    continue
                flush()
                try:
                    image_part = paragraph.part.related_parts[rid]
                    recovered = ocr_lines_to_blocks(ocr.run(image_part.blob))
                    for b in recovered:
                        b.type = "image_ocr" if b.type not in ("heading", "list") else b.type
                        b.metadata["embedded_in"] = "docx"
                    results.extend(recovered)
                except KeyError:
                    continue
    flush()
    return results


def _has_cells(rows) -> bool:
    return any(clean(c) for r in rows for c in r)


def table_to_markdown(rows: Iterable[Iterable[str]]) -> str:
    safe = [[clean(c).replace("\n", "<br>").replace("|", "\\|") for c in row] for row in rows]
    if not safe:
        return ""
    n = max(len(r) for r in safe)
    safe = [r + [""]*(n-len(r)) for r in safe]
    return "\n".join(["| " + " | ".join(safe[0]) + " |",
                      "| " + " | ".join(["---"]*n) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in safe[1:]])


def parse_docx(path: Path, ocr: OCR) -> tuple[list[ContentBlock], dict[str, Any]]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(path)
    out = []
    for elem in doc.element.body.iterchildren():
        if elem.tag.endswith("}p"):
            out.extend(docx_paragraph_blocks(Paragraph(elem, doc), ocr))
        elif elem.tag.endswith("}tbl"):
            table = Table(elem, doc)
            rows = []
            for row in table.rows:
                items = []
                for cell in row.cells:
                    snippets = []
                    for p in cell.paragraphs:
                        for b in docx_paragraph_blocks(p, ocr):
                            snippets.append(b.text)
                    items.append("\n".join(snippets))
                rows.append(items)
            if _has_cells(rows):
                out.append(make_block("table", table_to_markdown(rows), source="python-docx"))
    return out, {}


def parse_pptx(path: Path, ocr: OCR) -> tuple[list[ContentBlock], dict[str, Any]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
    slides = Presentation(str(path)).slides
    out = []
    big = 1 << 62

    def shapes_in_reading_order(shapes):
        # inherited placeholders have None position: keep them after positioned shapes, original order (stable)
        def pos(v):
            return big if v is None else v
        return sorted(shapes, key=lambda s: (pos(s.top), pos(s.left)))

    def num_s(v):
        return None if v is None else float(format(float(v), ".6g"))

    def chart_block(chart, num):
        plots = list(chart.plots)
        cats: list[str] = []
        try:
            cats = [str(c) for c in list(plots[0].categories)]
        except Exception:
            pass
        series = []
        for plot in plots:
            for s in plot.series:
                try:
                    vals = [num_s(v) for v in s.values]
                except Exception:
                    vals = []
                series.append({"name": str(s.name), "values": vals})
        def fmt(v):
            return "" if v is None else format(v, ".6g")
        rows = []
        for s in series:
            if cats:
                body = ", ".join(f"{c}={fmt(v)}" for c, v in zip(cats, s["values"]))
            else:
                body = ", ".join(fmt(v) for v in s["values"])
            rows.append(f"{s['name']}: {body}")
        fig = make_block("figure", "\n".join(rows), slide_num=num, source="python-pptx")
        fig.metadata["kind"] = "chart"  # 'kind' is make_block's positional name
        fig.metadata["chart"] = {"type": str(getattr(chart.chart_type, "name", chart.chart_type)),
                                 "categories": cats, "series": series}
        return fig
    def visit(shape, num):
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for item in shapes_in_reading_order(shape.shapes):
                visit(item, num)
            return
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            for block in ocr_lines_to_blocks(ocr.run(shape.image.blob), slide_num=num):
                block.type = "image_ocr" if block.type not in ("heading", "list") else block.type
                block.metadata["embedded_in"] = "pptx"
                out.append(block)
            return
        if getattr(shape, "has_chart", False):
            try:
                out.append(chart_block(shape.chart, num))
            except Exception:
                pass
            return
        if shape.has_table:
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            if _has_cells(rows):
                out.append(make_block("table", table_to_markdown(rows), slide_num=num, source="python-pptx"))
            return
        if shape.has_text_frame:
            is_title = False
            if shape.is_placeholder:
                try:
                    is_title = shape.placeholder_format.type in (
                        PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE)
                except Exception:
                    pass
            for i, para in enumerate(shape.text_frame.paragraphs):
                text = clean(para.text)
                if not text:
                    continue
                is_bullet = bool(para.level or para._p.xpath("./a:pPr/a:buChar") or para._p.xpath("./a:pPr/a:buAutoNum"))
                kind, lvl, txt = detect_type(text, "list" if is_bullet else "paragraph")
                if i == 0 and not is_bullet and para.level == 0 and len(text) < 100:
                    sizes = [r.font.size.pt for r in para.runs if r.font.size]
                    if kind == "paragraph" and (is_title or (sizes and max(sizes) >= 24)):
                        kind, lvl = "heading", 1
                out.append(make_block(kind, txt, slide_num=num, level=lvl, source="python-pptx"))
    for num, slide in enumerate(slides, 1):
        for shape in shapes_in_reading_order(slide.shapes):
            visit(shape, num)
    return out, {}


def parse_html(path: Path, ocr: OCR) -> tuple[list[ContentBlock], dict[str, Any]]:
    from bs4 import BeautifulSoup, Tag
    from markdownify import markdownify
    raw, encoding = read_text_file(path)
    soup = BeautifulSoup(raw, "html.parser")
    root = soup.body or soup
    for junk in root.find_all(["script", "style", "noscript", "nav"]):
        junk.decompose()
    result = []
    # Leaf-target walker prevents double-counting paragraphs nested in div/section.
    def visit(el):
        if not isinstance(el, Tag):
            return
        name = el.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "img", "pre", "blockquote", "math"):
            if name == "pre":
                code = el.find("code")
                cls = " ".join((code or el).get("class") or [])
                lang = (re.search(r"(?:language|lang)-([\w+#.-]+)", cls) or [None, ""])[1]
                text = (code or el).get_text()
                if text.strip():
                    result.append(make_block("code", text, verbatim=True, language=lang, source="beautifulsoup4"))
                return
            if name == "math":
                ann = el.find("annotation", attrs={"encoding": "application/x-tex"})
                text, fmt = (ann.get_text().strip(), "latex") if ann else (el.get_text(" ", strip=True), "text")
                if text:
                    result.append(make_block("formula", text, verbatim=fmt == "latex", format=fmt,
                                             source="beautifulsoup4"))
                return
            if name == "img":
                src = el.get("src") or ""
                content = None
                try:
                    if src.startswith("data:image/") and "," in src:
                        content = base64.b64decode(src.split(",", 1)[1])
                    elif src and not re.match(r"^[a-z]+://", src, re.I):
                        file = (path.parent / src.split("?", 1)[0]).resolve()
                        if file.is_relative_to(path.parent.resolve()) and file.is_file():
                            content = file.read_bytes()
                except Exception:
                    content = None
                if content:
                    for b in ocr_lines_to_blocks(ocr.run(content)):
                        b.type = "image_ocr" if b.type not in ("heading", "list") else b.type
                        result.append(b)
                return
            if name == "table":
                rows = [[c.get_text(" ", strip=True) for c in row.find_all(["td", "th"], recursive=False)]
                        for row in el.find_all("tr")]
                if _has_cells(rows):
                    result.append(make_block("table", table_to_markdown(rows), source="beautifulsoup4"))
                return
            if name in ("p", "li") and el.find(["img", "table", "math"]):
                # Preserve inline images relative to the surrounding text.
                inline_text: list[str] = []
                def visit_inline(child):
                    if isinstance(child, Tag):
                        if child.name in ("img", "math"):
                            if clean(" ".join(inline_text)):
                                result.append(make_block("paragraph", " ".join(inline_text), source="beautifulsoup4"))
                            inline_text.clear()
                            visit(child)
                        else:
                            for nested in child.children:
                                visit_inline(nested)
                    else:
                        inline_text.append(str(child))
                for child in el.children:
                    visit_inline(child)
                if clean(" ".join(inline_text)):
                    result.append(make_block("paragraph", " ".join(inline_text), source="beautifulsoup4"))
                return
            txt = clean(markdownify(str(el), heading_style="ATX"))
            if name.startswith("h") and name[1:].isdigit():
                txt = el.get_text(" ", strip=True)
            kind = "heading" if name.startswith("h") and name[1:].isdigit() else (
                "list" if name == "li" else "paragraph")
            kind, level, txt = detect_type(txt, kind, min(int(name[1]),3) if kind == "heading" else 0)
            if txt:
                result.append(make_block(kind, txt, level=level, source="beautifulsoup4"))
            return
        for child in el.children:
            visit(child)
    visit(root)
    return result, {"encoding": encoding}


def parse_plain(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    content, encoding = read_text_file(path)
    return parse_markdown_blocks(content, source="text"), {"encoding": encoding}


def remove_repeated_page_noise(blocks: list[ContentBlock]) -> tuple[list[ContentBlock], int]:
    pages = {b.location.get("page_num") for b in blocks if b.location.get("page_num")}
    if len(pages) < 3:
        return blocks, 0
    candidates: dict[str,set[int]] = {}
    def key(text: str) -> str:
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.casefold())).strip()
    for b in blocks:
        pnum = b.location.get("page_num")
        if not pnum or not b.bbox or b.type in _NO_PRUNE or len(b.text) > 100:
            continue
        at_edge = b.bbox[1] < .11 or b.bbox[3] > .89
        if at_edge:
            candidates.setdefault(key(b.text), set()).add(pnum)
    threshold = max(3, math.ceil(len(pages)*0.45))
    repeated = {s for s, pp in candidates.items() if len(pp) >= threshold}
    kept = []
    removed = 0
    for b in blocks:
        near_edge = bool(b.bbox and (b.bbox[1] < .11 or b.bbox[3] > .89))
        if (b.type not in _NO_PRUNE and near_edge and
            (key(b.text) in repeated or PAGE_NUMBER.fullmatch(b.text.strip()))):
            removed += 1
        else:
            kept.append(b)
    return kept, removed


def reconstruct_hierarchy(blocks: list[ContentBlock]):
    chapter = None
    chapter_title = None
    article = None
    article_title = None
    clause = None
    point = None
    awaiting_chapter_title = False
    for i, b in enumerate(blocks):
        # One legal clause / article may be a single OCR line or a wrapped paragraph.
        first = b.text.splitlines()[0].strip() if b.text else ""
        mc = CHAPTER.match(first)
        ma = ARTICLE.match(first)
        mk = CLAUSE.match(first)
        mp = POINT.match(first)
        if mc:
            chapter = mc.group(1).upper()
            rest_lines = b.text.splitlines()[1:]
            chapter_title = mc.group(2).strip() or (rest_lines[0].strip() if rest_lines and rest_lines[0].strip().isupper() else None)
            article, article_title, clause, point = None, None, None, None
            awaiting_chapter_title = not bool(chapter_title)
            b.type, b.level = "heading", 1
        elif awaiting_chapter_title and b.type != "heading" and b.text.isupper() and len(b.text) < 100:
            chapter_title = b.text
            awaiting_chapter_title = False
            b.type, b.level = "heading", 1
        elif ma:
            article, article_title = int(ma.group(1)), ma.group(2).strip()
            clause = point = None
            awaiting_chapter_title = False
            b.type, b.level = "heading", 2
        elif mk and article is not None:
            clause = int(mk.group(1))
            point = None
            b.type = "list"
        elif mp and article is not None and clause is not None:
            point = mp.group(1).lower()
            b.type = "list"
        crumb = []
        if chapter: crumb.append(f"Chương {chapter}" + (f" - {chapter_title}" if chapter_title else ""))
        if article is not None: crumb.append(f"Điều {article}" + (f". {article_title}" if article_title else ""))
        if clause is not None: crumb.append(f"Khoản {clause}")
        if point is not None: crumb.append(f"Điểm {point}")
        b.breadcrumb = " > ".join(crumb)
        b.metadata.update({"chapter": chapter, "article": article,
                           "clause": clause, "point": point})
        loc = b.location
        key = f"p{loc['page_num']:04d}" if "page_num" in loc else (
              f"s{loc['slide_num']:04d}" if "slide_num" in loc else "doc")
        b.id = f"{key}_b{i+1:05d}"


def markdown_output(blocks: list[ContentBlock]) -> str:
    output = []
    previous = None
    for b in blocks:
        loc = b.location
        if loc and loc != previous:
            if "page_num" in loc:
                output.append(f"<!-- PDF page {loc['page_num']} -->")
            else:
                output.append(f"<!-- Slide {loc['slide_num']} -->")
            previous = loc
        if b.type == "heading":
            output.append("#" * (b.level or 3) + " " + b.text)
        elif b.type == "image_ocr":
            output.append(b.text)  # avoid losing legal numbering by inserting labels
        else:
            output.append(b.text)
    return "\n\n".join(output).rstrip() + "\n"


def execute(src: Path, dest: Path | None = None, force: bool = False,
            ocr: OCR | None = None) -> dict[str, Any]:
    src = src.resolve()
    if not src.is_file() or src.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Không thấy file hoặc định dạng không được hỗ trợ: {src}")
    if dest is None:
        dest = src.parent / "processed" / src.stem  # per-document: assets/<block_id>.png must not collide
    dest.mkdir(parents=True, exist_ok=True)
    md_path = dest / f"{src.stem}.md"
    json_path = dest / f"{src.stem}.json"
    digest = sha256(src)
    if not force and md_path.exists() and json_path.exists():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if (cached.get("source", {}).get("sha256") == digest and cached.get("parser_version") == PARSER_VERSION
                    and all((dest / b["asset"]).exists() for b in cached.get("blocks", []) if b.get("asset"))):
                print(f"[parse] CACHE HIT sha256={digest[:12]} | {json_path}")
                return cached
        except (OSError, ValueError):
            pass
    for f in (dest / "assets").glob("*.png"):  # stale assets from an earlier parse
        f.unlink()
    print(f"[parse] START {src.name} | sha256={digest[:12]}", flush=True)
    ext = src.suffix.lower()
    if ext in {".md", ".txt"}:
        blocks, stats = parse_plain(src)
    elif ext == ".html" or ext == ".htm":
        blocks, stats = parse_html(src, ocr or OCR())
    else:
        engine = ocr or OCR()  # Lazy: import OCR model only when necessary.
        if ext == ".pdf":
            blocks, stats = parse_pdf(src, engine, dest / "assets")
        elif ext == ".docx":
            blocks, stats = parse_docx(src, engine)
        elif ext == ".pptx":
            blocks, stats = parse_pptx(src, engine)
        else:
            blocks = ocr_lines_to_blocks(engine.run(src))
            stats = {"ocr_images": 1}
    blocks, removed = remove_repeated_page_noise(blocks)
    reconstruct_hierarchy(blocks)
    # Assets get their final name <block_id>.png now that ids exist; drop orphaned tmp files.
    for b in blocks:
        tmp = getattr(b, "_tmp_asset", None)
        if tmp and (dest / "assets" / tmp).exists():
            (dest / "assets" / tmp).replace(dest / "assets" / f"{b.id}.png")
            b.asset = f"assets/{b.id}.png"
    for f in (dest / "assets").glob("tmp_p*.png"):
        f.unlink()
    stats["noise_blocks_removed"] = removed
    stats["blocks"] = len(blocks)
    stats["blocks_by_type"] = dict(Counter(b.type for b in blocks))
    if not blocks:
        raise RuntimeError("Không trích xuất được block nào; xem lại tài liệu hoặc OCR model")
    payload = {"schema_version": 1, "parser_version": PARSER_VERSION,
               "source": {"filename": src.name, "path": str(src),
                          "format": ext.removeprefix("."), "sha256": digest,
                          "size_bytes": src.stat().st_size},
               "stats": stats, "blocks": [asdict(b) for b in blocks]}
    md_path.write_text(markdown_output(blocks), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[parse] SAVED {md_path}\n[parse] SAVED {json_path}\n[parse] {stats}", flush=True)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified block parser PDF/DOCX/PPTX/HTML/MD/TXT/Image")
    ap.add_argument("source", type=Path, help="Tệp đầu vào")
    ap.add_argument("--out-dir", type=Path, default=None, help="Mặc định: <input_parent>/processed/<tên file>")
    ap.add_argument("--force", action="store_true", help="Bỏ qua cache SHA-256")
    args = ap.parse_args(argv)
    execute(args.source, args.out_dir, args.force)


if __name__ == "__main__":
    main()
