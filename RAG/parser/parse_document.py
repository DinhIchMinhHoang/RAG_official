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
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

PARSER_VERSION = "2.0.0"
MIN_PDF_TEXT = 20
MAX_OCR_SIDE = 2400
SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".png", ".jpg", ".jpeg"}
CHAPTER = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\b\s*[.:-]?\s*(.*)$", re.I)
ARTICLE = re.compile(r"^Điều\s+(\d+)\s*[.:-]\s*(.*)$", re.I)
CLAUSE = re.compile(r"^(\d{1,2})[.)]\s+(.+)$")
POINT = re.compile(r"^([a-zđ])\)\s+(.+)$", re.I)
BULLET = re.compile(r"^[-*+]\s+(.*)$")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
PAGE_NUMBER = re.compile(r"^(?:trang\s*)?\d{1,3}(?:\s*/\s*\d{1,3})?$", re.I)


def clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ContentBlock:
    id: str
    type: str  # heading | paragraph | table | list | image_ocr
    text: str
    level: int = 0
    location: dict[str, int] = field(default_factory=dict)
    breadcrumb: str = ""
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def make_block(kind: str, text: str, *, page_num: int | None = None,
               slide_num: int | None = None, level: int = 0,
               bbox: list[float] | None = None, **metadata: Any) -> ContentBlock:
    location = {"page_num": page_num} if page_num is not None else (
        {"slide_num": slide_num} if slide_num is not None else {})
    return ContentBlock(id="", type=kind, text=clean(text), level=min(max(level, 0), 3),
                        location=location, bbox=bbox, metadata=metadata)


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
    """Parse headings, paragraphs, list items and Markdown tables in source order."""
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


class OCR:
    def __init__(self):
        import pytesseract

        self.engine = pytesseract
        self.engine.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def run(self, data):
        from pytesseract import Output

        img = image_to_pil(data)
        w, h = img.size

        result = self.engine.image_to_data(
            img,
            lang="vie+eng",
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


def get_pdf_text_blocks(page: Any, page_num: int) -> list[ContentBlock]:
    """PyMuPDF4LLM for born-digital page; fitz used only for image/order metadata."""
    import pymupdf4llm
    # Disable PyMuPDF4LLM automatic OCR: the RapidOCR pass below decides that.
    try:
        md = pymupdf4llm.to_markdown(page.parent, pages=[page.number],
                                     page_chunks=False, use_ocr=False,
                                     show_progress=False, write_images=False)
    except TypeError:  # some versions have no use_ocr keyword
        md = pymupdf4llm.to_markdown(page.parent, pages=[page.number],
                                     page_chunks=False, show_progress=False,
                                     write_images=False)
    blocks = parse_markdown_blocks(md, page_num=page_num, source="pymupdf4llm")
    # Map blocks to natural page coordinates, allowing image OCR to be interleaved.
    text_boxes = [list(x[:4]) for x in page.get_text("blocks", sort=True) if x[6] == 0]
    for i, block in enumerate(blocks):
        # Bounding box matching is approximate for multi-line Markdown/table layout.
        probe = re.sub(r"\s+", " ", block.text).strip()[:55]
        found = []
        if probe and block.type != "table":
            try:
                found = page.search_for(probe)
            except Exception:
                pass
        if found:
            block.bbox = [found[0].x0/page.rect.width, found[0].y0/page.rect.height,
                          found[-1].x1/page.rect.width, found[-1].y1/page.rect.height]
        elif text_boxes:
            bbox = text_boxes[min(i, len(text_boxes)-1)]
            block.bbox = [bbox[0]/page.rect.width, bbox[1]/page.rect.height,
                          bbox[2]/page.rect.width, bbox[3]/page.rect.height]
    return blocks


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


def parse_pdf(path: Path, ocr: OCR) -> tuple[list[ContentBlock], dict[str, Any]]:
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
                b = get_pdf_text_blocks(p, idx)
                counts["digital_pages"] += 1
                # Recover OCR text from embedded images in page order.
                images = p.get_text("dict", sort=True)["blocks"]
                for item in images:
                    if item.get("type") != 1 or not item.get("image"):
                        continue
                    x0,y0,x1,y1 = item["bbox"]
                    if (x1-x0)*(y1-y0) < p.rect.width*p.rect.height*0.008:
                        continue
                    boxes = ocr_lines_to_blocks(ocr.run(item["image"]), page_num=idx)
                    for block in boxes:
                        # Embedded picture yields image_ocr, including inside scanned tables.
                        block.type = "image_ocr" if block.type not in ("heading", "list") else block.type
                        if block.bbox:
                            a,b0,c,d = block.bbox
                            block.bbox = [round((x0 + a*(x1-x0))/p.rect.width, 5),
                                          round((y0 + b0*(y1-y0))/p.rect.height, 5),
                                          round((x0 + c*(x1-x0))/p.rect.width, 5),
                                          round((y0 + d*(y1-y0))/p.rect.height, 5)]
                    b.extend(boxes)
                    counts["image_regions_ocr"] += 1
                b.sort(key=lambda x: ((x.bbox or [0, 0])[1], (x.bbox or [0, 0])[0]))
            if not b:
                counts["empty_pages"] += 1
            out.extend(b)
            print(f"[parse] Trang {idx}/{len(doc)}: {'RapidOCR' if scanned else 'PyMuPDF4LLM'} - {len(b)} blocks", flush=True)
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
            out.append(make_block("table", table_to_markdown(rows), source="python-docx"))
    return out, {}


def parse_pptx(path: Path, ocr: OCR) -> tuple[list[ContentBlock], dict[str, Any]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    slides = Presentation(str(path)).slides
    out = []
    def shapes_in_reading_order(shapes):
        return sorted(shapes, key=lambda s: (s.top, s.left))
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
        if shape.has_table:
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            out.append(make_block("table", table_to_markdown(rows), slide_num=num, source="python-pptx"))
            return
        if shape.has_text_frame:
            for i, para in enumerate(shape.text_frame.paragraphs):
                text = clean(para.text)
                if not text:
                    continue
                is_bullet = bool(para.level or para._p.xpath("./a:pPr/a:buChar") or para._p.xpath("./a:pPr/a:buAutoNum"))
                kind, lvl, txt = detect_type(text, "list" if is_bullet else "paragraph")
                if i == 0 and not is_bullet and para.level == 0 and len(text) < 100:
                    sizes = [r.font.size.pt for r in para.runs if r.font.size]
                    if sizes and max(sizes) >= 24 and kind == "paragraph":
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
        if name in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "img", "pre", "blockquote"):
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
                result.append(make_block("table", table_to_markdown(rows), source="beautifulsoup4"))
                return
            if name in ("p", "li") and el.find(["img", "table"]):
                # Preserve inline images relative to the surrounding text.
                inline_text: list[str] = []
                def visit_inline(child):
                    if isinstance(child, Tag):
                        if child.name == "img":
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
        if not pnum or not b.bbox or b.type == "heading" or len(b.text) > 100:
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
        if (b.type != "heading" and near_edge and
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
        dest = src.parent / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    md_path = dest / f"{src.stem}.md"
    json_path = dest / f"{src.stem}.json"
    digest = sha256(src)
    if not force and md_path.exists() and json_path.exists():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("source", {}).get("sha256") == digest and cached.get("parser_version") == PARSER_VERSION:
                print(f"[parse] CACHE HIT sha256={digest[:12]} | {json_path}")
                return cached
        except (OSError, ValueError):
            pass
    print(f"[parse] START {src.name} | sha256={digest[:12]}", flush=True)
    ext = src.suffix.lower()
    if ext in {".md", ".txt"}:
        blocks, stats = parse_plain(src)
    elif ext == ".html" or ext == ".htm":
        blocks, stats = parse_html(src, ocr or OCR())
    else:
        engine = ocr or OCR()  # Lazy: import OCR model only when necessary.
        if ext == ".pdf":
            blocks, stats = parse_pdf(src, engine)
        elif ext == ".docx":
            blocks, stats = parse_docx(src, engine)
        elif ext == ".pptx":
            blocks, stats = parse_pptx(src, engine)
        else:
            blocks = ocr_lines_to_blocks(engine.run(src))
            stats = {"ocr_images": 1}
    blocks, removed = remove_repeated_page_noise(blocks)
    reconstruct_hierarchy(blocks)
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
    ap.add_argument("--out-dir", type=Path, default=None, help="Mặc định: <input_parent>/processed")
    ap.add_argument("--force", action="store_true", help="Bỏ qua cache SHA-256")
    args = ap.parse_args(argv)
    execute(args.source, args.out_dir, args.force)


if __name__ == "__main__":
    main()
