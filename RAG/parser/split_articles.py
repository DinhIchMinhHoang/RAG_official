"""Tach van ban quy che thanh parent (Dieu) / child (Khoan, hoac Diem neu khoan qua dai).

Input : file .md/.txt do parse_pdf.py / parse_docx.py tao ra
Output: parents.jsonl, children.jsonl (cung thu muc voi input)

Chay:  python parser/split_articles.py data/processed/Quy-che-DTDH-3626.md
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

MAX_CHILD_CHARS = 1200  # khoan dai hon nguong nay se tach tiep theo diem
DIEMS = list("abcd") + ["đ"] + list("eghiklmnopqrstuvxy")  # tieng Viet khong co f, j, w, z

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC]+)\s*[.:-]?\s*(.*)$", re.I)
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\s*[.:]\s*(.+)$")
KHOAN_RE = re.compile(r"^(\d+)\s*[.)]\s+(\S.*)$")
DIEM_RE = re.compile(r"^([a-zđ])\)\s+(\S.*)$")
PAGE_NO_RE = re.compile(r"^\d{1,3}$")


def normalize(raw: str) -> list[str]:
    """NFC, bo ky hieu markdown, bo so trang, tra ve danh sach dong sach."""
    out = []
    for line in unicodedata.normalize("NFC", raw).splitlines():
        s = line.strip().replace("\u00a0", " ")
        s = re.sub(r"^#+\s*", "", s)
        s = re.sub(r"\*\*|__", "", s)
        s = re.sub(r"^[-*]\s+(?=\d+[.)]\s|[a-zđ]\)\s|Điều\s|Chương\s)", "", s)
        if not s or s.startswith("<!--") or PAGE_NO_RE.match(s):
            continue
        out.append(s)
    return out


def group_articles(lines: list[str]) -> list[dict]:
    articles = []
    chapter = None
    last_no = 0
    started = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Nhận diện Chương I, II, III...
        mc = CHAPTER_RE.match(line)

        if mc:
            chapter_no = mc.group(1).upper()

            # Chỉ bắt đầu từ Chương I của Quy chế
            if chapter_no == "I":
                started = True

            if started:
                title = ""

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()

                    # Lấy tên chương nếu dòng tiếp theo
                    # không phải Điều hoặc Chương
                    if (
                        not ARTICLE_RE.match(next_line)
                        and not CHAPTER_RE.match(next_line)
                    ):
                        title = next_line
                        i += 1

                chapter = {
                    "no": chapter_no,
                    "title": title
                }

            i += 1
            continue

        # Bỏ qua phần Quyết định ban hành
        if not started:
            i += 1
            continue

        # Nhận diện Điều
        ma = ARTICLE_RE.match(line)

        if ma:
            no = int(ma.group(1))

            # Chỉ nhận Điều theo đúng thứ tự
            if no == last_no + 1:
                last_no = no

                articles.append({
                    "chapter": chapter,
                    "no": no,
                    "title": ma.group(2).strip(),
                    "lines": []
                })

                i += 1
                continue

        # Thêm nội dung vào Điều hiện tại
        if articles:
            articles[-1]["lines"].append(line)

        i += 1

    print(f"[chunking] Tim thay {len(articles)} Dieu")

    return articles


def join(lines: list[str]) -> str:
    text = ""
    for l in lines:
        if not text:
            text = l
        elif l.startswith(("- ", "+ ", "|")):
            text += "\n" + l
        else:
            text += " " + l
    return text


def parse_body(lines: list[str]) -> dict:
    intro, khoans = [], []
    exp_k, exp_d, cur_k, cur_d = 1, 0, None, None
    for line in lines:
        mk = KHOAN_RE.match(line)
        md = DIEM_RE.match(line)
        if mk and int(mk.group(1)) == exp_k:
            cur_k = {"no": exp_k, "lines": [mk.group(2)], "diems": []}
            khoans.append(cur_k)
            exp_k, exp_d, cur_d = exp_k + 1, 0, None
        elif md and cur_k is not None and exp_d < len(DIEMS) and md.group(1) == DIEMS[exp_d]:
            cur_d = {"letter": md.group(1), "lines": [md.group(2)]}
            cur_k["diems"].append(cur_d)
            exp_d += 1
        elif cur_d is not None:
            cur_d["lines"].append(line)
        elif cur_k is not None:
            cur_k["lines"].append(line)
        else:
            intro.append(line)
    return {"intro": intro, "khoans": khoans}


def build(articles: list[dict]):
    parents, children = [], []
    for a in articles:
        pid = f"d{a['no']}"
        ch = a["chapter"] or {"no": "?", "title": ""}
        crumb = f"Chương {ch['no']} > Điều {a['no']}. {a['title']}"
        body = parse_body(a["lines"])

        # parent = ca Dieu, dua vao LLM
        parent_text = [f"Điều {a['no']}. {a['title']}"]
        if body["intro"]:
            parent_text.append(join(body["intro"]))
        for k in body["khoans"]:
            parent_text.append(f"{k['no']}. {join(k['lines'])}")
            parent_text += [f"{d['letter']}) {join(d['lines'])}" for d in k["diems"]]
        parents.append({"parent_id": pid, "chapter": ch["no"], "chapter_title": ch["title"],
                        "article": a["no"], "title": a["title"], "text": "\n".join(parent_text)})

        def add(cid, label, text):
            children.append({"child_id": cid, "parent_id": pid, "chapter": ch["no"],
                             "article": a["no"], "breadcrumb": f"{crumb} > {label}" if label else crumb,
                             "text": text})

        if body["intro"] and body["khoans"]:
            add(f"{pid}_k0", "", join(body["intro"]))
        if not body["khoans"]:
            add(f"{pid}_k0", "", join(body["intro"]) or a["title"])
        for k in body["khoans"]:
            head = join(k["lines"])
            full = head + "".join(f"\n{d['letter']}) {join(d['lines'])}" for d in k["diems"])
            if len(full) <= MAX_CHILD_CHARS or not k["diems"]:
                add(f"{pid}_k{k['no']}", f"Khoản {k['no']}", full)
            else:  # khoan dai: moi diem la 1 child, kem cau dan cua khoan
                for d in k["diems"]:
                    add(f"{pid}_k{k['no']}_{d['letter']}", f"Khoản {k['no']} > Điểm {d['letter']}",
                        f"{head}\n{d['letter']}) {join(d['lines'])}")
    return parents, children


def main(path: str):
    src = Path(path)
    articles = group_articles(normalize(src.read_text(encoding="utf-8")))
    parents, children = build(articles)
    for name, rows in (("parents", parents), ("children", children)):
        with open(src.with_name(f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(parents)} dieu, {len(children)} child -> {src.parent}")
    empty = [p["parent_id"] for p in parents if len(p["text"]) < 80]
    if empty:
        print("Canh bao, Dieu qua ngan (nghi loi OCR/tach):", empty)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/Quy-che-DTDH-3626.md")