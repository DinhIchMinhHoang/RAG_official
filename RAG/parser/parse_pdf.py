"""Backward-compatible PDF CLI; uses parse_document for all five stages.

Run from project root:
  python parser/parse_pdf.py data/Quy-che-DTDH-3626.pdf
  python parser/parse_pdf.py data/Quy-che-DTDH-3626.pdf --force
"""
from pathlib import Path
import argparse
from parse_document import execute


def main():
    ap = argparse.ArgumentParser(description="Parse PDF to Markdown and block JSON")
    ap.add_argument("pdf", nargs="?", type=Path, default=Path("data/Quy-che-DTDH-3626.pdf"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.pdf.suffix.lower() != ".pdf":
        ap.error("Chỉ nhận file PDF; dùng parse_document.py cho định dạng khác")
    execute(args.pdf, args.out_dir, args.force)


if __name__ == "__main__":
    main()
