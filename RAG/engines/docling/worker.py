"""Docling worker (table structure, no OCR): JSON lines over stdin/stdout. Ops: tables (pdf, first, last), shutdown."""
import json, os, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)  # native writes to fd 1 go to stderr, protocol stays on the dup
sys.stdout = sys.stderr  # library prints must not corrupt the protocol


def send(o):
    out.write(json.dumps(o, ensure_ascii=False) + "\n"); out.flush()


def tables(conv, req):
    r = conv.convert(req["pdf"], page_range=(req["first"], req["last"]))
    doc, res = r.document, []
    for t in doc.tables:
        if not t.prov:
            continue
        p = t.prov[0]; pg = doc.pages[p.page_no]
        bb = p.bbox.to_bottom_left_origin(pg.size.height)
        df = t.export_to_dataframe(doc)
        rows = [[str(c) for c in df.columns]] + [[str(c) for c in r_] for r_ in df.values.tolist()]
        res.append({"page": p.page_no, "page_w": float(pg.size.width), "page_h": float(pg.size.height),
                    "bbox": [bb.l, bb.b, bb.r, bb.t], "n_rows": int(t.data.num_rows), "n_cols": int(t.data.num_cols),
                    "markdown": t.export_to_markdown(doc), "rows": rows})
    return res


def main():
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from importlib.metadata import version
        t0 = time.perf_counter()
        o = PdfPipelineOptions(do_ocr=False)
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=o)})
        conv.initialize_pipeline(InputFormat.PDF)  # eager: load layout/table models now, so per-request timeouts cover conversion only
        send({"ready": True, "engine": "docling", "version": version("docling"), "load_s": round(time.perf_counter() - t0, 2), "vram_gb": 0.0})
    except Exception as e:
        send({"ready": False, "error": repr(e)[:500]}); return
    for line in sys.stdin:
        rid = None
        try:
            req = json.loads(line); rid = req.get("id")
            if req["op"] == "shutdown":
                return
            t = time.perf_counter()
            send({"id": rid, "ok": True, "result": tables(conv, req), "s": round(time.perf_counter() - t, 3)})
        except Exception as e:
            send({"id": rid, "ok": False, "error": repr(e)[:500]})


main()
