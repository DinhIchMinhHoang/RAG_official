"""CLI demo: ingest / ask / chat / info.   Chay (cwd RAG):  python cli.py --help
Exit codes: 0 ok, 1 loi runtime, 2 loi nguoi dung / tai lieu that bai / thieu cau hinh.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest as ing  # noqa: E402
from llm.answering import MAX_HISTORY_TURNS, REFUSAL_TEXT, assemble_contexts, build_messages, parse_answer  # noqa: E402
from llm.client import LLMClient, LLMConfig  # noqa: E402


def get_counter():
    from parser.chunker import default_token_counter
    return default_token_counter()


def open_collection(store, device=None, backend="faiss"):
    if backend == "qdrant":
        from embed.qdrant_collection import QdrantCollection
        return QdrantCollection.open(ing.qdrant_name(store), client=ing.connect_qdrant(), device=device)
    from embed.collection import Collection
    idx = Path(store) / "index"
    if not (idx / "manifest.json").exists():
        raise FileNotFoundError(f"Chưa có index tại {idx}. Chạy trước: python cli.py ingest <tệp...> --store {store}")
    return Collection.open(idx, device=device)


def api_key_error(cfg):
    if os.environ.get(cfg.api_key_env):
        return None
    return (f"Thiếu biến môi trường {cfg.api_key_env} (API key LLM). Đặt biến này, hoặc dùng --no-llm "
            "để chỉ xem đoạn truy xuất.")


# ---- ask ----
def fmt_pages(ps, pe):
    if ps is None:
        return "không có số trang"  # DOCX/HTML/MD have no pages
    return f"tr.{ps}" if pe is None or ps == pe else f"tr.{ps}-{pe}"


def clean_title(t):
    return re.sub(r"-[0-9a-f]{6}$", "", t or "")  # drop the doc_id hash suffix


MARKERS = {"figure": "[hình]", "formula": "[công thức]", "code": "[mã]", "table": "[bảng]"}


def _src_line(c):
    mk = "".join(f" {MARKERS[t]}" for t in c.get("types") or ())
    return (f"[{c['source_id']}] {clean_title(c.get('title') or c.get('doc_id'))} (doc {c.get('doc_id')}) "
            f"{fmt_pages(c.get('page_start'), c.get('page_end'))}{mk} | {c.get('breadcrumb') or ''}")


def source_extras(contexts, parents, store):
    """Per context: content `types` / `assets` / `languages` of its matched children, read from
    <store>/corpus/children.jsonl (shortcut: re-read per ask; search() results carry no child metadata)."""
    f = Path(store) / "corpus" / "children.jsonl" if store else None
    kids = {k["child_id"]: k for k in ing._read_jsonl(f)} if f else {}
    by_parent = {p["parent_id"]: p.get("matched_child_ids") or [] for p in parents}
    out = []
    for c in contexts:
        ks = [kids[i] for i in by_parent.get(c["parent_id"], []) if i in kids]
        assets = [x for k in ks for x in k.get("assets") or []]
        langs = sorted({x for k in ks for x in k.get("languages") or []})
        have = {k.get("content_type") for k in ks} | {x["type"] for x in assets} | ({"code"} if langs else set())
        out.append({**({"types": [t for t in MARKERS if t in have]} if have & MARKERS.keys() else {}),
                    **({"assets": assets} if assets else {}), **({"languages": langs} if langs else {})})
    return out


def resolve_rerank(mode, n_docs):
    """on/off/auto -> bool. auto = on only for a multi-document corpus (single doc: no measured gain)."""
    return mode == "on" or (mode == "auto" and n_docs >= 2)


def load_hint(r, threshold_ms=3000):
    if (r["timings_ms"].get("embed") or 0) > threshold_ms:
        return "Gợi ý: nhúng câu hỏi mất >3 s do nạp mô hình; dùng lệnh chat để giữ mô hình nóng."
    return None


def search_docs(col, question, a, rerank):
    """--doc restriction: native filter on qdrant; faiss Collection has none, so post-filter parents by doc_id
    (shortcut: candidates/rerank pool are chosen over the whole corpus, then cut; fine for small corpora)."""
    kw = dict(top_k=a.top, fusion=a.fusion, alpha=a.alpha, rerank=rerank, max_parents=a.max_parents)
    docs = set(getattr(a, "doc", None) or ())
    if not docs:
        return col.search(question, **kw)
    if getattr(a, "backend", "faiss") == "qdrant":
        return col.search(question, filter_doc_ids=sorted(docs), **kw)
    res = col.search(question, **{**kw, "max_parents": 10 ** 6})
    parents = [p for p in res["parents"] if p["doc_id"] in docs][:a.max_parents]
    keep = {c for p in parents for c in p["matched_child_ids"]}
    children = [c for c in res["children"] if c["child_id"] in keep]
    top = children[0] if children else None
    res.update(parents=parents, children=children, top1={
        "dense_cosine": top["scores"]["dense"] if top else None, "fused": top["score"] if top else None,
        "margin": None})
    return res


def ask_once(question, col, llm, a, counter, history=None, out=None):
    """One retrieval + answer round. Prints (unless --json) and returns the result dict."""
    out = out or sys.stdout
    quiet = getattr(a, "json", False)
    p = lambda s="": None if quiet else print(s, file=out)
    t0 = time.perf_counter()
    n_docs = len(set(a.doc)) if getattr(a, "doc", None) else getattr(a, "n_docs", 0)
    do_rerank = resolve_rerank(a.rerank, n_docs)
    mode = ("on" if do_rerank else "off") + (" (auto)" if a.rerank == "auto" else "")
    res = search_docs(col, question, a, do_rerank)
    tm = res["timings_ms"]
    r = {"question": question, "top1": res["top1"], "rerank_mode": mode, "answer": None, "refused": False,
         "gated": False,
         "invalid_citations": [], "truncated": False, "citations": [], "sources": [], "passages": [],
         "usage": None, "timings_ms": {"embed": tm.get("embed"), "rerank": tm.get("rerank"),
                                       "search": tm["total"] - tm.get("embed", 0) - tm.get("rerank", 0)}}
    if a.no_llm:
        for i, pr in enumerate(res["parents"], 1):
            r["passages"].append({"rank": i, "score": pr["score"], "doc_id": pr["doc_id"], "title": pr["title"],
                                  "page_start": pr["page_start"], "page_end": pr["page_end"],
                                  "breadcrumb": pr["breadcrumb"], "text": pr["text"][:200]})
            p(f"#{i} score={pr['score']:.4f} doc={pr['doc_id']} {fmt_pages(pr['page_start'], pr['page_end'])}\n"
              f"   {pr['text'][:200]!r}")
        r["timings_ms"]["total"] = (time.perf_counter() - t0) * 1000
        p(f"\n[embed {r['timings_ms']['embed']:.0f} ms | search {r['timings_ms']['search']:.0f} ms | "
          f"rerank {mode} {r['timings_ms']['rerank']:.0f} ms]")
        return r
    cos = res["top1"]["dense_cosine"]
    if not res["parents"] or (a.min_cosine is not None and (cos is None or cos < a.min_cosine)):
        r.update(answer=REFUSAL_TEXT, refused=True, gated=True)
        p(REFUSAL_TEXT)
        p(f"\n(Chặn cục bộ: cosine top1={cos} < --min-cosine={a.min_cosine}; không gọi LLM.)" if res["parents"]
          else "\n(Không truy xuất được đoạn nào; không gọi LLM.)")
    else:
        contexts = assemble_contexts(res["parents"], counter, a.budget_tokens)
        extras = source_extras(contexts, res["parents"], getattr(a, "store", None))
        final = None
        for ev in llm.stream_chat(build_messages(question, contexts, history), stage="answer"):
            if ev["type"] == "token":
                if not quiet:
                    out.write(ev["text"])
                    out.flush()
            else:
                final = ev
        p()
        parsed = parse_answer(final["text"], contexts)
        cited = {c["source_id"] for c in parsed["citations"]}
        r.update(answer=parsed["answer"], refused=parsed["refused"], citations=parsed["citations"],
                 invalid_citations=parsed["invalid_citations"], truncated=bool(final.get("truncated")),
                 usage=final.get("usage"), model=getattr(getattr(llm, "config", None), "model", None),
                 sources=[{"source_id": c["source_id"], "doc_id": c["doc_id"], "title": c["title"],
                           "page_start": c["page_start"], "page_end": c["page_end"],
                           "breadcrumb": c["breadcrumb"], "cited": c["source_id"] in cited,
                           "truncated": c["truncated"], **x} for c, x in zip(contexts, extras)])
        r["timings_ms"]["ttft"] = None if final.get("t_first_token_s") is None else final["t_first_token_s"] * 1000
        p("\nNguồn:")
        for c, x in zip(contexts, extras):
            p(f"  {_src_line({**c, **x})}{'  (đã trích dẫn)' if c['source_id'] in cited else ''}")
        p(f"\nCờ: refused={r['refused']} invalid_citations={r['invalid_citations']} truncated={r['truncated']}")
    r["timings_ms"]["total"] = (time.perf_counter() - t0) * 1000
    t, u = r["timings_ms"], r["usage"] or {}
    ttft = "-" if t.get("ttft") is None else f"{t['ttft']:.0f} ms"
    cost = f" | cost ${u['cost']}" if u.get("cost") is not None else ""
    p(f"[embed {t['embed']:.0f} ms | search {t['search']:.0f} ms | rerank {mode} {t['rerank']:.0f} ms | ttft {ttft} | "
      f"total {t['total']:.0f} ms | tokens {u.get('prompt_tokens', 0)}+{u.get('completion_tokens', 0)}{cost}]")
    return r


def _setup_retrieval(a, err):
    """-> (collection, llm|None, counter|None) or None after printing an error."""
    cfg = LLMConfig.from_env()
    if not a.no_llm and (msg := api_key_error(cfg)):
        print(msg, file=err)
        return None
    try:
        col = open_collection(a.store, getattr(a, "device", None), a.backend)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"Lỗi store: {e}", file=err)
        return None
    a.n_docs = len(ing.load_registry(a.store))
    return col, (None if a.no_llm else LLMClient(cfg)), (None if a.no_llm else get_counter())


def cmd_ask(a, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    setup = _setup_retrieval(a, err)
    if not setup:
        return 2
    col, llm, counter = setup
    try:
        r = ask_once(a.question, col, llm, a, counter, out=out)
    except RuntimeError as e:
        print(f"Lỗi LLM: {e}", file=err)
        return 1
    finally:
        col.close()
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2), file=out)
    elif hint := load_hint(r):
        print(hint, file=out)
    return 0


def cmd_chat(a, out=None, err=None, input_fn=input):
    out, err = out or sys.stdout, err or sys.stderr
    setup = _setup_retrieval(a, err)
    if not setup:
        return 2
    col, llm, counter = setup
    a.json = False
    history, last, asked = [], None, 0
    print("Chat RAG. Lệnh: /sources, /stats, /quit", file=out)
    try:
        while True:
            try:
                q = input_fn("> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q == "/quit":
                break
            if q == "/sources":
                for s in (last or {}).get("sources", []) or [{"source_id": "-", "title": "chưa có nguồn"}]:
                    print(_src_line(s), file=out)
            elif q == "/stats":
                u = llm.meter.summary() if llm else {}
                print(f"lượt hỏi={asked} lịch sử={len(history)} tin nhắn | LLM calls={u.get('calls', 0)} "
                      f"tokens={u.get('prompt_tokens', 0)}+{u.get('completion_tokens', 0)} cost=${u.get('cost', 0):.5f}",
                      file=out)
            else:
                try:
                    last = ask_once(q, col, llm, a, counter, history, out)
                except RuntimeError as e:
                    print(f"Lỗi LLM: {e}", file=err)
                    continue
                asked += 1
                if last["answer"]:
                    history = (history + [{"role": "user", "content": q},
                                          {"role": "assistant", "content": last["answer"]}])[-MAX_HISTORY_TURNS:]
    except KeyboardInterrupt:
        pass
    finally:
        col.close()
    return 0


# ---- ingest ----
def cmd_ingest(a, out=None, err=None):
    out = out or sys.stdout
    emit = lambda s: print(s, file=out)
    if a.tessdata:
        os.environ["TESSDATA_PREFIX"] = str(a.tessdata)
    files = ing.expand_paths(a.paths)
    if not files:
        print("Không có tệp nào để ingest.", file=err or sys.stderr)
        return 2
    client = None
    if a.backend == "qdrant":  # fail fast, before the slow parse/embed
        try:
            client = ing.connect_qdrant()
        except ConnectionError as e:
            print(f"Lỗi Qdrant: {e}", file=err or sys.stderr)
            return 2
    results = ing.ingest_paths(files, a.store, get_counter(), a.ceiling, a.force, out=emit)
    changed = any(r["status"] == "ok" for r in results)
    index_s = 0.0
    try:
        rebuilt, index_s = ing.rebuild_index(a.store, a.device, force=changed, backend=a.backend, client=client)
    except Exception as e:
        emit(f"!!! THẤT BẠI build index: {type(e).__name__}: {e}")
        return 2
    emit(f"Index: {'đã build lại' if rebuilt else 'không đổi (bỏ qua)'}")
    if a.stack and changed:
        ing.mirror_stack(results, out=emit)
    emit(ing.format_table(results, index_s))
    return 2 if any(r["status"] == "failed" for r in results) else 0


# ---- info ----
def cmd_info(a, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    p = lambda s="": print(s, file=out)
    store = Path(a.store)
    if not store.exists():
        print(f"Không thấy store {store}. Chạy: python cli.py ingest <tệp...> --store {store}", file=err)
        return 2
    reg = ing.load_registry(store)
    p(f"Tài liệu ({len(reg)}):")
    for d, v in reg.items():
        p(f"  {d}  {v['filename']}  trang={v['pages']} parents={v['parents']} children={v['children']}")
    m = store / "index" / "manifest.json"
    p(f"Backend: {a.backend}")
    p("Index FAISS manifest: " + (json.dumps({k: v for k, v in json.loads(m.read_text(encoding="utf-8")).items()
                                              if k != "sha256"}, ensure_ascii=False) if m.exists() else "chưa có"))
    if a.backend == "qdrant":
        name = ing.qdrant_name(store)
        try:
            client = ing.connect_qdrant()
            for n in (name, name + "_parents"):
                p(f"Qdrant {n}: " + (f"{client.count(n, exact=True).count} điểm" if client.collection_exists(n)
                                     else "chưa có (chạy ingest --backend qdrant)"))
        except ConnectionError as e:
            print(f"Lỗi Qdrant: {e}", file=err)
            return 2
    cfg = LLMConfig.from_env()
    p(f"LLM: model={cfg.model} base_url={cfg.base_url} api_key_env={cfg.api_key_env} "
      f"({'đã đặt' if os.environ.get(cfg.api_key_env) else 'CHƯA đặt'})")
    try:
        from store.settings import StoreSettings
        p(f"Store settings: {StoreSettings.from_env().redacted()}")
    except Exception as e:
        p(f"Store settings: không đọc được ({e})")
    try:
        import torch
        p(f"GPU: {torch.cuda.is_available()}")
    except ImportError:
        p("GPU: không rõ (torch chưa cài)")
    p("Model đã tải: không (embedder/reranker tải lười khi chạy ask/chat)")
    return 0


# ---- parser ----
def build_parser():
    ap = argparse.ArgumentParser(prog="cli.py", description="RAG demo: ingest / ask / chat / info")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="parse + chunk + embed tài liệu vào store")
    i.add_argument("paths", nargs="+", type=Path, help="tệp hoặc thư mục")
    i.add_argument("--store", required=True, type=Path)
    i.add_argument("--device", choices=["cuda", "cpu"], default=None)
    i.add_argument("--ceiling", type=int, default=400, help="trần token mỗi child")
    i.add_argument("--tessdata", type=Path, default=None, help="thư mục tessdata (TESSDATA_PREFIX) cho OCR")
    i.add_argument("--force", action="store_true", help="bỏ qua cache sha256, ingest lại")
    i.add_argument("--stack", action="store_true", help="đồng bộ thêm vào Postgres + MinIO nếu sẵn sàng")
    backend = lambda p: p.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss",
                                       help="faiss (tệp cục bộ) | qdrant (docker compose up -d)")
    backend(i)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", required=True, type=Path)
    backend(common)
    common.add_argument("--doc", action="append", default=None, help="chỉ truy xuất trong doc_id này (lặp lại được)")
    common.add_argument("--device", choices=["cuda", "cpu"], default=None)
    common.add_argument("--top", type=int, default=5, help="số child trả về")
    common.add_argument("--max-parents", type=int, default=3)
    common.add_argument("--fusion", choices=["rrf", "convex"], default="rrf")
    common.add_argument("--alpha", type=float, default=0.7, help="trọng số dense cho fusion convex")
    common.add_argument("--rerank", nargs="?", const="on", choices=["on", "off", "auto"], default="auto",
                        help="rerank bge-reranker-v2-m3: auto (mặc định) = bật khi store có >= 2 tài liệu")
    common.add_argument("--no-llm", action="store_true", help="chỉ in đoạn truy xuất")
    common.add_argument("--min-cosine", type=float, default=None, help="cosine dense top1 < X thì từ chối, không gọi LLM")
    common.add_argument("--budget-tokens", type=int, default=3500)
    k = sub.add_parser("ask", parents=[common], help="hỏi một câu")
    k.add_argument("question")
    k.add_argument("--json", action="store_true", help="in một object JSON")
    sub.add_parser("chat", parents=[common], help="REPL hội thoại (/sources /stats /quit)")
    n = sub.add_parser("info", help="thông tin store")
    n.add_argument("--store", required=True, type=Path)
    backend(n)
    return ap


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    return {"ingest": cmd_ingest, "ask": cmd_ask, "chat": cmd_chat, "info": cmd_info}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
