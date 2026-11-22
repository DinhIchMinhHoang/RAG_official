"""Fake engine worker (stdlib only) for tests. Env: FAKE_SLEEP, FAKE_CRASH_AT, FAKE_BAD_JSON, FAKE_LOG."""
import json, os, sys, time

out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)  # native writes to fd 1 go to stderr, protocol stays on the dup
sys.stdout = sys.stderr
if os.environ.get("FAKE_STRAY_FD1"):
    os.write(1, b"native noise\n")  # simulates a C library printing to fd 1


def send(o):
    out.write(json.dumps(o, ensure_ascii=False) + "\n"); out.flush()


TABLE = {"page": 1, "page_w": 612.0, "page_h": 792.0, "bbox": [0.0, 0.0, 612.0, 792.0], "n_rows": 2, "n_cols": 2,
         "markdown": "| a | b |\n| --- | --- |\n| 1 | 2 |", "rows": [["a", "b"], ["1", "2"]]}


def main():
    time.sleep(float(os.environ.get("FAKE_BOOT", 0)))  # simulated spawn+model load before ready
    send({"ready": True, "engine": os.environ.get("FAKE_ENGINE", "fake"), "version": "fake", "load_s": 0.0, "vram_gb": 0.0,
          "max_batch": int(os.environ.get("FAKE_MAX_BATCH", 1)), "boot_s": float(os.environ.get("FAKE_BOOT", 0))})
    crash, bad, n = int(os.environ.get("FAKE_CRASH_AT", 0)), int(os.environ.get("FAKE_BAD_JSON", 0)), 0
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("op") == "shutdown":
            return
        n += 1
        if os.environ.get("FAKE_LOG"):
            with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
                f.write(line.strip() + "\n")
        if crash and n == crash:
            os._exit(3)
        if bad and n == bad:
            out.write("garbage{{\n"); out.flush(); continue
        time.sleep(float(os.environ.get("FAKE_SLEEP", 0)))
        op = req["op"]
        if op == "formula":
            res = {"latex": "\\frac{a}{b}"}
        elif op == "formulas":
            res = {"latex": ["\\frac{a}{b}"] * len(req["images"])}
        elif op == "table":
            res = {"markdown": TABLE["markdown"]}
        else:
            res = [dict(TABLE, page=req.get("first", 1))]
        send({"id": req["id"], "ok": True, "result": res, "s": 0.0})


main()
