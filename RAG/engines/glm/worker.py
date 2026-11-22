"""GLM-OCR worker: JSON lines over stdin/stdout. Ops: formula, table, shutdown. Run with the engine's own venv python."""
import json, os, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
out = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)  # native writes to fd 1 go to stderr, protocol stays on the dup
sys.stdout = sys.stderr  # library prints must not corrupt the protocol
MID = "zai-org/GLM-OCR"
OPS = {"formula": ("Formula Recognition:", 256, "latex"), "table": ("Table Recognition:", 1024, "markdown")}


MAX_BATCH = 8  # measured on 35 crops: 0.98 s/crop single, 0.48 batch 4, 0.32 batch 8, outputs identical


class Repeat:
    """Stop criterion: a sequence ending in one p-token unit (p <= 8) repeated >= max(4, 8/p) times is a loop; one bad crop must not cost 5 s."""
    def __call__(self, input_ids, scores, **kw):
        import torch
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for p in range(1, 9):
            k = max(4, 8 // p)  # same threshold family as plausible_latex 'loop' (8 equal tokens): never truncate legit "0 & 0 & 0"
            if input_ids.shape[1] >= k * p:
                tail = input_ids[:, -k * p:].view(-1, k, p)
                done |= (tail == tail[:, :1]).all(2).all(1)
        return done


def send(o):
    out.write(json.dumps(o, ensure_ascii=False) + "\n"); out.flush()


def main():
    try:
        import torch
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        t0 = time.perf_counter()
        proc = AutoProcessor.from_pretrained(MID)
        proc.tokenizer.padding_side = "left"
        model = GlmOcrForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to("cuda").eval()
        send({"ready": True, "engine": "glm", "version": MID, "max_batch": MAX_BATCH, "load_s": round(time.perf_counter() - t0, 2),
              "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 2)})
    except Exception as e:
        send({"ready": False, "error": repr(e)[:500]}); return
    for line in sys.stdin:
        rid = None
        try:
            req = json.loads(line); rid = req.get("id")
            if req["op"] == "shutdown":
                return
            op = req["op"]
            prompt, max_new, key = OPS["formula" if op == "formulas" else op]
            t = time.perf_counter()
            imgs = req["images"][:MAX_BATCH] if op == "formulas" else [req["image"]]
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}] for im in imgs]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to(model.device)
            inp.pop("token_type_ids", None)
            with torch.inference_mode():
                o = model.generate(**inp, max_new_tokens=max_new, do_sample=False, stopping_criteria=[Repeat()] if key == "latex" else None)
            txts = [proc.decode(x[inp["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
            result = {"latex": txts} if op == "formulas" else {key: txts[0]}
            send({"id": rid, "ok": True, "result": result, "s": round(time.perf_counter() - t, 3)})
        except Exception as e:
            send({"id": rid, "ok": False, "error": repr(e)[:500]})


main()
