import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.client import LLMClient, LLMConfig, UsageMeter  # noqa: E402

KEY = "sk-SECRET-123"
MSG = [{"role": "user", "content": "hi"}]


def sse(*chunks, done=True):
    lines = [b": OPENROUTER PROCESSING\n", b"\n"]
    for c in chunks:
        lines += [b"data: " + json.dumps(c).encode() + b"\n", b"\n"]
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


class Resp:
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def delta(text=None, finish=None):
    return {"choices": [{"delta": {"content": text} if text else {}, "finish_reason": finish}]}


def http_err(code, retry_after=None, body=b"bad"):
    hdrs = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "m", hdrs, io.BytesIO(body))


def make(opener, **env):
    sleeps = []
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY, **env}):
        c = LLMClient(LLMConfig.from_env(), opener=opener, sleep=sleeps.append)
    return c, sleeps


class ClientTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY})
        p.start()
        self.addCleanup(p.stop)

    def test_stream_parse_usage_done(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.001}
        lines = sse(delta("Xin "), delta("chào"), delta(None, "stop"), {"choices": [], "usage": usage})
        c, _ = make(lambda req, timeout: Resp(lines))
        evs = list(c.stream_chat(MSG, stage="t"))
        self.assertEqual([e["text"] for e in evs if e["type"] == "token"], ["Xin ", "chào"])
        f = evs[-1]
        self.assertEqual((f["type"], f["text"], f["finish_reason"], f["truncated"]), ("final", "Xin chào", "stop", False))
        self.assertEqual(f["usage"], usage)
        self.assertIsNotNone(f["t_first_token_s"])
        self.assertEqual(c.meter.summary()["stages"]["t"]["completion_tokens"], 3)

    def test_chat_and_length_truncated(self):
        c, _ = make(lambda req, timeout: Resp(sse(delta("abc"), delta(None, "length"))))
        r = c.chat(MSG)
        self.assertTrue(r["truncated"])
        self.assertEqual(r["finish_reason"], "length")
        self.assertNotIn("type", r)

    def test_request_body_and_headers(self):
        seen = {}

        def opener(req, timeout):
            seen["req"], seen["timeout"] = req, timeout
            return Resp(sse(delta("ok", "stop")))

        c, _ = make(opener)
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        c.chat(MSG, response_format=schema, max_tokens=5)
        body = json.loads(seen["req"].data)
        self.assertEqual(body["response_format"], schema)
        self.assertEqual(body["max_tokens"], 5)
        self.assertEqual(body["model"], "qwen/qwen3.7-flash")
        self.assertEqual(body["reasoning"], {"effort": "none"})  # default: reasoning off (see LLMConfig)
        self.assertTrue(seen["req"].full_url.endswith("/chat/completions"))
        self.assertEqual(seen["req"].get_header("X-title"), "RAG_official")

    def test_429_then_success_retry_after(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise http_err(429, "7")
            return Resp(sse(delta("ok", "stop")))

        c, sleeps = make(opener)
        self.assertEqual(c.chat(MSG)["text"], "ok")
        self.assertEqual(sleeps, [7.0])

    def test_retries_exhausted_and_no_retry_on_4xx(self):
        n = []
        c, sleeps = make(lambda req, timeout: n.append(1) or (_ for _ in ()).throw(http_err(503)))
        with self.assertRaises(RuntimeError):
            c.chat(MSG)
        self.assertEqual(len(n), 5)
        self.assertEqual(len(sleeps), 4)
        n.clear()
        c, _ = make(lambda req, timeout: n.append(1) or (_ for _ in ()).throw(http_err(400)))
        with self.assertRaises(RuntimeError):
            c.chat(MSG)
        self.assertEqual(len(n), 1)

    def test_midstream_error_raises(self):
        lines = sse(delta("a"), {"error": {"message": "boom"}})
        c, _ = make(lambda req, timeout: Resp(lines))
        with self.assertRaises(RuntimeError) as cm:
            c.chat(MSG)
        self.assertIn("boom", str(cm.exception))

    def test_key_never_in_exception_text(self):
        def http_opener(req, timeout):
            raise http_err(401, body=f"bad key {KEY}".encode())

        def net_opener(req, timeout):
            raise urllib.error.URLError(f"cannot reach with {KEY}")

        def stream_opener(req, timeout):
            return Resp(sse({"error": {"message": f"leak {KEY}"}}))

        for op in (http_opener, net_opener, stream_opener):
            c, _ = make(op)
            with self.assertRaises(RuntimeError) as cm:
                c.chat(MSG)
            self.assertNotIn(KEY, str(cm.exception))
            self.assertNotIn(KEY, repr(cm.exception))

    def test_missing_key_error_has_no_secret(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            c = LLMClient(LLMConfig(), opener=lambda r, timeout: Resp([]))
            with self.assertRaises(RuntimeError):
                c.chat(MSG)

    def test_key_var_name_is_configurable(self):
        seen = {}

        def opener(req, timeout):
            seen["auth"] = req.get_header("Authorization")
            return Resp(sse(delta("ok", "stop")))

        with mock.patch.dict(os.environ, {"MY_KEY": "abc"}):
            LLMClient(LLMConfig(api_key_env="MY_KEY"), opener=opener).chat(MSG)
        self.assertEqual(seen["auth"], "Bearer abc")

    def test_image_content_rejected(self):
        called = []
        c, _ = make(lambda req, timeout: called.append(1))
        bad = [{"role": "user", "content": [{"type": "text", "text": "x"},
                                            {"type": "image_url", "image_url": {"url": "data:..."}}]}]
        with self.assertRaises(ValueError):
            c.chat(bad)
        with self.assertRaises(ValueError):
            list(c.stream_chat([{"role": "user", "content": {"a": 1}}]))
        self.assertEqual(called, [])
        ok = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        c2, _ = make(lambda req, timeout: Resp(sse(delta("ok", "stop"))))
        self.assertEqual(c2.chat(ok)["text"], "ok")

    def test_usage_meter_totals(self):
        m = UsageMeter()
        m.record({"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5}, "a")
        m.record({"prompt_tokens": 5, "completion_tokens": 1}, "b")
        m.record(None, "a")
        s = m.summary()
        self.assertEqual((s["calls"], s["prompt_tokens"], s["completion_tokens"], s["cost"]), (3, 15, 3, 0.5))
        self.assertEqual(s["stages"]["a"]["calls"], 2)

    def test_config_defaults_and_env(self):
        d = LLMConfig.from_env({})
        self.assertEqual((d.base_url, d.model, d.api_key_env), ("https://openrouter.ai/api/v1", "qwen/qwen3.7-flash", "OPENROUTER_API_KEY"))
        self.assertEqual((d.timeout, d.max_tokens, d.temperature, d.max_concurrency, d.reasoning_effort), (60, 800, 0.1, 4, "none"))
        self.assertIsNone(LLMConfig.from_env({"RAG_LLM_REASONING_EFFORT": ""}).reasoning_effort)  # empty string turns it off
        self.assertEqual(d.headers, {"HTTP-Referer": "http://localhost", "X-Title": "RAG_official",
                                     "User-Agent": "RAG_official/0.1 (+local; python-urllib)"})
        o = LLMConfig.from_env({"RAG_LLM_BASE_URL": "http://x/v1", "RAG_LLM_MODEL": "m", "RAG_LLM_API_KEY_ENV": "K",
                                "RAG_LLM_REASONING_EFFORT": "low", "RAG_LLM_TIMEOUT": "5",
                                "RAG_LLM_MAX_TOKENS": "9", "RAG_LLM_TEMPERATURE": "0.5",
                                "RAG_LLM_MAX_CONCURRENCY": "2"})
        self.assertEqual((o.base_url, o.model, o.api_key_env, o.reasoning_effort, o.timeout, o.max_tokens, o.temperature, o.max_concurrency),
                         ("http://x/v1", "m", "K", "low", 5, 9, 0.5, 2))


def asse(*events):
    out = []
    for e in events:
        out += [b"event: " + e["type"].encode() + b"\n", b"data: " + json.dumps(e).encode() + b"\n", b"\n"]
    return out


def a_events(stop="end_turn", texts=("Xin ", "chào")):
    return asse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        *[{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}} for t in texts],
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )


AENV = {"RAG_LLM_PROVIDER": "anthropic", "RAG_LLM_MODEL": "claude-x", "ANTHROPIC_API_KEY": KEY}


class AnthropicTests(unittest.TestCase):
    def mk(self, opener, **env):
        sleeps = []
        with mock.patch.dict(os.environ, {**AENV, **env}):
            c = LLMClient(LLMConfig.from_env(), opener=opener, sleep=sleeps.append)
        return c, sleeps

    def setUp(self):
        p = mock.patch.dict(os.environ, AENV)
        p.start()
        self.addCleanup(p.stop)

    def test_parse_request_and_system(self):
        seen = {}

        def opener(req, timeout):
            seen["req"] = req
            return Resp(a_events())

        c, _ = self.mk(opener, ANTHROPIC_BASE_URL="http://bogus.invalid", CLAUDE_CODE_OAUTH_TOKEN="zzz")
        msgs = [{"role": "system", "content": "S1"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "yo"}, {"role": "system", "content": "S2"}]
        evs = list(c.stream_chat(msgs, stage="a"))
        self.assertEqual([e["text"] for e in evs if e["type"] == "token"], ["Xin ", "chào"])
        f = evs[-1]
        self.assertEqual((f["text"], f["finish_reason"], f["truncated"]), ("Xin chào", "stop", False))
        self.assertEqual(f["usage"], {"prompt_tokens": 12, "completion_tokens": 5, "cost": None})
        self.assertIsNotNone(f["t_first_token_s"])
        req = seen["req"]
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(req.get_header("X-api-key"), KEY)
        self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        body = json.loads(req.data)
        self.assertEqual(body["system"], "S1\n\nS2")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
        self.assertTrue(body["stream"])
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]), ("claude-x", 800, 0.1))

    def test_base_url_from_rag_env_and_temperature_none(self):
        seen = {}

        def opener(req, timeout):
            seen["req"] = req
            return Resp(a_events())

        c, _ = self.mk(opener, RAG_LLM_BASE_URL="http://proxy.local/", RAG_LLM_TEMPERATURE="none",
                       ANTHROPIC_BASE_URL="http://bogus.invalid")
        c.chat(MSG)
        self.assertEqual(seen["req"].full_url, "http://proxy.local/v1/messages")
        self.assertNotIn("temperature", json.loads(seen["req"].data))
        self.assertNotIn("system", json.loads(seen["req"].data))

    def test_max_tokens_truncated(self):
        c, _ = self.mk(lambda req, timeout: Resp(a_events(stop="max_tokens")))
        r = c.chat(MSG)
        self.assertEqual((r["finish_reason"], r["truncated"]), ("length", True))

    def test_error_event_raises_and_scrubs(self):
        lines = asse({"type": "error", "error": {"type": "overloaded_error", "message": f"boom {KEY}"}})
        c, _ = self.mk(lambda req, timeout: Resp(lines))
        with self.assertRaises(RuntimeError) as cm:
            c.chat(MSG)
        self.assertIn("boom", str(cm.exception))
        self.assertNotIn(KEY, str(cm.exception))

    def test_http_error_key_scrubbed(self):
        def op(req, timeout):
            raise http_err(401, body=f"bad {KEY}".encode())

        c, _ = self.mk(op)
        with self.assertRaises(RuntimeError) as cm:
            c.chat(MSG)
        self.assertNotIn(KEY, str(cm.exception))

    def test_429_retry(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise http_err(429, "3")
            return Resp(a_events())

        c, sleeps = self.mk(opener)
        self.assertEqual(c.chat(MSG)["text"], "Xin chào")
        self.assertEqual(sleeps, [3.0])

    def test_model_required(self):
        with self.assertRaises(ValueError) as cm:
            LLMConfig.from_env({"RAG_LLM_PROVIDER": "anthropic"})
        self.assertIn("RAG_LLM_MODEL", str(cm.exception))
        with self.assertRaises(ValueError):
            LLMConfig.from_env({"RAG_LLM_PROVIDER": "nope"})
        d = LLMConfig.from_env({"RAG_LLM_PROVIDER": "anthropic", "RAG_LLM_MODEL": "m"})
        self.assertEqual((d.base_url, d.api_key_env), ("https://api.anthropic.com", "ANTHROPIC_API_KEY"))

    def test_cost_from_prices(self):
        c, _ = self.mk(lambda req, timeout: Resp(a_events()), RAG_LLM_PRICE_IN="3", RAG_LLM_PRICE_OUT="15")
        r = c.chat(MSG)
        self.assertAlmostEqual(r["usage"]["cost"], (12 * 3 + 5 * 15) / 1e6)
        self.assertAlmostEqual(c.meter.summary()["cost"], r["usage"]["cost"])

    def test_image_rejected(self):
        called = []
        c, _ = self.mk(lambda req, timeout: called.append(1))
        bad = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}]
        with self.assertRaises(ValueError):
            c.chat(bad)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
