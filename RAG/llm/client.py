"""Text-only OpenAI-compatible chat client (streaming, retries, usage meter). Stdlib only.

Config comes from RAG_LLM_* environment variables; the API key is read only from the
variable NAMED by RAG_LLM_API_KEY_ENV and is scrubbed from every error text.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional

MAX_RETRIES = 4


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai" (OpenAI-compatible chat/completions) | "anthropic" (Messages API)
    base_url: str = "https://openrouter.ai/api/v1"
    # Cheapest suitable model on OpenRouter (~$0.00004 per RAG query measured 2026-11-22). Reasoning MUST be off:
    # with the default reasoning it spent all 800 output tokens on hidden thinking and returned an empty answer.
    model: str = "qwen/qwen3.7-flash"
    api_key_env: str = "OPENROUTER_API_KEY"
    reasoning_effort: Optional[str] = "none"
    timeout: float = 60.0
    max_tokens: int = 800
    temperature: Optional[float] = 0.1
    price_in: Optional[float] = None  # USD per million tokens (anthropic cost only)
    price_out: Optional[float] = None
    max_concurrency: int = 4
    headers: Dict[str, str] = field(
        # Explicit User-Agent: Cloudflare in front of some providers (OpenCode) answers 403 / error 1010 to urllib's default one.
        default_factory=lambda: {"HTTP-Referer": "http://localhost", "X-Title": "RAG_official",
                                 "User-Agent": "RAG_official/0.1 (+local; python-urllib)"}
    )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        e = os.environ if env is None else env
        d = cls()
        provider = (e.get("RAG_LLM_PROVIDER") or d.provider).strip().lower()
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"RAG_LLM_PROVIDER must be 'openai' or 'anthropic', got {provider!r}")
        anth = provider == "anthropic"
        model = e.get("RAG_LLM_MODEL") or (None if anth else d.model)
        if not model:
            raise ValueError("RAG_LLM_MODEL is required for provider anthropic: set it to the Claude model id you have access to")
        temp = (e.get("RAG_LLM_TEMPERATURE") or "").strip().lower()
        price = lambda n: float(e[n]) if e.get(n) else None  # noqa: E731
        return cls(
            provider=provider,
            # anthropic: base URL only from RAG_LLM_BASE_URL, never ANTHROPIC_BASE_URL
            base_url=e.get("RAG_LLM_BASE_URL") or ("https://api.anthropic.com" if anth else d.base_url),
            model=model,
            api_key_env=e.get("RAG_LLM_API_KEY_ENV") or ("ANTHROPIC_API_KEY" if anth else d.api_key_env),
            reasoning_effort=(e["RAG_LLM_REASONING_EFFORT"] if "RAG_LLM_REASONING_EFFORT" in e else d.reasoning_effort) or None,
            timeout=float(e.get("RAG_LLM_TIMEOUT") or d.timeout),
            max_tokens=int(e.get("RAG_LLM_MAX_TOKENS") or d.max_tokens),
            temperature=None if temp == "none" else float(temp or d.temperature),
            price_in=price("RAG_LLM_PRICE_IN"),
            price_out=price("RAG_LLM_PRICE_OUT"),
            max_concurrency=int(e.get("RAG_LLM_MAX_CONCURRENCY") or d.max_concurrency),
        )


class UsageMeter:
    """Thread-safe token/call/cost accounting with per-stage labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: Dict[str, Dict[str, float]] = {}

    def record(self, usage: Optional[Mapping[str, Any]], stage: str = "default") -> None:
        u = usage or {}
        with self._lock:
            s = self._stages.setdefault(
                stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
            )
            s["calls"] += 1
            s["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            s["completion_tokens"] += int(u.get("completion_tokens") or 0)
            s["cost"] += float(u.get("cost") or 0.0)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            stages = {k: dict(v) for k, v in self._stages.items()}
        total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        for v in stages.values():
            for k in total:
                total[k] += v[k]
        return {**total, "stages": stages}


def check_text_only(messages: List[Mapping[str, Any]]) -> None:
    """POLICY: only plain text reaches the API; anything else (image_url...) is refused."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            continue
        if isinstance(c, list) and all(
            isinstance(p, Mapping) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in c
        ):
            continue
        raise ValueError("text-only client: non-text message content is refused")


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, opener=None, sleep=time.sleep) -> None:
        self.config = config or LLMConfig.from_env()
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep
        self._sem = threading.Semaphore(max(1, self.config.max_concurrency))
        self.meter = UsageMeter()

    def _key(self) -> str:
        key = os.environ.get(self.config.api_key_env, "")
        if not key:
            raise RuntimeError(f"API key variable {self.config.api_key_env} is not set")
        return key

    def _scrub(self, text: Any) -> str:
        text = str(text)
        key = os.environ.get(self.config.api_key_env, "")
        return text.replace(key, "***") if key else text

    def _connect(self, body: Dict[str, Any]):
        """POST with retries on 429/5xx/connection errors (before any byte is consumed)."""
        key = self._key()
        if self.config.provider == "anthropic":
            url = self.config.base_url.rstrip("/") + "/v1/messages"
            hdr = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        else:
            url = self.config.base_url.rstrip("/") + "/chat/completions"
            hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", **self.config.headers}
            if "opencode.ai" in self.config.base_url:  # OpenCode refuses requests without a session id (400 MissingSessionID)
                self._session_id = getattr(self, "_session_id", None) or uuid.uuid4().hex
                hdr.setdefault("x-opencode-session", self._session_id)
        data = json.dumps(body).encode("utf-8")
        for attempt in range(MAX_RETRIES + 1):
            wait = 2 ** attempt
            try:
                req = urllib.request.Request(url, data=data, headers=hdr)
                return self._open(req, timeout=self.config.timeout)
            except urllib.error.HTTPError as e:
                msg = self._scrub(e.read().decode("utf-8", "replace"))[:300]
                if e.code != 429 and e.code < 500 or attempt == MAX_RETRIES:
                    raise RuntimeError(f"HTTP {e.code}: {msg}") from None
                try:
                    wait = max(wait, float((e.headers or {}).get("Retry-After")))
                except (TypeError, ValueError):
                    pass
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"network: {self._scrub(e)}") from None
            self._sleep(wait)

    def stream_chat(self, messages, *, stage: str = "default", response_format=None,
                    max_tokens=None, temperature=None, reasoning_effort=None) -> Iterator[Dict[str, Any]]:
        check_text_only(messages)
        c = self.config
        if c.provider == "anthropic":
            yield from self._stream_anthropic(messages, stage, max_tokens, temperature)
            return
        body: Dict[str, Any] = {
            "model": c.model, "stream": True, "stream_options": {"include_usage": True},
            "messages": messages,
            "max_tokens": c.max_tokens if max_tokens is None else max_tokens,
        }
        if (c.temperature if temperature is None else temperature) is not None:
            body["temperature"] = c.temperature if temperature is None else temperature
        effort = reasoning_effort or c.reasoning_effort
        if effort:
            body["reasoning"] = {"effort": effort}
        if response_format:
            body["response_format"] = response_format
        with self._sem:
            t0 = time.perf_counter()
            first, finish, usage, parts = None, None, {}, []
            resp = self._connect(body)
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
                    if not line.startswith("data:"):  # blank lines / ':' keep-alive comments
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    ch = json.loads(payload)
                    if ch.get("error"):
                        raise RuntimeError(f"stream error: {self._scrub(json.dumps(ch['error'], ensure_ascii=False))[:300]}")
                    for choice in ch.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            if first is None:
                                first = time.perf_counter() - t0
                            parts.append(piece)
                            yield {"type": "token", "text": piece}
                        finish = choice.get("finish_reason") or finish
                    usage = ch.get("usage") or usage
        self.meter.record(usage, stage)
        yield {"type": "final", "text": "".join(parts), "finish_reason": finish, "usage": usage,
               "truncated": finish == "length", "latency_s": round(time.perf_counter() - t0, 3),
               "t_first_token_s": None if first is None else round(first, 3)}

    def _stream_anthropic(self, messages, stage, max_tokens, temperature) -> Iterator[Dict[str, Any]]:
        c = self.config
        text = lambda m: m["content"] if isinstance(m["content"], str) else "".join(p["text"] for p in m["content"])  # noqa: E731
        body: Dict[str, Any] = {
            "model": c.model, "stream": True,
            "max_tokens": c.max_tokens if max_tokens is None else max_tokens,
            "messages": [{"role": m["role"], "content": text(m)} for m in messages if m["role"] in ("user", "assistant")],
        }
        system = "\n\n".join(text(m) for m in messages if m["role"] == "system")
        if system:
            body["system"] = system
        temp = c.temperature if temperature is None else temperature
        if temp is not None:
            body["temperature"] = temp
        with self._sem:
            t0 = time.perf_counter()
            first, stop, tin, tout, parts = None, None, 0, 0, []
            resp = self._connect(body)
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
                    if not line.startswith("data:"):  # 'event:' names, blanks
                        continue
                    ev = json.loads(line[5:].strip())
                    t = ev.get("type")
                    if t == "error":
                        raise RuntimeError(f"stream error: {self._scrub(json.dumps(ev.get('error'), ensure_ascii=False))[:300]}")
                    if t == "message_start":
                        tin = int(((ev.get("message") or {}).get("usage") or {}).get("input_tokens") or 0)
                    elif t == "content_block_delta":
                        d = ev.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            if first is None:
                                first = time.perf_counter() - t0
                            parts.append(d["text"])
                            yield {"type": "token", "text": d["text"]}
                    elif t == "message_delta":
                        stop = (ev.get("delta") or {}).get("stop_reason") or stop
                        tout = int((ev.get("usage") or {}).get("output_tokens") or tout)
                    elif t == "message_stop":
                        break
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(stop, stop)
        usage: Dict[str, Any] = {"prompt_tokens": tin, "completion_tokens": tout, "cost": None}
        if c.price_in is not None and c.price_out is not None:
            usage["cost"] = (tin * c.price_in + tout * c.price_out) / 1e6
        self.meter.record(usage, stage)
        yield {"type": "final", "text": "".join(parts), "finish_reason": finish, "usage": usage,
               "truncated": finish == "length", "latency_s": round(time.perf_counter() - t0, 3),
               "t_first_token_s": None if first is None else round(first, 3)}

    def chat(self, messages, **kw) -> Dict[str, Any]:
        final = None
        for ev in self.stream_chat(messages, **kw):
            if ev["type"] == "final":
                final = ev
        return {k: v for k, v in final.items() if k != "type"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    if not ap.parse_args().ping:
        ap.print_help()
        return 2
    client = LLMClient(LLMConfig.from_env())
    try:
        r = client.chat([{"role": "user", "content": "Trả lời đúng một từ: xin chào"}],
                        max_tokens=50, stage="ping")
    except Exception as e:
        print(f"ping failed: {client._scrub(e)}")
        return 1
    print(f"model={client.config.model} finish_reason={r['finish_reason']} usage={r['usage']} "
          f"latency_s={r['latency_s']} reply={r['text'].strip()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
