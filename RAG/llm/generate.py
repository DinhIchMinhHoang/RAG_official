"""RAG answer generation with Ollama or a local 4-bit Transformers model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

try:
    from .prompt import NO_CONTEXT_REFUSAL, build_messages, source_records, truncate_contexts
    from .rerank import Reranker
    from .retrieve import RAGRetriever
except ImportError:
    # Also supports ``python -m llm.generate`` from inside the RAG directory.
    from llm.prompt import NO_CONTEXT_REFUSAL, build_messages, source_records, truncate_contexts
    from llm.rerank import Reranker
    from llm.retrieve import RAGRetriever


DEFAULT_BACKEND = "ollama"
# Use the instruct variant for QA. The generic qwen3:4b tag currently resolves
# to a thinking-only 2507 model, which can spend the whole output budget on reasoning.
DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 600.0
LOGGER = logging.getLogger("rag.ollama")


class OpenAICompatibleClient:
    """Minimal standard-library client for Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.cpu_fallback = os.getenv("OLLAMA_CPU_FALLBACK", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url phải là URL HTTP(S) hợp lệ.")
        if timeout <= 0:
            raise ValueError("timeout phải > 0.")

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Streaming keeps the connection active while thinking models
            # produce a long response, instead of waiting for the final token.
            "stream": True,
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_parts: List[str] = []
                response_model = model
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event = line[len("data:"):].strip()
                    if not event or event == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        LOGGER.debug("Bỏ qua Ollama SSE không phải JSON: %s", event[:200])
                        continue
                    if isinstance(chunk, dict) and chunk.get("model"):
                        response_model = str(chunk["model"])
                    choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        content_parts.append(content)
                    elif isinstance(content, list):
                        content_parts.extend(
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, dict)
                            and part.get("type") in {"text", "output_text"}
                        )
                answer = "".join(content_parts).strip()
                if not answer:
                    raise RuntimeError(
                        "Ollama không trả về nội dung câu trả lời. Model có thể chỉ sinh reasoning; "
                        "hãy dùng model instruct, ví dụ `qwen3:4b-instruct`, hoặc tăng --max-tokens."
                    )
                return {"answer": answer, "model": response_model}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            try:
                error_body = json.loads(detail)
                error_message = error_body.get("error", {}).get("message", detail)
            except (json.JSONDecodeError, AttributeError):
                error_message = detail or str(exc.reason)
            lower_message = error_message.lower()
            if self.cpu_fallback and "out of memory" in lower_message and any(
                marker in lower_message for marker in ("cuda", "gpu", "kv cache")
            ):
                LOGGER.warning(
                    "Ollama GPU thiếu VRAM; thử lại model %s trên CPU. "
                    "Có thể tắt fallback bằng OLLAMA_CPU_FALLBACK=0.",
                    model,
                )
                try:
                    return self._chat_completion_on_cpu(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Ollama hết VRAM GPU và chạy lại trên CPU cũng thất bại: {fallback_exc}"
                    ) from exc
            if "llama-server binary not found" in error_message.lower():
                raise RuntimeError(
                    "Bản cài Ollama bị thiếu llama-server.exe nên không thể chạy model. "
                    "Hãy cài đè bản Windows chính thức tại https://ollama.com/download/windows, "
                    "sau đó khởi động lại Ollama. Model đã tải trong ~/.ollama/models được giữ riêng."
                ) from exc
            if "model" in error_message.lower() and "not found" in error_message.lower():
                raise RuntimeError(
                    f"Ollama chưa có model '{model}'. Chạy `ollama pull {model}` hoặc truyền model đã cài. "
                    f"Chi tiết: {error_message}"
                ) from exc
            raise RuntimeError(
                f"Ollama API trả HTTP {exc.code}: {error_message}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Không kết nối được Ollama tại {self.base_url}: {exc.reason}. "
                "Hãy khởi động Ollama và nạp model trước."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"LLM API không gửi dữ liệu trong {self.timeout:g} giây. "
                "Model có thể đang nạp/chạy quá chậm; thử model nhỏ hơn hoặc tăng --timeout."
            ) from exc

    def _chat_completion_on_cpu(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Retry Ollama's native API with zero GPU layers after a CUDA OOM."""
        base_url = self.base_url
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[:-len(suffix)]
                break
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": 0,
                "num_ctx": 4096,
            },
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_parts: List[str] = []
                response_model = model
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.debug("Bỏ qua Ollama NDJSON không hợp lệ: %s", line[:200])
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    if chunk.get("model"):
                        response_model = str(chunk["model"])
                    message = chunk.get("message") or {}
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str):
                        content_parts.append(content)
                    if chunk.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Ollama CPU API trả HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Không kết nối được Ollama CPU API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama CPU API không gửi dữ liệu trong {self.timeout:g} giây; "
                "hãy tăng --timeout hoặc dùng model nhỏ hơn."
            ) from exc

        answer = "".join(content_parts).strip()
        if not answer:
            raise RuntimeError("Ollama CPU API trả về câu trả lời rỗng.")
        return {"answer": answer, "model": response_model}


class RAGGenerator:
    """Retrieve sources, optionally rerank them, then generate a grounded answer."""

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model: Optional[str] = None,
        retriever: Optional[RAGRetriever] = None,
        reranker: Optional[Reranker] = None,
        rerank_mode: str = "passthrough",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 800,
        top_k: int = 10,
        max_parents: int = 3,
        max_context_chars: int = 4000,
        device: Optional[str] = None,
    ) -> None:
        if backend not in {"ollama", "transformers"}:
            raise ValueError("backend phải là 'ollama' hoặc 'transformers'.")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature phải nằm trong [0, 2].")
        if max_tokens < 1 or top_k < 1 or max_parents < 1 or max_context_chars < 1:
            raise ValueError("max_tokens, top_k, max_parents và max_context_chars phải >= 1.")

        self.backend = backend
        if model:
            self.model_name = model
        elif backend == "transformers":
            self.model_name = DEFAULT_TRANSFORMERS_MODEL
        else:
            self.model_name = os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.max_parents = max_parents
        self.max_context_chars = max_context_chars
        self.retriever = retriever or RAGRetriever(device=device)
        self.reranker = reranker or Reranker(mode=rerank_mode, device=device)
        self.device = device

        if backend == "ollama":
            self.client = OpenAICompatibleClient(
                base_url=base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL,
                api_key=api_key,
                timeout=timeout,
            )
            self.tokenizer = None
            self.transformers_model = None
        else:
            self.client = None
            self.tokenizer, self.transformers_model = self._load_transformers_model()

    def _load_transformers_model(self) -> tuple[Any, Any]:
        """Load a HuggingFace causal LM with NF4 4-bit quantization."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Backend transformers cần torch, transformers và bitsandbytes."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Backend transformers 4-bit cần GPU CUDA khả dụng; hãy dùng Ollama hoặc cấu hình GPU."
            )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        device_map: Any = {"": self.device} if self.device and self.device.startswith("cuda") else "auto"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        model.eval()
        return tokenizer, model

    def _generate_transformers(self, messages: List[Dict[str, str]]) -> str:
        import torch

        tokenizer = self.tokenizer
        model = self.transformers_model
        if tokenizer is None or model is None:
            raise RuntimeError("Transformers model chưa được nạp.")
        if getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            input_ids = inputs
        else:
            prompt_text = "\n".join(
                f"<|{message['role']}|>\n{message['content']}"
                for message in messages
            ) + "\n<|assistant|>\n"
            encoded = tokenizer(prompt_text, return_tensors="pt")
            encoded = encoded.to(model.device)
            input_ids = encoded["input_ids"]

        generation_options: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_options["temperature"] = self.temperature
        with torch.inference_mode():
            output = model.generate(input_ids, **generation_options)
        new_tokens = output[0][input_ids.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("Transformers model trả về câu trả lời rỗng.")
        return answer

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Return ``query``, grounded ``answer`` and the cited ``sources``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Câu hỏi không được để trống.")
        query = unicodedata.normalize("NFC", query).strip()
        print(f"\n[1/3] Đang truy hồi các điều khoản quy chế liên quan...", flush=True)
        contexts = self.retriever.get_contexts(
            query,
            top_k=self.top_k,
            max_parents=self.max_parents,
        )
        contexts = self.reranker.rerank(query, contexts, top_n=self.max_parents)
        if not contexts:
            return {"query": query, "answer": NO_CONTEXT_REFUSAL, "sources": []}

        prompt_contexts = truncate_contexts(
            contexts,
            max_context_chars=self.max_context_chars,
        )
        if not prompt_contexts:
            return {"query": query, "answer": NO_CONTEXT_REFUSAL, "sources": []}
        messages = build_messages(
            query,
            prompt_contexts,
            max_context_chars=self.max_context_chars,
        )

        print(f"[2/3] Đã chọn {len(prompt_contexts)} điều khoản căn cứ. Đang gửi tới mô hình {self.model_name}...", flush=True)
        print(f"[3/3] Đang sinh câu trả lời (xin chờ trong giây lát)...", flush=True)

        if self.backend == "ollama":
            completion = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = completion["answer"]
        else:
            answer = self._generate_transformers(messages)

        return {
            "query": query,
            "answer": answer,
            "sources": source_records(prompt_contexts),
        }

    def answer(self, query: str) -> Dict[str, Any]:
        """Backward-compatible alias for :meth:`generate_answer`."""
        return self.generate_answer(query)


def generate_answer(query: str, **generator_options: Any) -> Dict[str, Any]:
    """Convenience function for one-off use; reuse ``RAGGenerator`` in services."""
    return RAGGenerator(**generator_options).generate_answer(query)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hỏi đáp Quy chế đào tạo qua Ollama hoặc Transformers.",
    )
    parser.add_argument("query", help="Câu hỏi tiếng Việt")
    parser.add_argument("--backend", choices=("ollama", "transformers"), default=DEFAULT_BACKEND)
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama: qwen3:4b-instruct; Transformers: HuggingFace model ID",
    )
    parser.add_argument("--base-url", default=None, help="Ollama OpenAI-compatible URL")
    parser.add_argument("--rerank-mode", choices=("passthrough", "cross-encoder"), default="passthrough")
    parser.add_argument("--dense-index", default=None)
    parser.add_argument("--bm25-index", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Thiết bị encoder/reranker; Ollama chạy theo cấu hình của Ollama server",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-parents", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Thời gian chờ tối đa giữa các phần dữ liệu từ Ollama (mặc định: 600 giây)",
    )
    return parser


def _configure_console_encoding() -> None:
    """Keep Vietnamese CLI help/output usable in Windows legacy consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        retriever = RAGRetriever(
            dense_index=args.dense_index,
            bm25_index=args.bm25_index,
            device=args.device,
        )
        generator = RAGGenerator(
            backend=args.backend,
            model=args.model,
            retriever=retriever,
            rerank_mode=args.rerank_mode,
            base_url=args.base_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            max_parents=args.max_parents,
            max_context_chars=args.max_context_chars,
            device=args.device,
        )
        result = generator.generate_answer(args.query)
    except Exception as exc:
        print(f"\n[LỖI] Không thể tạo câu trả lời: {exc}\n", flush=True)
        return 1

    print("\n" + "=" * 60)
    print("CÂU TRẢ LỜI CỦA HỆ THỐNG:")
    print("=" * 60)
    print(result["answer"])
    print("\nCÁC CĂN CỨ TRÍCH DẪN:")
    for src in result.get("sources", []):
        print(f" * {src.get('breadcrumb', 'Điều khoản quy chế')}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
