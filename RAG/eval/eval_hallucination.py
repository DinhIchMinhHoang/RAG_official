"""Measure exact refusal behavior and source-supported article citations."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


RAG_DIR = Path(__file__).resolve().parents[1]
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from llm.generate import RAGGenerator
    from llm.prompt import NO_CONTEXT_REFUSAL
    from llm.retrieve import RAGRetriever
except ImportError:
    from RAG.llm.generate import RAGGenerator
    from RAG.llm.prompt import NO_CONTEXT_REFUSAL
    from RAG.llm.retrieve import RAGRetriever


ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+)\b", flags=re.IGNORECASE)


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được eval set {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("eval_set.json phải chứa một JSON array.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ValueError(f"Phần tử eval #{index + 1} thiếu question hợp lệ.")
        if not isinstance(row.get("answerable"), bool):
            raise ValueError(f"{row.get('id', index)} thiếu trường answerable kiểu bool.")
        if not isinstance(row.get("relevant_parent_ids", []), list):
            raise ValueError(f"{row.get('id', index)}: relevant_parent_ids phải là list.")
    if not any(row["answerable"] for row in rows) or not any(not row["answerable"] for row in rows):
        raise ValueError("eval set cần có cả câu hỏi answerable và câu hỏi bẫy.")
    return rows


def load_parent_articles(path: Path) -> Dict[str, int]:
    articles: Dict[str, int] = {}
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                parent_id = row.get("parent_id")
                article = row.get("article")
                if isinstance(parent_id, str) and article is not None:
                    articles[parent_id] = int(article)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Không đọc được parent metadata {path}: {exc}") from exc
    return articles


def article_numbers_from_sources(
    sources: Sequence[Mapping[str, Any]],
    parent_articles: Mapping[str, int],
) -> Set[int]:
    numbers: Set[int] = set()
    for source in sources:
        parent_id = source.get("parent_id")
        if isinstance(parent_id, str) and parent_id in parent_articles:
            numbers.add(parent_articles[parent_id])
        breadcrumb = str(source.get("breadcrumb", ""))
        numbers.update(int(value) for value in ARTICLE_RE.findall(breadcrumb))
        source_text = str(source.get("text", ""))
        numbers.update(int(value) for value in ARTICLE_RE.findall(source_text))
    return numbers


def evaluate_answer(
    item: Mapping[str, Any],
    answer_result: Mapping[str, Any],
    parent_articles: Mapping[str, int],
) -> Dict[str, Any]:
    answer = unicodedata.normalize("NFC", str(answer_result.get("answer", ""))).strip()
    sources = answer_result.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    source_articles = article_numbers_from_sources(sources, parent_articles)
    cited_articles = {int(value) for value in ARTICLE_RE.findall(answer)}
    if item["answerable"]:
        expected_articles = {
            parent_articles[parent_id]
            for parent_id in item.get("relevant_parent_ids", [])
            if parent_id in parent_articles
        }
        citation_correct: Optional[bool] = bool(cited_articles) and \
            cited_articles.issubset(source_articles) and bool(cited_articles & expected_articles)
        refusal_correct: Optional[bool] = None
    else:
        expected_refusal = unicodedata.normalize(
            "NFC", str(item.get("expected_refusal", NO_CONTEXT_REFUSAL))
        ).strip()
        refusal_correct = answer == expected_refusal
        citation_correct = None

    return {
        "id": item.get("id"),
        "question": item["question"],
        "answerable": item["answerable"],
        "answer": answer,
        "refusal_correct": refusal_correct,
        "citation_correct": citation_correct,
        "cited_articles": sorted(cited_articles),
        "source_articles": sorted(source_articles),
        "expected_parent_ids": item.get("relevant_parent_ids", []),
        "source_parent_ids": [source.get("parent_id") for source in sources],
    }


def _accuracy(values: List[bool]) -> Dict[str, Any]:
    total = len(values)
    correct = sum(values)
    return {
        "correct": correct,
        "total": total,
        "rate": correct / total if total else None,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    refusal = summary["refusal_accuracy"]
    citation = summary["citation_accuracy"]

    def percent(metric: Mapping[str, Any]) -> str:
        return "N/A" if metric["rate"] is None else f"{metric['rate'] * 100:.1f}%"

    lines = [
        "# Hallucination evaluation",
        "",
        "| Chỉ số | Đúng / tổng | Tỷ lệ |",
        "|---|---:|---:|",
        f"| Refusal Accuracy (câu bẫy, khớp chính xác câu từ chối) | {refusal['correct']} / {refusal['total']} | {percent(refusal)} |",
        f"| Citation Accuracy (câu có đáp án) | {citation['correct']} / {citation['total']} | {percent(citation)} |",
        "",
        "Citation được tính đúng khi câu trả lời có nêu ít nhất một Điều, mọi Điều được nêu đều có trong nguồn trả về, và có ít nhất một Điều khớp parent gán nhãn vàng.",
        "",
        "## Kết quả từng câu",
        "",
        "| ID | Loại | Kết quả | Câu hỏi | Trả lời |",
        "|---|---|---|---|---|",
    ]
    for row in report["results"]:
        kind = "answerable" if row["answerable"] else "trap"
        passed = row["citation_correct"] if row["answerable"] else row["refusal_correct"]
        result = "PASS" if passed else ("ERROR" if row.get("error") else "FAIL")
        answer = str(row.get("answer") or row.get("error", "")).replace("|", "\\|").replace("\n", " ")
        question = str(row["question"]).replace("|", "\\|")
        lines.append(f"| {row.get('id', '')} | {kind} | {result} | {question} | {answer[:180]} |")
    return "\n".join(lines) + "\n"


def run_evaluation(
    items: Sequence[Mapping[str, Any]],
    generator: RAGGenerator,
    parent_articles: Mapping[str, int],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    refusal_values: List[bool] = []
    citation_values: List[bool] = []
    for item in items:
        try:
            generated = generator.generate_answer(item["question"])
            row = evaluate_answer(item, generated, parent_articles)
            if row["answerable"]:
                citation_values.append(bool(row["citation_correct"]))
            else:
                refusal_values.append(bool(row["refusal_correct"]))
        except Exception as exc:
            row = {
                "id": item.get("id"),
                "question": item["question"],
                "answerable": item["answerable"],
                "answer": "",
                "refusal_correct": False if not item["answerable"] else None,
                "citation_correct": False if item["answerable"] else None,
                "cited_articles": [],
                "source_articles": [],
                "expected_parent_ids": item.get("relevant_parent_ids", []),
                "source_parent_ids": [],
                "error": str(exc),
            }
            if item["answerable"]:
                citation_values.append(False)
            else:
                refusal_values.append(False)
        results.append(row)

    return {
        "summary": {
            "refusal_accuracy": _accuracy(refusal_values),
            "citation_accuracy": _accuracy(citation_values),
            "refusal_definition": "Exact match after trimming whitespace with the expected refusal sentence.",
            "citation_definition": "At least one cited Điều; all cited Điều numbers occur in returned source articles; at least one matches a gold relevant parent.",
        },
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá từ chối và trích dẫn của RAGGenerator.")
    parser.add_argument("--eval-set", type=Path, default=Path(__file__).with_name("eval_set.json"))
    parser.add_argument("--parents", type=Path, default=RAG_DIR / "data/processed/parents.jsonl")
    parser.add_argument("--dense-index", type=Path, default=RAG_DIR / "data/indexes/dense_bge_m3")
    parser.add_argument("--bm25-index", type=Path, default=RAG_DIR / "data/indexes/bm25")
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--rerank-mode", choices=("passthrough", "cross-encoder"), default="passthrough")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu, phục vụ smoke test")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).with_name("hallucination_report.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path(__file__).with_name("hallucination_report.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải >= 1")
    try:
        items = load_eval_set(args.eval_set)
        if args.limit is not None:
            items = items[:args.limit]
        parent_articles = load_parent_articles(args.parents)
        retriever = RAGRetriever(
            dense_index=args.dense_index,
            bm25_index=args.bm25_index,
            device=args.device,
        )
        generator = RAGGenerator(
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            rerank_mode=args.rerank_mode,
            retriever=retriever,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            device=args.device,
        )
        report = run_evaluation(items, generator, parent_articles)
    except Exception as exc:
        print(f"\n[LỖI] Đánh giá hallucination thất bại: {exc}\n", flush=True)
        return 1

    markdown = markdown_report(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
