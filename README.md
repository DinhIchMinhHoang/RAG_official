# RAG_official — Hệ thống hỏi đáp tiếng Việt

Hai hệ thống hỏi đáp tiếng Việt:

1. **Extractive QA** (`Extractive_QA/`, repo riêng, gắn vào đây dưới dạng submodule): trích câu trả lời từ một đoạn văn ngữ cảnh cho sẵn.
2. **RAG** (Retrieval-Augmented Generation): truy hồi tài liệu liên quan từ một kho tài liệu rồi dùng LLM sinh câu trả lời.

## Đầu vào / Đầu ra

| Hệ thống | Đầu vào | Đầu ra |
|---|---|---|
| Extractive QA | Câu hỏi + đoạn văn ngữ cảnh | Đoạn trích trong ngữ cảnh làm câu trả lời |
| RAG | Câu hỏi + kho tài liệu | Câu trả lời được sinh ra + tài liệu nguồn được truy hồi |

## Dữ liệu

- UIT-ViQuAD 2.0 (cho Extractive QA), hoặc
- Một kho tài liệu như quy chế đào tạo (cho RAG).

## Kiến thức liên quan

Question answering, BM25, dense retrieval, pretrained LM, LLM, LoRA.

## Yêu cầu tối thiểu

- [ ] Fine-tune PhoBERT hoặc XLM-R cho extractive QA; đánh giá bằng EM và F1.
- [ ] Hệ thống RAG gồm retriever (BM25 và dense embedding) và một LLM mã nguồn mở.

## Mở rộng

- [ ] So sánh BM25, dense và hybrid retrieval.
- [ ] Đánh giá hiện tượng ảo giác (hallucination).
- [ ] Chatbot demo (giao diện trong `front_end/`).

## Cấu trúc

```
RAG_official/
├── Extractive_QA/   # submodule: fine-tune + đánh giá extractive QA
└── front_end/       # giao diện demo (Vite)
```

## Cài đặt

```bash
git clone --recurse-submodules https://github.com/DinhIchMinhHoang/RAG_official.git
cd RAG_official/front_end && npm install && npm run dev
```

> README tạm, sẽ cập nhật khi có code và kết quả.
