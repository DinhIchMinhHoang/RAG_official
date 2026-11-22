# CLI demo RAG

Chạy từ thư mục `RAG` (Python của `RAG_lab\.venv`, thêm `-X utf8`).

```powershell
$PY = "D:\00_Workspace\AI_Lab\RAG_official\RAG_lab\.venv\Scripts\python.exe"
```

## 1. Nạp tài liệu (PDF/DOCX/PPTX/HTML/MD/TXT/ảnh, hoặc cả thư mục)

```powershell
$env:TESSDATA_PREFIX = "D:\00_Workspace\AI_Lab\RAG_official\RAG_lab\data\tessdata_fast"   # chỉ cần cho OCR
& $PY -X utf8 cli.py ingest data\raw\a.pdf data\raw\docs --store .rag_data\demo --device cuda
& $PY -X utf8 cli.py ingest a.pdf --store .rag_data\demo --force       # nạp lại, bỏ qua cache sha256
& $PY -X utf8 cli.py ingest a.pdf --store .rag_data\demo --stack       # đồng bộ thêm Postgres + MinIO nếu chạy
```

Tệp không đổi (cùng sha256) được bỏ qua; index chỉ build lại khi corpus đổi. Tài liệu lỗi in `THẤT BẠI`,
các tài liệu khác vẫn chạy, mã thoát 2.

## 2. Hỏi

```powershell
$env:OPENROUTER_API_KEY = "sk-..."          # tên biến đổi được bằng RAG_LLM_API_KEY_ENV
& $PY -X utf8 cli.py ask "Điều kiện tốt nghiệp là gì?" --store .rag_data\demo
& $PY -X utf8 cli.py ask "..." --store .rag_data\demo --rerank --fusion convex --alpha 0.7
& $PY -X utf8 cli.py ask "..." --store .rag_data\demo --min-cosine 0.45    # cosine top1 thấp thì từ chối, không gọi LLM
& $PY -X utf8 cli.py ask "..." --store .rag_data\demo --no-llm             # chỉ xem đoạn truy xuất, không cần API key
& $PY -X utf8 cli.py ask "..." --store .rag_data\demo --json
& $PY -X utf8 cli.py ask "..." --store .rag_data\demo --doc <doc_id> --doc <doc_id2>   # giới hạn theo tài liệu
```

- `--rerank on|off|auto` (mặc định `auto` = bật khi store có >= 2 tài liệu; đo thực tế: rerank sửa lỗi truy xuất
  đa tài liệu/đa ngôn ngữ, không lợi gì với 1 tài liệu). Chế độ dùng được in ở dòng thời gian cuối.
- `--backend faiss|qdrant` (ingest/ask/chat/info; mặc định faiss). Qdrant cần `docker compose up -d` (thư mục
  `deploy`), kiểm tra bằng `python -m store.doctor`; collection tên `rag_<tên thư mục store>`.
- Mỗi lệnh `ask` nạp lại bge-m3 (~10 s); khi nhúng > 3 s CLI nhắc dùng `chat` để giữ mô hình nóng.
- Tài liệu không có số trang (DOCX/HTML/MD) hiện `không có số trang`.

## 3. Hội thoại và thông tin

```powershell
& $PY -X utf8 cli.py chat --store .rag_data\demo       # /sources  /stats  /quit (nhớ 6 tin nhắn gần nhất)
& $PY -X utf8 cli.py info --store .rag_data\demo       # tài liệu, manifest, cấu hình (đã ẩn bí mật), GPU
```

## Biến môi trường

`RAG_LLM_MODEL`, `RAG_LLM_BASE_URL`, `RAG_LLM_API_KEY_ENV`, `RAG_LLM_MAX_TOKENS`; stack: `RAG_PG_*`, `RAG_MINIO_*`.

## Lỗi thường gặp

- `Thiếu biến môi trường OPENROUTER_API_KEY`: đặt biến hoặc thêm `--no-llm`.
- `Chưa có index`: chạy `ingest` trước với cùng `--store`.
- Mã thoát: 0 ổn, 1 lỗi LLM, 2 lỗi cấu hình/tài liệu.
