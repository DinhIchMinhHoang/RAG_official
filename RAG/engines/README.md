# Engine workers
Out-of-process OCR engines used by `parser/engine_router.py`. Each worker is its own process in its own venv, started lazily, kept warm for one ingest run, closed before embedding. Missing python or weights -> engine `skipped`, pipeline unchanged.

Protocol: JSON lines over stdin/stdout, files by absolute path.
- Handshake (first line): `{"ready":true,"engine":"glm","version":"...","load_s":4.5,"vram_gb":2.26}` or `{"ready":false,"error":"..."}`
- Request: `{"id":n,"op":"formula|table","image":path}`, `{"id":n,"op":"tables","pdf":path,"first":a,"last":b}`, `{"id":n,"op":"shutdown"}`
- Reply: `{"id":n,"ok":true,"result":{...},"s":1.02}` or `{"id":n,"ok":false,"error":"..."}`

Workers: `glm/` (formula, table), `docling/` (tables), `fake/` (stdlib, tests; env FAKE_SLEEP, FAKE_CRASH_AT, FAKE_BAD_JSON, FAKE_LOG).
Env: `RAG_ENGINE_GLM_PYTHON`, `RAG_ENGINE_DOCLING_PYTHON`. Mode `auto` = GLM only; Docling is opt-in (`--engines docling` or `glm,docling`).

## Tạo venv trên máy mới (Windows, GPU NVIDIA, CUDA 12.6)
Chạy từ thư mục `RAG\engines`:
```
python -m venv glm\.venv
glm\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
glm\.venv\Scripts\python.exe -m pip install -r glm\requirements.txt

python -m venv docling\.venv          # chỉ khi dùng --engines docling
docling\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
docling\.venv\Scripts\python.exe -m pip install -r docling\requirements.txt
```
Router tự tìm `engines\<tên>\.venv\Scripts\python.exe`, hoặc dùng `RAG_ENGINE_<TÊN>_PYTHON`.

## Tải trước trọng số (worker chạy với `HF_HUB_OFFLINE=1`, không tự tải)
```
glm\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; s('zai-org/GLM-OCR')"
```
Docling: chạy một lần `docling\.venv\Scripts\python.exe -c "from docling.document_converter import DocumentConverter as C; from docling.datamodel.base_models import InputFormat as F; C().initialize_pipeline(F.PDF)"` khi còn mạng (không đặt HF_HUB_OFFLINE) để tải mô hình layout/bảng. Chưa kiểm tra giấy phép của các trọng số.
