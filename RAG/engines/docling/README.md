# Docling engine worker
Table structure for digital PDFs (`PdfPipelineOptions(do_ocr=False)`). Ops: `tables` (pdf, first, last; 1-based inclusive), `shutdown`.
Setup: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`, fetch model weights once; the worker then runs with `HF_HUB_OFFLINE=1`.
Locate: env `RAG_ENGINE_DOCLING_PYTHON`, else `engines/docling/.venv/Scripts/python.exe`, else dev `RAG_lab/.venv-docling`.
