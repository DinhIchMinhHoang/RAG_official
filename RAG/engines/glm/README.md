# GLM-OCR engine worker
Formula and image-table recognition (`zai-org/GLM-OCR`, bf16, CUDA, greedy). Ops: `formula`, `table`, `shutdown`.
Setup: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`, download weights once (HF cache); the worker then runs with `HF_HUB_OFFLINE=1`.
Locate: env `RAG_ENGINE_GLM_PYTHON`, else `engines/glm/.venv/Scripts/python.exe`, else dev `RAG_lab/.venv-glmocr`.
Protocol: see `../README.md`.
