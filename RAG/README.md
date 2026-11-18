# RAG Parser - Block-Level Document Ingestion

Phiên bản refactor từ `parse_pdf.py` cho kho quy chế đại học. Chạy Python **3.10+**.

## Cấu trúc thư mục

Sao chép các file từ gói ZIP vào thư mục gốc dự án như sau:

```text
ViQA_Advanced_RAG/
├── data/
│   ├── Quy-che-DTDH-3626.pdf
│   └── processed/               <-- tạo tự động
├── parser/
│   ├── parse_document.py        <-- lõi parser dùng chung 9 định dạng
│   ├── parse_pdf.py             <-- entrypoint gọn cho PDF
│   └── split_articles.py        <-- bản vá tương thích Markdown mới
└── requirements_parser.txt
```

## Cài trên Windows (PowerShell/VS Code Terminal)

Từ thư mục gốc dự án:

```powershell
cd "D:\00_Workspace\UET_Uni\ViQA_Advanced_RAG"
python -m pip install -r requirements_parser.txt
python parser\parse_pdf.py data\Quy-che-DTDH-3626.pdf --force
```

Chạy lại không thay đổi PDF và **không** có `--force` sẽ hit cache SHA-256:

```powershell
python parser\parse_pdf.py data\Quy-che-DTDH-3626.pdf
```

Để xử lý thêm DOCX/PPTX/HTML/MD/TXT/PNG/JPG:

```powershell
python parser\parse_document.py data\sample.docx
```

Hoặc chỉ rõ đường dẫn output cho mọi định dạng:

```powershell
python parser\parse_pdf.py data\Quy-che-DTDH-3626.pdf --out-dir "D:\00_Workspace\UET_Uni\ViQA_Advanced_RAG\data\processed"
```

## Output

```text
data/processed/Quy-che-DTDH-3626.md
                Quy-che-DTDH-3626.json
```

File `.md` để xem bằng mắt và chạy `split_articles.py`. Chú thích `<!-- PDF page N -->` giúp debug và **được bỏ qua** trong `split_articles.py` đã vá.

File `.json` chứa `source` (SHA-256, đường dẫn, định dạng), `stats`, `blocks`.
Mỗi block có:

```json
{
  "id": "p0016_b00341",
  "type": "table",
  "text": "| ... | ... |",
  "level": 0,
  "location": {"page_num": 16},
  "breadcrumb": "Chương III > Điều 17 > Khoản 2",
  "bbox": [0.12, 0.19, 0.91, 0.47],
  "metadata": {
    "source": "rapidocr",
    "ocr": true,
    "approximate_table": true,
    "chapter": "III", "article": 17, "clause": 2, "point": null
  }
}
```

**Đây là minh họa schema, không phải block trích từ tài liệu thật.** `bbox` khi có là tọa độ chuẩn hóa trong khoảng 0..1. File DOCX không có page number cố định nên `location: {}`; PPTX có `slide_num`.

## Năm stage

1. SHA-256 check + định tuyến bằng phần mở rộng + `charset-normalizer` cho văn bản.
2. PDF số dùng `pymupdf4llm`, DOCX dùng `python-docx`, PPTX dùng `python-pptx`, HTML dùng `beautifulsoup4` + `markdownify`, văn bản dùng đọc text; tất cả trả ContentBlock.
3. Nếu PDF scan (<20 ký tự hoặc trang chủ yếu là ảnh với text layer ít), render qua fitz với cạnh dài mục tiêu 1600px và OCR bằng `rapidocr-onnxruntime`. Với ảnh nhúng trong các định dạng khác, resize bằng Pillow rồi OCR, giữ vị trí tương đối. Khởi tạo RapidOCR **một lần**.
4. Loại running header/footer chỉ khi nằm mép trang và lặp đủ nhiều; gắn Chương > Điều > Khoản > Điểm vào breadcrumb. Bảng nhận từ ảnh scan dùng heuristic căn cột và có `approximate_table: true`.
5. Xuất `.md` + `.json` để đưa vào retrieval/chunking.

## Parent-Child Chunking

```powershell
python parser\split_articles.py data\processed\Quy-che-DTDH-3626.md
```

Ghi `parents.jsonl` và `children.jsonl` vào `processed/`. Kiểm tra số Điều được phát hiện để phát hiện thiếu Điều do OCR.

## Những giới hạn cần chú ý

- **PDF thật trong dự án là 53 trang scan; 52 trang không có text layer.** OCR gần như toàn bộ là bắt buộc, không thể biến thành tốc độ text-PDF.
- RapidOCR mặc định không bảo đảm dấu tiếng Việt chính xác. Nên đối chiếu thử trang 2, trang 16 (có bảng), và các trang có Điều/Khoản dài. Nếu OCR làm hỏng nhiều dấu, phải chỉnh nhận dạng model/ngôn ngữ hoặc chuyển sang OCR nhận dạng tiếng Việt tốt hơn.
- Việc resize 1600px chỉ là đánh đổi tốc độ/độ chính xác, không có bảo đảm 0.4 giây/trang.
- Nhận dạng bảng scan ở đây là **heuristic**, không phải layout model; bảng chia cột/phức tạp cần kiểm tra thủ công.
- Text boxes của PDF số được nối với Markdown blocks qua vị trí ước lượng; tài liệu nhiều cột, hình nổi/overlap có thể không đúng thứ tự hoàn toàn.
- Thứ tự đọc các shape của PPTX được xếp theo tọa độ top-to-bottom, left-to-right; không phải thứ tự đọc ngữ nghĩa được bảo đảm.
- Mục lục/số thứ tự **auto-numbering** của Word không nằm sẵn trong text của python-docx; có thể cần hậu xử lý OOXML để khôi phục chính xác.
- HTML chỉ OCR ảnh **local** và `data:image` trong cùng vùng đường dẫn tài liệu; không tự tải URL ngoài.
- Nếu OCR trả về các hình ảnh không chữ thì không tạo block OCR rỗng. Kiểm tra `stats` và nội dung văn bản xuất.
- Cache SHA-256 tính theo bytes tài liệu cùng version parser; nếu đổi cấu hình, thêm `--force`.

## Kiểm thử đã thực hiện

- Python `py_compile` cho ba script.
- Chạy thử Markdown, DOCX, PPTX, HTML, ảnh nhúng Word/HTML, table OCR heuristic và cache SHA-256.
- Kiểm tra layout PDF gốc với fitz (53 trang, 53 ảnh, 52 trang text layer rỗng).
- **Chưa chạy OCR thật** với `rapidocr-onnxruntime` vì môi trường kiểm thử không thể cài package từ Internet. Các kiểm thử ảnh sử dụng stub trả text giả để kiểm tra luồng chương trình, **không phải** xác nhận chất lượng OCR.




# Embedding cho RAG quy chế đào tạo

**Bản hiện tại đã bổ sung BM25 và hybrid: đọc `README_HYBRID.md` trước. File này mô tả riêng nhánh dense.**

Bộ mã này bổ sung nhánh **dense retrieval** cho cấu trúc dự án trong ảnh. Chép thư mục `embed/` vào thư mục gốc `RAG/` của bạn. Các tệp cùng tên trong ảnh chưa được gửi kèm nội dung, nên hãy so sánh trước khi thay thế. Giữ `build_bm25.py`, `chunking.py` và các module khác đang dùng.

## 1. Kết quả xem dữ liệu và index.html

- `parents.jsonl`: 51 Điều, 51 `parent_id` khác nhau.
- `children.jsonl`: 225 child, 225 `child_id` khác nhau; tất cả đều tìm được parent, `chapter`/`article` khớp parent.
- Child dài nhất: `d8_k1_b`, 1.487 ký tự; parent dài nhất: 5.730 ký tự. Ký tự không tương đương token; chỉ khi có tokenizer mới xác định được độ dài token.
- `index.html` là slide kế hoạch, chưa phải ứng dụng hỏi đáp. Các slide 13–24 đề xuất dense + BM25, RRF, rerank, sau đó lấy parent cho LLM.
- Phần này hoàn thành mã embedding, lưu FAISS và tìm thử child → parent. BM25 và RRF được hướng dẫn trong README_HYBRID.md; HyDE, reranker, sinh câu trả lời và bảng đánh giá chưa nằm trong bộ mã này.

**Thiết kế:** chỉ embed child. Đầu vào model là `breadcrumb + "\n" + text`; parent lưu nguyên văn để lookup. Không ghép toàn bộ parent vào từng child vì sẽ làm loãng đơn vị truy hồi. Duy trì cùng `child_id` cho BM25 sau này.

BGE-M3 tạo dense vector 1024 chiều và hỗ trợ tối đa 8192 token. Bộ mã chọn ngưỡng vận hành 1024 token, batch 4, float32, tự chọn CPU/GPU. Không thêm `query:`/`passage:` và không word-segment cho BGE-M3. Chỉ lấy dense vectors bằng Sentence Transformers; sparse của BGE-M3 không tự động được tạo và cũng không phải BM25.

Cả query và child được chuẩn hóa L2. FAISS `IndexFlatIP` tìm inner product trên vector chuẩn hóa, tương đương cosine similarity; không cần huấn luyện index. FAISS là thư viện tìm vector, còn metadata/parent được lưu bằng JSONL.

## 2. Cấu trúc sau khi chép mã

```text
RAG/
  data/processed/children.jsonl
  data/processed/parents.jsonl
  embed/__init__.py
  embed/embedder.py
  embed/build_dense.py
  embed/search_dense.py
  requirements-embedding.txt
```

Mọi lệnh dưới đây chạy ở thư mục **RAG**, nơi có cả `embed` và `data`. Dùng `python -m ...` để import đúng package; không dùng `python embed/build_dense.py`.

## 3. Cài và kiểm tra dữ liệu (PowerShell / Terminal VS Code)

Python 3.10+; chọn cùng interpreter với môi trường bạn đang làm việc.

```powershell
python -m pip install -r requirements-embedding.txt
python -m embed.build_dense --validate-only
```

`sentence-transformers` kéo theo PyTorch và Transformers. Nếu bạn cần PyTorch dùng CUDA, chọn lệnh cài phù hợp trên trang PyTorch trước khi cài requirements. CPU vẫn chạy được; BGE-M3 khá nặng nên nên đóng ứng dụng không cần thiết nếu RAM ít. Lần đầu cần Internet để tải model Hugging Face, các lần sau dùng cache. Không cần API key hay API embedding trả phí.

`--validate-only` dùng thư viện chuẩn, có thể chạy trước khi cài requirements. Nó kiểm tra JSON, trường bắt buộc, ID trùng và ánh xạ parent; **không** tải model hoặc kiểm tra số token.

## 4. Tạo embedding và index

```powershell
python -m embed.build_dense
```

Nếu muốn ép CPU hoặc giảm batch khi thiếu bộ nhớ:

```powershell
python -m embed.build_dense --device cpu --batch-size 1
```

Nếu input vượt 1024 token, chương trình báo lỗi, không âm thầm truncate. Bạn có thể chunk lại hoặc tăng ngưỡng, tối đa 8192:

```powershell
python -m embed.build_dense --max-length 2048 --batch-size 2
```

Kết quả tạo ở `data/indexes/dense_bge_m3/`:

| Tệp | Nội dung |
|---|---|
| `children.faiss` | FAISS cosine index, một vector cho mỗi child |
| `embeddings.npy` | Ma trận float32 để kiểm tra hoặc dùng lại |
| `children.jsonl` | Snapshot metadata và text; dòng i tương ứng vector i |
| `parents.jsonl` | Snapshot parent để lookup, tránh lệch phiên bản với index |
| `metadata.json` | Model/revision, ngưỡng token, dimension, số dòng, SHA-256 |

Với dữ liệu hiện tại, chạy thành công sẽ cho ma trận `(225, 1024)`. Đây là kích thước dự kiến, không phải kết quả embedding đã chạy sẵn trong gói tải xuống.

Mã từ chối ghi vào thư mục output đã có. Khi sửa dữ liệu hoặc cách embed, tạo index mới rồi thử tìm trên index đó:

```powershell
python -m embed.build_dense --output data/indexes/dense_bge_m3_v2
python -m embed.search_dense "Điều kiện tốt nghiệp là gì?" --index data/indexes/dense_bge_m3_v2
```

Không sửa trực tiếp các snapshot JSONL trong index. Bộ mã kiểm tra hash để phát hiện index/metadata bị lệch. Khi nâng thư viện sau này, nên lưu môi trường đã chạy thành công (`python -m pip freeze`) cùng kết quả thí nghiệm; requirements hiện tại là khoảng phiên bản, chưa phải lockfile đã xác nhận trên máy bạn. Revision model được lưu từ cấu hình model nếu có; có thể chỉ định commit bằng `--revision` lúc build để tái lập chặt hơn.

## 5. Tìm thử

```powershell
python -m embed.search_dense "Sinh viên bị cảnh báo học tập trong trường hợp nào?" --top-k 10 --max-parents 5
python -m embed.search_dense "Học phí được tính như thế nào?" --top-k 5
python -m embed.search_dense "Điều kiện xét tốt nghiệp là gì?" --top-k 10
```

Output JSON gồm `children` (xếp giảm dần theo cosine) và `parents` (khử trùng parent theo thứ tự child tốt nhất). Số parent có thể ít hơn `max-parents` nếu top-k child cùng thuộc một số Điều. Điểm cosine không phải xác suất trả lời đúng; hiện chưa có ngưỡng từ chối cho câu hỏi ngoài tài liệu.

Để tích hợp trong Python:

```python
from embed.search_dense import DenseRetriever

# Khởi tạo một lần, tái sử dụng cho nhiều câu hỏi.
retriever = DenseRetriever("data/indexes/dense_bge_m3")
result = retriever.search("Học phí được tính như thế nào?", top_k=20, max_parents=5)
for hit in result["children"]:
    print(hit["child_id"], hit["parent_id"], hit["score"], hit["breadcrumb"])
contexts = [p["text"] for p in result["parents"]]
```

Luồng hybrid đã bổ sung theo slide (xem README_HYBRID.md): lấy danh sách child từ dense và BM25 → fuse theo `child_id` bằng RRF → rerank child → **sau đó** lookup và khử trùng parent. `parents` trả về ở đây chỉ phục vụ kiểm tra nhánh dense độc lập; đừng khử trùng parent trước bước RRF/rerank của pipeline hoàn chỉnh. Về sau cần giới hạn ngân sách token tổng của parent trước khi đưa vào LLM.

## 6. Chất lượng dữ liệu cần lưu ý

Có 88/225 child còn ký tự `|` và có lỗi OCR như `học phan`, `đảo tạo`. Hai child có `\\sum` hoặc `\\frac`; `d8_k1_b` đã có cả LaTeX và phần diễn giải học phí. Bộ mã giữ nguyên nội dung, chỉ NFC và strip ở biên trước khi embed, không tự sửa chữ hoặc công thức. Không xóa `|` toàn cục vì có thể làm hỏng bảng. Khi có câu hỏi về điểm/học phí, cần đối chiếu bản PDF để xác nhận công thức và số liệu trước khi báo cáo chất lượng.

Vì `split_articles.py` chủ yếu giới hạn theo ký tự rồi tách Khoản/Điểm, một số child vẫn vượt ngưỡng 1200 ký tự. Kiểm tra token trước encode giải quyết việc cắt ngầm, nhưng không thay thế việc rà lại ranh giới chunk nếu retrieval chưa tốt.

## 7. Kiểm thử và giới hạn xác minh

```powershell
python -m unittest discover -s tests -v
```

Tests kiểm tra roundtrip ghi/đọc FAISS, chuẩn hóa cosine, giữ metadata, lookup/khử trùng parent, ID trùng, parent thiếu, phát hiện snapshot bị sửa, input quá token. Tests dùng **vector xác định giả lập**, không tải BGE-M3 và không đánh giá khả năng hiểu tiếng Việt. Xem `VALIDATION.md` để biết các bước đã thực sự chạy.

Chưa đo Recall@k/MRR/nDCG vì chưa có `eval_set.json` được gửi. Bước tiếp theo sau khi chạy model thật là dùng tập câu hỏi gán nhãn Điều đúng của bạn để đo dense, rồi so với BM25/hybrid. Không kết luận BGE-M3 tốt nhất chỉ dựa vào slide hoặc một vài câu tìm thử.

Nguồn kỹ thuật:
- https://huggingface.co/BAAI/bge-m3
- https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
- https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances



# BM25 + Dense + RRF cho dự án RAG

Đọc file này trước. Bộ mã mới giữ nhánh BGE-M3 + FAISS cũ và bổ sung BM25, tìm riêng BM25, gộp hybrid. Chép các tệp trong `embed/` vào `RAG/embed/`. Đặc biệt thêm `bm25.py`, thay `build_bm25.py` chưa triển khai và thêm `search_hybrid.py`. Giữ các tệp khác của dự án; bạn không cần thay README.md hay xóa requirements.txt của mình.

## 1. Thêm thư viện vào requirements.txt hiện có

Giữ NumPy/Sentence Transformers/FAISS đã thêm từ lần trước. Chỉ bổ sung:

```text
rank-bm25==0.2.2
underthesea>=6.8,<10
```

Chạy trong Terminal tại thư mục RAG, với cùng môi trường Python đã dùng cho dense:

```powershell
python -m pip install -r requirements.txt
```

File `requirements-embedding.txt` trong gói là danh sách phụ thuộc toàn bộ nhánh dense + BM25 để tham khảo. Nếu đã ghép vào requirements.txt của bạn thì không cần cài riêng file này.

## 2. Tạo các index

Nếu bạn đã chạy dense thành công và **dữ liệu chưa đổi**, giữ index cũ, chỉ chạy BM25:

```powershell
python -m embed.build_bm25
```

Nếu chưa có dense thì chạy lần lượt:

```powershell
python -m embed.build_dense
python -m embed.build_bm25
```

Dense lưu ở `data/indexes/dense_bge_m3/`, BM25 ở `data/indexes/bm25/`. Không cần parse/chunk lại chỉ để thêm BM25. BGE-M3 chỉ tải lần đầu nếu chưa có cache; tạo BM25 không tải hoặc chạy BGE-M3.

Nếu index đích đã tồn tại, chương trình từ chối ghi đè. Chọn thư mục mới bằng `--output` để rebuild, rồi truyền đúng đường dẫn lúc search.

```powershell
python -m embed.build_bm25 --output data/indexes/bm25_v2
python -m embed.search_hybrid "Học phí được tính thế nào?" --bm25-index data/indexes/bm25_v2
```

## 3. Tìm kiếm hybrid

```powershell
python -m embed.search_hybrid "Học phí được tính như thế nào?" --top-k 10 --max-parents 5
python -m embed.search_hybrid "Sinh viên bị cảnh báo học tập trong trường hợp nào?"
```

Nếu muốn chỉ định CPU:

```powershell
python -m embed.search_hybrid "Điều kiện xét tốt nghiệp là gì?" --device cpu
```

Để so sánh từng nhánh:

```powershell
python -m embed.bm25 "Học phí được tính như thế nào?" --top-k 5
python -m embed.search_dense "Học phí được tính như thế nào?" --top-k 5
```

## 4. Cách nối với dense

1. Hai nhánh đều tìm trên cùng child: `breadcrumb + newline + text`.
2. Dense dùng BGE-M3 trên văn bản gốc. BM25 dùng NFC, lowercase, tách từ underthesea và bỏ dấu câu; giữ dấu tiếng Việt, số Điều và từ ghép có dấu gạch dưới. Corpus/query dùng cùng tokenizer, kiểm tra phiên bản underthesea khi nạp index.
3. Mỗi nhánh lấy tối đa `candidate_k=100` child. BM25 chỉ trả điểm dương; câu hỏi ngoài từ vựng không tạo danh sách xếp hạng ngẫu nhiên.
4. RRF gộp theo `child_id`, rank bắt đầu từ 1: `score = dense_weight/(60+dense_rank) + bm25_weight/(60+bm25_rank)`. Nhánh không chứa child đó đóng góp 0. Không cộng trực tiếp cosine và điểm BM25.
5. Chọn `top_k=10` child sau fusion, rồi lookup và khử trùng parent, lấy tối đa `max_parents=5`. Có thể có ít hơn 5 parent vì các child cùng thuộc một Điều.

Output chứa `children`, `parents`; mỗi child có `score` RRF, `dense_rank`, `bm25_rank`, `dense_score`, `bm25_score`. `score` parent là điểm RRF child tốt nhất. Điểm RRF không phải xác suất đúng và không có ngưỡng từ chối được hiệu chỉnh.

Mã đối chiếu nội dung/metadata giữa hai index. Nếu dữ liệu không khớp, cần rebuild từ cùng hai JSONL; không ghép index từ hai lần chunk khác nhau.

Bạn có thể thay đổi số ứng viên và trọng số để làm thí nghiệm:

```powershell
python -m embed.search_hybrid "Cảnh báo học tập" --candidate-k 100 --top-k 10 --rrf-k 60 --dense-weight 1 --bm25-weight 1
```

Các mặc định là baseline cần đánh giá; chưa chứng minh cấu hình tốt nhất trên dữ liệu này.

## 5. Tích hợp vào ứng dụng Python

```python
from embed.search_hybrid import HybridRetriever

# Khởi tạo một lần; không tải lại model mỗi câu hỏi.
retriever = HybridRetriever()
result = retriever.search("Học phí được tính thế nào?", candidate_k=100, top_k=10, max_parents=5)
contexts = [p["text"] for p in result["parents"]]
for hit in result["children"]:
    print(hit["child_id"], hit["dense_rank"], hit["bm25_rank"], hit["score"])
```

Chưa có cross-encoder reranker và LLM trong gói này. Khi thêm reranker, đặt nó **sau RRF, trước chọn top-k và lookup parent**. Parent vẫn cần giới hạn ngân sách token trước khi đưa vào prompt.

## 6. Chi tiết BM25 và lưu trữ

`rank-bm25` với `BM25Okapi(k1=1.2, b=0.65, epsilon=0.25)`. JSON lưu tokenized corpus, child/parent snapshots, cấu hình và hash. Khi load, tái dựng thống kê BM25 từ tokens; với 225 child chi phí nhỏ. Không lưu bằng pickle.

Mặc định dùng underthesea. Nếu cần thí nghiệm bỏ tách từ, chỉ định rõ và tạo index riêng; đây là baseline tách theo regex chứ không tương đương tách từ tiếng Việt:

```powershell
python -m embed.build_bm25 --tokenizer regex --output data/indexes/bm25_regex
```

Corpus và query tự dùng tokenizer ghi trong metadata, không âm thầm fallback. File index tạo bởi phiên bản underthesea khác yêu cầu rebuild để tránh lệch tokenizer.

## 7. Kiểm chứng

```powershell
python -m unittest discover -s tests -v
```

Xem `VALIDATION.md`: đã chạy BM25 thật trên 225 child và kiểm thử tích hợp với FAISS thật/vector dense giả lập. Chưa chạy BGE-M3 thật hoặc đo Recall@k/MRR/nDCG của hybrid. Không kèm model hoặc index dense tạo sẵn trong gói.

Tài liệu:
- https://github.com/dorianbrown/rank_bm25
- https://undertheseanlp.github.io/underthesea/docs/next/api/word-tokenize
- https://huggingface.co/BAAI/bge-m3
