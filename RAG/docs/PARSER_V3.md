# Parser v3: khối nội dung có kiểu (code, công thức, hình, bảng)

> Parser v3.1.0 (`PARSER_VERSION = "3.1.0"`): thêm engine router tùy chọn, xem mục 10. Tên tệp giữ nguyên.

## 1. Mục đích

Parser v3 (`parser/parse_document.py`, `PARSER_VERSION = "3.0.0"`) chuyển PDF/DOCX/PPTX/HTML/MD/TXT/ảnh thành danh sách khối (`ContentBlock`) theo thứ tự đọc, giữ nguyên code (thụt dòng), công thức, hình và bảng thay vì làm phẳng thành văn bản. Mỗi khối lớn có thể kèm ảnh cắt (asset PNG) để frontend hiển thị. Chunker (`parser/chunker.py`) coi bảng/code/công thức/hình là khối nguyên tử và ghi lại `block_ids`, `assets`, `languages` vào child để truy ngược về khối gốc.

## 2. Schema khối (cho frontend)

Tệp `<store>/parsed/<doc_id>/<tên>.json`: `{schema_version:1, parser_version, source{filename,path,format,sha256,size_bytes}, stats, blocks[]}`.

| Trường | Ý nghĩa |
|---|---|
| `id` | `p0003_b00017` (PDF, trang 3), `s0002_b...` (slide), `doc_b...` (không có trang) |
| `type` | `heading` \| `paragraph` \| `list` \| `table` \| `code` \| `formula` \| `figure` \| `image_ocr` |
| `text` | nội dung (code giữ nguyên thụt dòng; bảng là markdown) |
| `level` | 0..3; chỉ có nghĩa với `heading` |
| `location` | `{"page_num":N}` hoặc `{"slide_num":N}` hoặc `{}` (DOCX/HTML/MD) |
| `breadcrumb` | chỉ có với văn bản pháp lý (`Chương > Điều > Khoản > Điểm`), còn lại `""` |
| `bbox` | `[x0,y0,x1,y1]` chuẩn hoá 0..1 theo trang (PDF, OCR); `null` nếu không có |
| `asset` | đường dẫn tương đối `assets/<block_id>.png` (so với thư mục `store/parsed/<doc_id>/`), hoặc `null` |
| `metadata` | `source`, và các khoá theo kiểu bên dưới; văn bản pháp lý có thêm `chapter/article/clause/point` |

Asset: công thức cắt 200 dpi, hình 150 dpi (lề 3 pt). Chỉ `formula` và `figure` (PDF) có asset. Tên cuối `<block_id>.png` được đặt sau khi có id.

Ví dụ (rút gọn, `bbox`/`source` có thể lược):

```json
{"id":"p0001_b00002","type":"heading","text":"2 Phương pháp","level":2,"location":{"page_num":1},"bbox":[0.1,0.2,0.5,0.23],"asset":null,"metadata":{"font_size":14.0,"bold":true,"source":"pymupdf"}}
{"id":"p0001_b00003","type":"paragraph","text":"Mô hình gồm hai bước.","level":0,"location":{"page_num":1},"asset":null,"metadata":{"source":"pymupdf"}}
{"id":"p0001_b00004","type":"list","text":"- mục một\n- mục hai","level":0,"location":{"page_num":1},"asset":null,"metadata":{}}
{"id":"p0002_b00009","type":"table","text":"| A | B |\n| --- | --- |\n| 1 | 2 |","location":{"page_num":2},"asset":null,"metadata":{"rows":[["A","B"],["1","2"]],"n_rows":2,"n_cols":2}}
{"id":"p0002_b00010","type":"code","text":"def f(x):\n    return x","location":{"page_num":2},"asset":null,"metadata":{"language":"python"}}
{"id":"p0003_b00011","type":"formula","text":"x̄ = Σ xi / n","location":{"page_num":3},"asset":"assets/p0003_b00011.png","metadata":{"format":"text"}}
{"id":"p0003_b00012","type":"figure","text":"Hình 1. Kiến trúc\n<chữ OCR trong hình>","location":{"page_num":3},"asset":"assets/p0003_b00012.png","metadata":{"kind":"raster","caption":"Hình 1. Kiến trúc","ocr_text":"..."}}
{"id":"s0004_b00020","type":"figure","text":"Doanh thu: Q1=10, Q2=12","location":{"slide_num":4},"asset":null,"metadata":{"kind":"chart","chart":{"type":"COLUMN_CLUSTERED","categories":["Q1","Q2"],"series":[{"name":"Doanh thu","values":[10,12]}]}}}
{"id":"p0005_b00030","type":"image_ocr","text":"Văn bản OCR của trang quét","location":{"page_num":5},"bbox":[0.1,0.1,0.9,0.2],"asset":null,"metadata":{"ocr":true,"ocr_confidence":0.91}}
```

Ghi chú theo kiểu:
- `heading`: PDF thêm `font_size`, `bold`; `level` 1..3.
- `list`: các mục nối bằng `\n`.
- `table`: PDF có `rows` (mảng 2 chiều), `n_rows`, `n_cols`; DOCX/PPTX/HTML/MD chỉ có `text` markdown (xuống dòng trong ô là `<br>`). Bảng từ OCR có `approximate_table: true`.
- `code`: `language` (có thể `""`; PDF chỉ đoán `python` nếu dòng đầu bắt đầu `>>> `, `shell` nếu `$ `; MD lấy từ fence, HTML từ `class="language-x"`).
- `formula`: `format` = `"text"` (chuỗi Unicode trích từ PDF). `"latex"` chỉ xuất hiện với MD (`$$...$$`, `\[...\]`) và HTML `<math>` có annotation TeX. PDF KHÔNG sinh LaTeX.
- `figure`: `kind` = `raster` | `vector` | `chart`; `caption` (đoạn "Hình/Figure/Biểu đồ/Chart N" gần nhất trong 40 pt), `ocr_text` (OCR chữ trong hình, bỏ nếu rác). `text` = caption + ocr_text, có thể rỗng. `chart` (PPTX): `metadata.chart = {type, categories, series[{name, values}]}`, không có asset.
- `image_ocr`: văn bản OCR từ trang quét hoặc ảnh nhúng (DOCX/PPTX/HTML ảnh nhúng có `embedded_in`).

## 3. Trường thêm của child (`corpus/children.jsonl`)

`block_ids` (danh sách khối nguồn), `bbox` (hợp các bbox cùng trang đầu), `content_type` (`text`|`table`|`code`|`formula`|`figure`|`mixed`), `assets` = `[{block_id, type, asset, bbox, page}]` (chỉ `formula`/`figure` có asset), `languages` (ngôn ngữ các khối code). CLI `--json` thêm `types`, `assets`, `languages` vào mỗi nguồn.

Cách frontend dựng hiển thị từ child:
1. Đọc `block_ids`, nạp JSON khối của `doc_id` và tra theo `id` (cùng nội dung có trên MinIO `blocks/<doc_id>` khi dùng `--stack`).
2. `code`: `<pre><code>` giữ thụt dòng, gắn nhãn `metadata.language` nếu có.
3. `formula`: hiện ảnh `parsed/<doc_id>/<asset>`; `text` (Unicode) làm `alt`/dự phòng.
4. `table`: dựng từ `metadata.rows` hoặc render markdown `text`.
5. `figure`: ảnh từ `asset` + `caption`; nếu `kind=="chart"` vẽ từ `metadata.chart`.
LaTeX chưa được sinh: không hiển thị công thức PDF bằng KaTeX/MathJax.

## 4. Quy tắc trích xuất theo định dạng

- PDF số: PyMuPDF `rawdict`, giữ thứ tự thô của MuPDF (không sort, vì sort hỏng trang 2 cột). Khoảng cách ký tự >= 0.20 x cỡ chữ được chèn dấu cách (PDF thiếu glyph space). Chân trang (số trang) ở 10% dưới bị bỏ.
- Heading: cỡ chữ >= 1.2 x cỡ thân (<= 3 dòng) hoặc in đậm 1 dòng; <= 120 ký tự, không kết thúc bằng `.`. Slide (ngang): khối chữ đầu trong 20% trên, cỡ >= 1.1 x thân là heading cấp 1.
- Code: dòng phông monospace (courier, consolas, `CMTT`...) cỡ >= 7, trong khoảng 5%-92% chiều cao trang, chạy >= 2 dòng gần nhau hoặc 1 dòng có ký tự `=(){};$>#`. Thụt dòng tính theo độ rộng ký tự.
- Công thức: dòng "hạt giống" dùng phông toán (CMMI, CMSY, MSAM, `*Math`...) hoặc CMR/CMSS có ký hiệu toán; dòng có từ >= 4 chữ cái (hoặc 2 từ >= 3 chữ) là văn bản. Gộp cụm bằng union-find trên bbox nở ra; giữ nếu >= 2 dòng, nhiều ký hiệu (>= 25%) hoặc đứng riêng trong block; tối đa 20 cụm/trang.
- Hình: raster nếu ảnh >= 0.8% diện tích trang (OCR chữ nếu có Tesseract); vector nếu cụm vẽ >= 3% trang, >= 5 nét, có nét viền/đường cong, không phải bảng hay chữ nền; bỏ qua trang > 5000 drawing.
- Bảng: `page.find_tables()` (>= 2 hàng, >= 2 cột, có ô không rỗng); ô được dựng lại từ chữ có khoảng cách để khỏi mất dấu cách.
- Trang quét (chữ < 20 ký tự, hoặc ảnh > 60% trang và chữ < 250): Tesseract (`--oem 1 --psm 6`, `vie+eng`), render tối đa 300 dpi / 2400 px cạnh dài, ra `image_ocr`, hàng căn cột >= 3 hàng thành bảng xấp xỉ.
- Markdown/TXT: fence ``` / `~~~` thành `code` (nguyên văn), `$$..$$`/`\[..\]` thành `formula` latex, bảng `|`, heading `#`.
- HTML: `<pre>` thành code, `<math>` thành formula, `<table>`, `<img>` (OCR), bỏ `script/style/noscript/nav`.
- PPTX: theo thứ tự trên-trái; placeholder TITLE hoặc dòng đầu chữ >= 24 pt thành heading 1; nhóm shape duyệt đệ quy; bảng; ảnh OCR; biểu đồ gốc đọc từ XML thành `figure` kind `chart`.
- DOCX: đoạn theo thứ tự run, style Heading/Title, bảng, ảnh nhúng OCR (không có trang/bbox).

## 5. Quy tắc làm sạch

NFC; bỏ ký tự điều khiển, zero-width, soft hyphen; ligature `ﬁ` thành `fi`; bỏ thẻ `<mark|br|sup|sub|u>` và chú thích HTML; gộp khoảng trắng; nhiều dòng trống thành 1. Code và công thức latex KHÔNG qua làm sạch (`verbatim`). Nối dòng PDF: bỏ gạch cuối dòng chỉ khi mảnh trước là chữ thường (`state-of-the-` giữ gạch). Bỏ đầu/chân trang lặp lại (>= 3 trang; xuất hiện ở >= max(3, 45% số trang), vùng bbox y < 0.11 hoặc > 0.89, <= 100 ký tự) và số trang riêng lẻ; không bao giờ bỏ heading/code/formula/figure/table.

## 6. Quy tắc chunker cho kiểu mới

- `table/code/formula/figure` là nguyên tử: chỉ cắt khi một khối vượt ceiling, cắt theo dòng/hàng; bảng lặp lại 2 dòng tiêu đề nếu tiêu đề <= 1/3 ceiling. Code giữ thụt dòng của dòng đầu (`rstrip("\n")`, không `strip`).
- Giới hạn: ceiling 400 token (child), min 80 (gộp nhóm nhỏ vào nhóm kề nếu tổng <= ceiling), hard_max 1000 (vượt thì `ChunkingError`); parent tối đa 2000 token, section dài hơn bị cắt thành cửa sổ ~1500.
- Hình không chữ nhưng có asset vẫn được giữ trong section; nhóm chỉ gồm hình rỗng được gộp vào nhóm trước (hoặc sau); cửa sổ toàn hình rỗng không tạo child.
- Child lưu `content_type`, `assets`, `languages`, `block_ids`; đủ phủ >= 90% khối có chữ, nếu không báo lỗi.

## 7. Giới hạn đã biết

- LaTeX chưa được sinh cho PDF (công thức chỉ là Unicode + ảnh).
- Biểu đồ dạng ảnh không có giá trị số: chỉ ảnh + caption + OCR; KHÔNG index giá trị ước lượng. Biểu đồ PPTX gốc thì đọc đúng từ XML.
- Bảng có ô gộp hàng (row-span) bị làm phẳng: câu trích nguyên văn e1 Bảng 5 không còn literal; S3-d-009 đổi từ đúng sang sai.
- Đầu/chân trang lặp lại như dấu arXiv có thể bị nhận là heading.
- Gạch cuối dòng với từ viết hoa vẫn còn (`Ta-ble`).
- Cắt ảnh bị mất mẫu số ở slide trang 94/95.
- Lỗi cấp trang: rơi về văn bản thuần của trang đó (`page_errors` trong stats).
- Tesseract cần `TESSDATA_PREFIX` trỏ thư mục có `vie.traineddata` (hoặc `OCR_LANG`/`TESSERACT_CMD`).
- `PARSER_VERSION` 3.0.0 làm cache cũ mất hiệu lực (cache dựa trên sha256 + phiên bản + asset còn đủ).
- Tài liệu không ra khối nào thì `execute` ném lỗi (ví dụ `g3__table_test.pptx` với bảng toàn ô rỗng).
- Metadata OCR còn ghi `source: "rapidocr"` dù engine là Tesseract (nhãn cũ).

## 8. Bằng chứng đánh giá (đo trên mẫu nhỏ)

- Cổng truy xuất S2 quote-hit@3: 0,933 -> 0,933 (28/30).
- S3: 0,913 -> 0,957 (22/23). Theo loại: formula 0,67 -> 1,0; multi_section 0,5 -> 1,0; table_lookup 1,0 -> 0,75; một lật S3-d-009.
- Hồi quy parser: 20/21 tài liệu mẫu ok (g3 rỗng theo thiết kế).
- Khối formula: e2 14 -> 1, e1 18 -> 10. Số child a2: 60 -> 50 (do cấu trúc văn bản pháp lý nhiều danh sách, không phải mất chữ: mất chữ đo được <= 0,7% mỗi trang ở e2, <= 0,1% ở a2).
- Nghiệm thu PDF slide: "Công thức tính trung bình cộng là gì?" trả lời đúng kèm công thức; "Trung vị lương" = 54 đúng; câu hỏi bảng = 230 đúng; câu hỏi trung bình của Messi vẫn từ chối (lỗi truy xuất, mục trung bình xếp hạng 6-7).

## 10. Engine router (v3.1.0)

**Mục đích.** Dùng engine OCR phụ chạy trên GPU để bổ sung cho parser: công thức -> LaTeX, bảng dạng ảnh -> bảng markdown. Tùy chọn: thiếu venv/trọng số thì engine `skipped` và kết quả y hệt `--engines off`.

**Kiến trúc.** Mỗi engine là một tiến trình riêng (`engines/<tên>/worker.py`) trong venv riêng, giao tiếp JSON lines qua stdin/stdout, khởi động lười, giữ ấm trong một lần ingest, đóng trước khi embedding (giải phóng VRAM). Router: `parser/engine_router.py`. Cài đặt: `engines/README.md`.

**Cờ và env.** `--engines off|auto|glm|docling|glm,docling` (`auto` = chỉ GLM; Docling là opt-in), `--engine-budget` (mặc định 45 s). Env: `RAG_ENGINES`, `RAG_ENGINE_BUDGET`, `RAG_ENGINE_GLM_PYTHON`, `RAG_ENGINE_DOCLING_PYTHON`, `RAG_ENGINE_TRACE=1`.

**Ngân sách.** `hiệu_dụng = min(budget, 120 - đã_trôi - 0,28 * số_trang)`; dưới 15 s thì bỏ qua. Hạn cứng = hiệu dụng + 5 s, áp lên cả thời gian chờ handshake lẫn từng request (spawn + nạp mô hình tính vào `used_s`). Nếu ngân sách còn lại nhỏ hơn thời gian nạp đo/seed của engine thì không khởi động. Docling bị bỏ qua nếu chi phí dự kiến > 40% ngân sách còn lại.

**Chính sách theo vùng.**
- Crop công thức: GLM -> LaTeX lưu ở `metadata.latex`; `text` vẫn là Unicode (chỉ thay bằng LaTeX khi Unicode có < 3 ký tự chữ-số). Crop tầm thường (<= 3 ký tự thường, hoặc cao < 12 pt) bỏ qua; cần >= 3 công thức đáng xử lý. LaTeX bị loại nếu rỗng, quá dài, ngoặc lệch, lặp, hoặc lệch quá nhiều với Unicode (`latex_flags: ["rejected:..."]`).
- Bảng số trong ảnh: GLM trả bảng HTML, router đổi sang markdown, chỉ nhận khi >= 2 hàng, >= 2 cột và >= 30% từ/số của OCR ảnh có trong bảng (hoặc ảnh không có OCR). Nhận thì thành khối `table` mới ngay sau khối nguồn (`id` = `<nguồn>_e1`, `metadata.engine = "glm"`, `diacritics_risk = true`). Tối đa 3 mỗi tài liệu; không gửi request khi ngân sách còn lại < 25 s.
- Bảng PDF số có nghi vấn (ô trống/độ lấp đầy thấp): Docling, chỉ khi opt-in; thay thế nếu khớp vị trí (IoU >= 0,5) và >= 90% từ có trong bảng gốc (`metadata.engine = "docling"`, hàng cũ ở `parser_rows`).
- Trang quét: chỉ Tesseract. Biểu đồ: không engine.

**Cache** ở `<store>/engine_cache` (khóa = sha256 dữ liệu + engine + phiên bản + tham số); thống kê thời gian ở `engine_stats.json`. **Thống kê**: `stats.engines` (mode, budget_s, effective_s, used_s, từng engine: state, s, formula/table done/cached/rejected/skipped), bảng tổng kết của `ingest` có dòng `engine glm ok formulas 36/36 ...` và `engine docling: opt-in`. **Schema thêm**: `metadata.latex`, `metadata.engine`, `metadata.latex_flags`, `metadata.engine` ở khối table, `stats.engines_mode` (đổi chế độ thì cache parse mất hiệu lực).

**Cache parse và phiên bản.** Cache parse chỉ trúng khi `engines_mode` khớp: cài engine sau đó hoặc đổi chế độ thì cần `--force` (hoặc tự parse lại); cache trúng trả về `stats.engines` cũ. Khóa cache engine dùng chuỗi `ENGINE_VERSION` tĩnh: tăng trong `engine_router.py` khi prompt worker hoặc revision mô hình đổi. Chế độ mặc định của CLI/ingest là `auto` (khởi động GLM nếu venv tồn tại; dùng `--engines off` để tắt). `--engine-budget` phải hữu hạn và >= 0 (0 = bỏ qua engine).

**Số đo** (RTX 4050 6 GB, PDF slide `Data-understanding-vi.pdf`, 104 trang):
- Lần đo trước: off 73 s, auto 113 s (lần 2-3); GLM 36/36 công thức nhận; engine 38,9 s so với ~42 s hiệu dụng. Lần đầu (nguội) mất 175 s vì chờ ready 46 s; nay chờ ready bị chặn ở min(120, còn lại + 5 s).
- Đo lại sau chỉnh sửa (máy bận hơn, số tuyệt đối dao động): slide off 106 s, auto 128 s (engine 24,4 s, 36/36); e2 bert off 63 s, auto 62 s (không đủ công thức: GLM skipped); a2 pháp lý off 59 s, auto 61 s (không ứng viên). Lần đo trước e2: auto 82 s do Docling spawn 25 s rồi timeout; nay Docling không chạy trong `auto`.
- Đồng nhất off vs auto (slide, e2, a2): mọi khối không phải formula/table giống hệt (id, type, text, metadata) trừ khóa engine; 0 vi phạm. Slide: 36 formula, text đổi 1 (Unicode < 3 ký tự chữ-số), khối mới 0; children khác 1/108 (chính child chứa formula đó); e2, a2: 0 child khác.
- Bảng ảnh GLM trên slide (thăm dò riêng, ngân sách 100 s): trang 27 nhận (bảng 7 cột), trang 96 nhận (bảng phân phối), trang 85 bị loại do OCR lệch (OCR của biểu đồ phân tán nhiễu). Nguyên nhân kết quả bị bỏ trước đây: GLM trả HTML `<table>`, bộ lọc cũ chỉ nhận ký tự `|`. Mỗi request 5-17 s. Trong `ingest` mặc định (45 s), công thức dùng hết ~25 s nên còn < 25 s và bảng ảnh không được gửi; muốn dùng cần tăng `--engine-budget`.

**Giới hạn đã biết.** Khởi động nguội (nạp mô hình lần đầu) chậm hơn số đo ấm; chưa mô phỏng được cold start thật, chỉ có test với worker giả (boot 40 s không vượt budget + 5 s). VRAM 2,1-3,2 GB. Bảng do GLM đọc có nguy cơ rơi/sai dấu tiếng Việt (`diacritics_risk`). Lợi ích của Docling chưa được chứng minh (chưa sửa được bảng row-span e1 Bảng 5 / S3-d-009). Giấy phép của mô hình/trọng số chưa được kiểm tra. Trọng số phải tải trước (worker chạy `HF_HUB_OFFLINE=1`). Chưa đo cold start thật và chưa đo lại ingest có bảng ảnh với budget lớn.

## 9. Kết luận benchmark mô hình (chỉ chạy cục bộ)

RTX 4050 6 GB, một lần chạy, mẫu nhỏ. Chi tiết ở `RAG_lab\results\ocr_compare\MODELS_CONCLUSION.md` (tệp nghiên cứu cục bộ, KHÔNG nằm trong git).

- Tesseract (giữ cho trang quét): ~1,5 s/trang, 80 s cho 53 trang; GLM-OCR ~1030 s và làm rơi dấu tiếng Việt.
- GLM-OCR 0.9B: công thức 20/24 khớp chính xác (1,0 s/crop, 2,2 GB); bảng tốt; giữ thụt dòng code; biểu đồ chỉ xấp xỉ giá trị.
- Texify: công thức 18/24 (0,34 s/crop, 0,65 GB), cần `transformers==4.44.2`.
- Docling mặc định: bảng và LaTeX tốt, 2,4 s/trang. Granite-Docling kém hơn với tiếng Việt.
- Qwen3-VL-2B: giá trị biểu đồ sai/bịa.
- Khuyến nghị: chưa thêm mô hình vào pipeline. Nếu cần LaTeX, thêm GLM-OCR hoặc Texify chỉ trên crop công thức. Biểu đồ: ảnh + caption; biểu đồ PPTX gốc đọc từ XML.
