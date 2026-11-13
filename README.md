# Advanced RAG trên ViAdminRAG

Đề tài Y06.1: tìm hiểu và áp dụng các kỹ thuật Advanced RAG cho bài toán hỏi–đáp văn bản hành chính, pháp lý và giáo dục tiếng Việt. Xây dựng Naive RAG làm baseline, thử từng module cải tiến rồi kết hợp và đánh giá.

## Dataset

Sử dụng [nhokjack/ViAdminRAG](https://huggingface.co/datasets/nhokjack/ViAdminRAG). Bản dữ liệu đang dùng gồm **400 câu hỏi thuộc lĩnh vực Giáo dục**, được xây dựng từ **50 tài liệu nguồn**. Câu hỏi gồm Tóm tắt (200 câu) và Hỏi đáp nhiều bước (200 câu); độ khó gồm Dễ (90), Trung bình (194) và Khó (116).

Mỗi mẫu dữ liệu gốc gồm các trường:

| Trường                                        | Nội dung                                      |
| ----------------------------------------------- | ---------------------------------------------- |
| `ID`                                          | Mã định danh câu hỏi                      |
| `Content`, `Source URL`                     | Toàn văn tài liệu và đường dẫn nguồn |
| `Category`, `Question type`, `Difficulty` | Lĩnh vực, loại câu hỏi và độ khó      |
| `input`                                       | Câu hỏi đầu vào                           |
| `expected_output`                             | Câu trả lời tham chiếu                     |
| `context`                                     | Các đoạn bằng chứng tham chiếu           |

Dữ liệu được chuẩn hóa thành **tài liệu để truy xuất** và **câu hỏi để chạy, đánh giá hệ thống**. Tải và xử lý từ thư mục dự án:

```powershell
python src/data_processing/download_dataset.py
python src/data_processing/prepare_dataset.py
```

```text
data/
  raw/viadminrag/             # Dữ liệu gốc và thông tin nguồn
  processed/
    documents.jsonl          # 50 tài liệu duy nhất để xây corpus
    queries.jsonl            # 400 câu hỏi kèm đáp án và bằng chứng
    preparation_report.json  # Kết quả kiểm tra dữ liệu
```

| File                | Trường chính                                     | Vai trò                                                                             |
| ------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `documents.jsonl` | `doc_id`, `content`, `source_url`             | Chia đoạn, lập chỉ mục và truy vết nguồn; ID tạo từ SHA-256 của nội dung |
| `queries.jsonl`   | `query_id`, `input`                             | Xác định câu hỏi và truy vấn đầu vào                                       |
| `queries.jsonl`   | `expected_output`, `context`, `source_doc_id` | Đáp án, bằng chứng và tài liệu nguồn dùng để đánh giá                 |
| `queries.jsonl`   | `category`, `question_type`, `difficulty`     | Phân nhóm kết quả                                                                |

Tập thực nghiệm mục tiêu là toàn bộ 400 câu hỏi, không chia tập phát triển/đánh giá. Khi chạy RAG, chỉ dùng `input` làm truy vấn; đáp án, bằng chứng và ID tài liệu tham chiếu không được đưa vào pipeline để chọn sẵn nguồn. Hiện có qrels nháp cho 229 câu để chấm Retrieval; chưa có tập kiểm tra độc lập để xác nhận khả năng khái quát.

Ngoài ra, `queries_unrelated.jsonl` chứa 100 câu hỏi gần miền giáo dục để kiểm tra khả năng từ chối khi thiếu bằng chứng. Tập này chưa thuộc benchmark mặc định; nhãn không trả lời được từ corpus vẫn cần rà soát.

## Kế hoạch thực nghiệm

### 1. Triển khai Naive RAG baseline

Baseline chia tài liệu thành các đoạn **1.200 ký tự, overlap 200 ký tự**, tạo 301 chunks tại `data/processed/chunks/fixed.jsonl`. Dùng embedding `bge-m3:latest` qua Ollama và tìm **Dense Top-5** bằng cosine similarity. Câu hỏi gốc được dùng trực tiếp để truy xuất, không viết lại hoặc rerank.

Generation dùng `qwen2.5:3b`, ghép câu hỏi với các đoạn truy xuất theo một prompt cố định. Model được yêu cầu trả lời bằng tiếng Việt, dẫn ID đoạn và nêu rõ khi thiếu bằng chứng. Cấu hình tại `configs/naive.yaml`, với `temperature: 0`, `seed: 42`, `num_ctx: 8192` và `num_predict: 512`.

Chuẩn bị thư viện, model và index khi dịch vụ Ollama đang hoạt động:

```powershell
python -m pip install -r requirements.txt
ollama pull bge-m3:latest
ollama run qwen2.5:3b
python experiments/run.py chunk
python experiments/run.py check
python experiments/run.py index
```

Chạy một câu hỏi hoặc toàn bộ tập dữ liệu:

```powershell
python experiments/run.py ask --question "Quy định về học phí được nêu như thế nào?"
python experiments/run.py run
```

→ Lệnh `ask` hiển thị câu trả lời, nguồn và thời gian xử lý.

→ Lệnh `run` lưu `predictions.jsonl` và `manifest.json` tại `results/<tên_cấu_hình>/<run_id>/`.

### 2. Thử từng module Advanced RAG

Mỗi cấu hình thêm một module vào baseline, dùng chung chunks, Dense index và qrels để so sánh đóng góp của từng kỹ thuật.

| Cấu hình          | Module và cách triển khai                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `advanced_query`  | Qwen viết lại câu hỏi một lần trước truy xuất; Generation vẫn nhận câu hỏi gốc                                        |
| `advanced_hybrid` | Kết hợp BM25 và Dense bằng RRF (`rrf_k: 60`), lấy tối đa 20 ứng viên từ mỗi retriever với cấu hình Top-5 hiện tại |
| `advanced_rerank` | Dense Top-20 → Cross-Encoder `BAAI/bge-reranker-v2-m3`(CPU) → Top-5                                                             |

Chuẩn bị model Reranker:

```powershell
python -m pip install -r requirements-advanced.txt
python experiments/prepare_reranker.py
```

Chọn cấu hình bằng `--config`, ví dụ:

```powershell
python experiments/run.py run --config configs/advanced_hybrid.yaml
```

### 3. Kết hợp các module của Advanced RAG

Cấu hình **`advanced_all`** kết hợp viết lại query → Hybrid Search → rerank → sinh đáp án. Đây là tên dùng trong code của pipeline kết hợp được gọi là `advanced_full`.

Để so sánh năm cấu hình, dùng `python experiments/benchmark.py --generation`. Mỗi cấu hình được chạy tuần tự và lưu kết quả riêng.

### 4. Giao diện web local

FastAPI phục vụ web và API trên cùng cổng 8000, hỗ trợ chọn một trong năm pipeline, Top-k và chế độ chỉ truy xuất hoặc sinh câu trả lời. Giao diện hiển thị các đoạn nguồn, thời gian và cho tải kết quả JSON. Mỗi lượt hỏi độc lập, không gửi lịch sử hoặc đáp án tham chiếu vào model.

Sau khi chuẩn bị model và index, chạy từ thư mục dự án:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Mở **http://localhost:8000**; tài liệu API tại **http://localhost:8000/docs**. Việc mở trang không tự chạy suy luận, tạo index hoặc tải model. Giữ một worker để khóa xử lý một lượt suy luận tại một thời điểm.

### 5. Đánh giá và phân tích

So sánh baseline, từng module và cấu hình kết hợp theo hai nhóm:

- **Retrieval:** Recall@5, MRR@5, nDCG@5, Hit@5 và thời gian truy xuất; phân tích thêm theo loại câu hỏi và độ khó.
- **Generation:** Token F1 và thời gian sinh.

| Metric                | Ý nghĩa                                                                                                                                                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recall@5              | Tỷ lệ chunk liên quan được tìm thấy trong Top-5 so với toàn bộ chunk liên quan theo qrels của câu hỏi. Đo mức độ thu thập đủ bằng chứng.                                                                          |
| MRR@5                 | Nghịch đảo thứ hạng của chunk liên quan đầu tiên trong Top-5; bằng 0 nếu không tìm thấy. Đo khả năng đưa bằng chứng đúng lên đầu.                                                                              |
| nDCG@5                | Điểm chất lượng thứ hạng trong Top-5, ưu tiên chunk liên quan ở vị trí cao và chuẩn hóa theo thứ hạng lý tưởng.                                                                                                      |
| Hit@5                 | Tỷ lệ câu hỏi có ít nhất một chunk liên quan trong Top-5; không yêu cầu tìm đủ mọi bằng chứng.                                                                                                                          |
| Thời gian truy xuất | Số giây trung bình để embedding câu hỏi và tìm ứng viên; thời gian viết lại query và rerank được báo cáo riêng.                                                                                                      |
| Token F1              | Trung bình điều hòa giữa precision và recall của token trùng giữa câu trả lời và đáp án tham chiếu, có xét số lần xuất hiện. Chỉ đo trùng khớp văn bản, chưa phản ánh đầy đủ độ đúng ngữ nghĩa. |
| Thời gian sinh       | Số giây trung bình để model sinh câu trả lời từ câu hỏi và bằng chứng đã truy xuất.                                                                                                                                      |

Trước khi đánh giá, Qrels được xây bằng cách ánh xạ `context` tham chiếu vào các chunk, chỉ tự động nhận khi từng bằng chứng khớp duy nhất sau chuẩn hóa khoảng trắng và được bao phủ đầy đủ.

Để chạy và chấm tự động năm cấu hình:

```powershell
python experiments/evaluate_suite.py --generation --output-dir results/evaluation_full
```

## Kết quả thực nghiệm

### Chất lượng truy xuất và câu trả lời

| Chỉ số | `naive` | `advanced_query` | `advanced_hybrid` | `advanced_rerank` | `advanced_all` |
| -------- | --------- | ------------------ | ------------------- | ------------------- | ---------------- |
| Recall@5 | 0.8464    | 0.7838             | 0.8661              | 0.8967              | **0.9041** |
| MRR@5    | 0.7631    | 0.7031             | 0.8119              | 0.8544              | **0.8635** |
| nDCG@5   | 0.7407    | 0.6832             | 0.7894              | 0.8310              | **0.8402** |
| Hit@5    | 0.9345    | 0.8734             | 0.9345              | 0.9563              | **0.9607** |
| Token F1 | 0.6022    | 0.5992             | 0.6082              | 0.6060              | **0.6124** |

### Thời gian trung bình (giây/câu)

| Giai đoạn     | `naive`      | `advanced_query` | `advanced_hybrid` | `advanced_rerank` | `advanced_all` |
| --------------- | -------------- | ------------------ | ------------------- | ------------------- | ---------------- |
| Xử lý query   | —             | 0.22               | 0.00                | —                  | 0.22             |
| Truy xuất      | 0.02           | 0.02               | 0.03                | 3.03                | 3.04             |
| Rerank          | —             | —                 | —                  | 4.93                | 4.94             |
| Sinh đáp án  | 0.85           | 0.80               | 0.87                | 4.19                | 4.20             |
| **Tổng** | **0.87** | **1.04**     | **0.89**      | **12.15**     | **12.40**  |

### Nhận xét

- **Hybrid phù hợp với dữ liệu có nhiều tên riêng và số hiệu văn bản.** Các tài liệu giáo dục thường dùng chung những cụm như “Bộ Giáo dục và Đào tạo”, “Thông tư” hoặc “quy định”, nên gần nhau về ngữ nghĩa dù nói về các văn bản khác nhau. BM25 bổ sung tín hiệu từ số hiệu, tên môn học và thuật ngữ xuất hiện trực tiếp trong câu hỏi, giúp hạn chế việc Dense chỉ chọn đoạn cùng chủ đề nhưng sai văn bản.
- **Reranking giúp phân biệt các đoạn cùng chủ đề nhưng khác nội dung cần trả lời.** Trong corpus, nội dung về Giáo dục công dân và Giáo dục kinh tế và pháp luật lặp lại ở nhiều bài viết, trong khi câu hỏi có thể yêu cầu vai trò, mục tiêu hoặc đối tượng áp dụng. Cross-Encoder xét trực tiếp từng cặp câu hỏi–đoạn để xếp lại ứng viên theo yêu cầu cụ thể. Chi phí xử lý cao hơn vì phải chấm nhiều cặp văn bản trên CPU.
- **Viết lại query chưa hiệu quả vì có thể làm mất chi tiết phân biệt trong câu hỏi gốc.** Chẳng hạn, ở `QA_0570`, Qwen bỏ phần “bao gồm Đạo đức, Giáo dục công dân và Giáo dục kinh tế và pháp luật”, chỉ giữ tên chung “môn Giáo dục công dân”. Trong kết quả local, Naive tìm được chunk có nhãn liên quan nhưng truy vấn viết lại thì không. Chi tiết bị bỏ vốn giúp nhận diện đoạn nói về vai trò xuyên suốt các cấp học; đây là một cơ chế lỗi có thể giải thích kết quả của module, chưa phải nguyên nhân đã xác nhận cho mọi câu.

### Ví dụ phân tích truy xuất

**Ví dụ 1 — `QA_0572`: phân biệt đoạn mô tả chung với đoạn trả lời đúng trọng tâm**

> Hãy tóm tắt các đặc điểm và mục tiêu của môn Giáo dục kinh tế và pháp luật ở giai đoạn giáo dục định hướng nghề nghiệp.

- **Naive:** Top-5 không có chunk được gán nhãn. Đoạn đứng đầu nói về môn học lựa chọn và các ngành nghề liên quan, nhưng bỏ lỡ đoạn tham chiếu mô tả giai đoạn giáo dục định hướng nghề nghiệp.
- **Advanced full:** đưa chunk có nhãn lên hạng 1. Đoạn này nêu rõ môn học được lựa chọn theo nguyện vọng, định hướng nghề nghiệp và giới thiệu nội dung kinh tế, pháp luật phù hợp với lứa tuổi.
- **Insight:** tên môn học xuất hiện ở nhiều đoạn và nhiều bài viết, nên chỉ khớp chủ đề chưa đủ. Với truy vấn đã viết lại, chunk đúng đứng hạng 12 ở Dense và hạng 3 ở BM25; sau RRF đứng hạng 5, rồi được rerank đưa lên đầu. Trường hợp này cho thấy tín hiệu từ khóa và bước đánh giá cặp câu hỏi–đoạn giúp ưu tiên bằng chứng sát trọng tâm hơn.

**Ví dụ 2 — `QA_0530`: tìm đúng đoạn quy định trách nhiệm của từng chủ thể**

> Theo quy định về phổ cập giáo dục và giáo dục bắt buộc trong Luật Giáo dục 2019 (sửa đổi), nhà nước, công dân và gia đình có những trách nhiệm gì?

- **Naive:** không có chunk được gán nhãn trong Top-5. Đoạn đứng đầu trong kết quả tìm kiếm ở một bài viết khác có nội dung tương tự về nghĩa vụ học tập và trách nhiệm gia đình, nhưng không phải bằng chứng từ tài liệu nguồn.
- **Advanced full:** đưa hai chunk có nhãn vào hạng 2 và 4. Các đoạn này chứa phần quy định trách nhiệm của Nhà nước, công dân và gia đình trong bài viết nguồn về giáo dục bắt buộc.
- **Insight:** một chunk đúng đứng hạng 2 ở BM25 nhưng không xuất hiện trong Top-20 Dense dùng để gộp; sau RRF đứng hạng 11 và được rerank đưa lên hạng 2. Hybrid giúp bổ sung bằng chứng mà Dense bỏ lỡ, còn rerank đưa bằng chứng đó vào ngữ cảnh cuối. Đồng thời, việc Naive lấy được đoạn gần tương đương ở bài khác cho thấy “trượt qrels” không đồng nghĩa câu trả lời chắc chắn sai.
