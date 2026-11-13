# Advanced RAG trên ViAdminRAG

Đề tài Y06.1: tìm hiểu và áp dụng các kỹ thuật Advanced RAG cho bài toán hỏi–đáp văn bản hành chính, pháp lý và giáo dục tiếng Việt. Xây dựng Naive RAG làm baseline, thử từng module cải tiến rồi kết hợp và đánh giá.

## Dataset

Sử dụng [nhokjack/ViAdminRAG](https://huggingface.co/datasets/nhokjack/ViAdminRAG). Bản dữ liệu hiện tại gồm **400 câu hỏi thuộc lĩnh vực Giáo dục**, được xây dựng từ **50 tài liệu nguồn**. Mỗi mẫu có câu hỏi, đáp án chuẩn, nội dung tài liệu và các đoạn bằng chứng tham chiếu. Hai nhóm Tóm tắt (200 câu) và Hỏi đáp nhiều bước (200 câu) cho phép phân tích khả năng tổng hợp thông tin; độ khó gồm Dễ (90), Trung bình (194) và Khó (116).

Dữ liệu được tách thành **tài liệu để truy xuất** và **câu hỏi để chạy, đánh giá hệ thống**. Cách tổ chức này giúp các cấu hình Naive RAG và Advanced RAG dùng chung đầu vào, đồng thời cho phép thử nhiều cách chia đoạn trên cùng tài liệu gốc.

Tải và xử lý dữ liệu từ thư mục dự án, chỉ cần Python và thư viện chuẩn:

```powershell
python src/data_processing/download_dataset.py
python src/data_processing/prepare_dataset.py
```

```text
data/
  raw/viadminrag/             # Dữ liệu gốc, dataset card và manifest
  processed/
    documents.jsonl          # 50 tài liệu duy nhất để xây corpus
    queries.jsonl            # 400 câu hỏi kèm đáp án và bằng chứng
    preparation_report.json  # Kết quả kiểm tra dữ liệu
```

| File                | Trường                                        | Ý nghĩa trong thực nghiệm                                       |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| `documents.jsonl` | `doc_id`                                      | ID ổn định của tài liệu, tạo từ SHA-256 của nội dung      |
|                     | `content`, `source_url`                     | Toàn văn để chia đoạn, lập chỉ mục và truy vết nguồn    |
| `queries.jsonl`   | `query_id`, `input`                         | ID câu hỏi và truy vấn đưa vào hệ thống                    |
|                     | `expected_output`                             | Đáp án chuẩn để đánh giá câu trả lời sinh ra            |
|                     | `context`                                     | Các đoạn bằng chứng để đối chiếu kết quả truy xuất     |
|                     | `source_doc_id`                               | Liên kết câu hỏi với tài liệu nguồn                         |
|                     | `category`, `question_type`, `difficulty` | Phân nhóm kết quả theo chủ đề, loại câu hỏi và độ khó |

Dùng **toàn bộ 400 câu hỏi để đánh giá** các cấu hình RAG trên cùng corpus 50 tài liệu, không chia tập phát triển/đánh giá. Khi chạy RAG, chỉ dùng `input` làm truy vấn; `context`, `expected_output` và `source_doc_id` là thông tin tham chiếu để đánh giá, không dùng để chọn sẵn tài liệu hoặc đưa trực tiếp vào prompt.

## Kế hoạch thực nghiệm

### 1. Triển khai Naive RAG baseline

1. **Chốt cấu hình:** kiểm tra CPU, RAM và GPU để chọn embedding model và một model Generation chạy local qua Ollama. Ghi lại model, kích thước chunk, overlap, Top-k, prompt và tham số sinh.
2. **Chia tài liệu thành chunk:** đọc `documents.jsonl`, chia theo độ dài cố định và cho các đoạn liền kề chồng lấn một phần (overlap). Lưu `chunks.jsonl` với `chunk_id`, `doc_id`, `text`, vị trí bắt đầu/kết thúc trong tài liệu và nguồn. Chốt đơn vị đo độ dài cùng kích thước trước khi chạy.
3. **Tạo embedding và index:** chuyển nội dung từng chunk thành vector bằng embedding model đã chọn. Lưu vector index cùng ánh xạ từ vector tới `chunk_id` để sử dụng lại khi truy vấn.
4. **Truy xuất Dense Top-k:** lấy `input` của mỗi câu hỏi, tạo vector bằng cùng embedding model và tìm k chunk gần nhất trong toàn bộ index. Lưu ID chunk, thứ hạng, điểm tương đồng và thời gian truy xuất. Baseline dùng câu hỏi gốc, không rerank.
5. **Sinh câu trả lời:** ghép các chunk Top-k với câu hỏi theo một prompt cố định, gọi model Ollama chạy local. Yêu cầu trả lời dựa trên bằng chứng truy xuất, dẫn ID chunk và nêu rõ khi thiếu thông tin. Lưu câu trả lời, các chunk đã sử dụng và thời gian Generation.
6. **Chuẩn bị nhãn đánh giá Retrieval:** đối chiếu `context` tham chiếu với các chunk để tạo `qrels.jsonl` gồm `query_id`, `chunk_id`, `relevance`. Kiểm tra các trường hợp bằng chứng trải qua nhiều chunk hoặc không khớp; không mặc định khớp chuỗi chính xác. Nhãn này chỉ phục vụ đánh giá.
7. **Chạy và lưu kết quả baseline:** xử lý toàn bộ 400 câu hỏi, lưu kết quả theo `query_id` cùng cấu hình chạy để làm mốc so sánh cho các module Advanced RAG.

### 2. Thử từng module Advanced RAG

Mỗi cấu hình thêm một module vào baseline, giữ nguyên các thành phần còn lại để xác định đóng góp riêng.

| Module | Hướng triển khai dự kiến |
| --- | --- |
| Chunking nâng cao | Chia theo cấu trúc hoặc ngữ nghĩa |
| Xử lý query | Viết lại hoặc mở rộng câu hỏi trước khi truy xuất |
| Hybrid Search | Kết hợp BM25 và Dense Retrieval bằng RRF |
| Reranking | Xếp hạng lại các đoạn ứng viên |

Mỗi phương pháp chunking lưu bộ chunks và index riêng, đồng thời ánh xạ lại qrels tương ứng. Các cấu hình dùng chung tài liệu nguồn và bộ câu hỏi. Không triển khai truy xuất lặp nhiều bước.

### 3. Kết hợp các module

Tích hợp các module đã triển khai thành pipeline Advanced RAG hoàn chỉnh, có cấu hình bật/tắt từng module. Giữ cùng model Ollama, mẫu prompt và tham số Generation giữa baseline, từng module và cấu hình kết hợp; ngữ cảnh đầu vào thay đổi theo kết quả của mỗi pipeline.

### 4. Đánh giá và phân tích

Chốt trước các cấu hình và tham số cần so sánh, đánh giá trên toàn bộ 400 câu hỏi:

- **Retrieval:** Recall@k, MRR@k, nDCG@k và thời gian truy xuất, dựa trên qrels đã kiểm tra.
- **Generation:** độ chính xác so với `expected_output`, mức bám bằng chứng truy xuất và thời gian sinh câu trả lời. Chốt cách chấm trước khi chạy đánh giá.
- **So sánh:** baseline, baseline cộng từng module và pipeline kết hợp; phân tích theo loại câu hỏi và độ khó, kèm các trường hợp lỗi tiêu biểu.

## Chạy Naive RAG

Code baseline đã được tách thành các thành phần dùng chung để bổ sung Advanced RAG sau này:

```text
configs/naive.yaml          # Đường dẫn, chunking, model, Top-k và tham số sinh
src/
  chunking/fixed.py         # Chia đoạn cố định và lưu vị trí trong văn bản gốc
  retrieval/dense.py        # Embedding, lưu index và tìm kiếm cosine bằng NumPy
  ollama_client.py          # Kết nối HTTP tới Ollama
  generation.py            # Prompt và gọi model sinh câu trả lời
  pipeline.py              # Ghép Retrieval với Generation
experiments/run.py         # Lệnh chia đoạn, tạo index, hỏi và chạy bộ câu hỏi
tests/test_naive.py         # Kiểm thử offline, không gọi model thật
```

Cài thư viện và chia đoạn, chưa cần Ollama:

```powershell
python -m pip install -r requirements.txt
python experiments/run.py chunk
python -m unittest discover -s tests -v
```

Cấu hình khởi đầu dùng **1.200 ký tự/chunk, overlap 200 ký tự, Top-5**. Vị trí `start_char`, `end_char` là chỉ số ký tự Python, trong đó `end_char` không thuộc đoạn. Đây là đơn vị ký tự, không phải token của model. Bản dữ liệu hiện tại tạo **301 chunks** tại `data/processed/chunks/fixed.jsonl`.

Embedding dự kiến dùng `bge-m3:latest`, Generation dùng `qwen2.5:3b`, đều cấu hình được trong `configs/naive.yaml`. Đây là lựa chọn khởi đầu, chưa xác nhận phù hợp tài nguyên máy. Client dùng API chính thức của Ollama: [embed](https://docs.ollama.com/api/embed) và [chat](https://docs.ollama.com/api/chat).

Các lệnh thiết lập và chạy với dịch vụ Ollama đang hoạt động (chỉ cần tải model nếu máy chưa có):

```powershell
ollama pull bge-m3:latest
ollama pull qwen2.5:3b
python experiments/run.py check
python experiments/run.py index
python experiments/run.py ask --question "Quy định về học phí được nêu như thế nào?"
python experiments/run.py run --limit 3
python experiments/run.py run
```

Lệnh `ask` hiển thị tiếng Việt, tách câu trả lời, nguồn truy xuất và thời gian xử lý. Mặc định chỉ trích ngắn nội dung từng đoạn; dùng `--show-chunks` để xem đầy đủ hoặc `--format json` để lấy dữ liệu gốc. Trích dẫn `[1]`, `[2]` trên màn hình tương ứng thứ tự nguồn; JSON giữ ID chunk gốc.

`--limit 3` chỉ chạy kiểm tra nhanh; bỏ tùy chọn này để chạy đủ 400 câu. Thêm `--retrieval-only` vào `ask` hoặc `run` để bỏ qua Generation, nhưng vẫn cần model embedding. Mỗi lần chạy `run` lưu `predictions.jsonl` và manifest riêng tại `results/naive/<run_id>/`, gồm cấu hình, model digest, checksum dữ liệu, chunk truy xuất, câu trả lời và thời gian xử lý. Nếu chạy lỗi, giữ phần kết quả đã hoàn tất và đánh dấu lần chạy thất bại.

Index kiểm tra model embedding, cấu hình chia đoạn và checksum dữ liệu trước khi truy vấn; cần dựng lại khi nguồn hoặc cấu hình thay đổi. Embedding vượt giới hạn model và ngữ cảnh vượt ngân sách cấu hình sẽ báo lỗi thay vì tự cắt ngầm.

**Tiến độ:** đã có code baseline, 301 chunks và Dense index; đã kiểm thử offline và chạy thử câu hỏi qua Ollama. Chưa tạo qrels hoặc chạy đánh giá đầy đủ 400 câu hỏi.
