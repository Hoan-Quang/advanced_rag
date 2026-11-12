# Advanced RAG trên ViAdminRAG

Đề tài Y06.1: so sánh Naive RAG, Hybrid Search và Reranking cho bài toán hỏi–đáp văn bản hành chính, pháp lý và giáo dục tiếng Việt.

## Dataset

Sử dụng [nhokjack/ViAdminRAG](https://huggingface.co/datasets/nhokjack/ViAdminRAG), gồm **50 tài liệu và 400 câu hỏi** thuộc lĩnh vực Giáo dục: 200 câu Tóm tắt, 200 câu Hỏi đáp nhiều bước; độ khó gồm 90 câu Dễ, 194 câu Trung bình và 116 câu Khó.

Tải và xử lý dữ liệu từ thư mục dự án, chỉ cần Python và thư viện chuẩn:

```powershell
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

```text
data/
  raw/viadminrag/             # Dữ liệu gốc, dataset card và manifest
  processed/
    documents.jsonl          # doc_id, content, source_url
    queries.jsonl            # Câu hỏi và thông tin tham chiếu
    preparation_report.json  # Kết quả kiểm tra dữ liệu
```

Tài liệu được loại trùng theo `Content`, với `doc_id` tạo từ SHA-256 của nội dung. Mỗi câu hỏi gồm `query_id`, `input`, `expected_output`, `category`, `question_type`, `difficulty`, `context` và `source_doc_id`. Giữ ID gốc khi duy nhất, chuyển nhãn loại câu hỏi và độ khó sang tiếng Việt; bản raw giữ nguyên.

Chỉ dùng `input` làm truy vấn; `context`, `expected_output` và `source_doc_id` dùng để đánh giá. Toàn bộ `data/` được Git bỏ qua; manifest lưu revision và checksum của bản tải.

**Tiến độ:** đã tải, kiểm tra và tách documents/queries. Tiếp theo: chia đoạn, tạo qrels và chia tập phát triển/đánh giá theo nguồn tài liệu.
