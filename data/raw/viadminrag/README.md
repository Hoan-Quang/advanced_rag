---
language:
- vi
task_categories:
- question-answering
- text-generation
tags:
- legal
- administrative
- vietnam
- rag
- qa
dataset_info:
  features:
    - name: ID
      dtype: string
    - name: Content
      dtype: string
    - name: Source URL
      dtype: string
    - name: Category
      dtype: string
    - name: Question type
      dtype: string
    - name: Difficulty
      dtype: string
    - name: input
      dtype: string
    - name: expected_output
      dtype: string
    - name: context
      sequence: string
---

# ViAdminRAG Dataset

## Giới thiệu (Dataset Description)
**ViAdminRAG** là bộ dữ liệu (dataset) tiếng Việt được xây dựng chuyên dụng cho bài toán **Question Answering (QA)** và mô hình **Retrieval-Augmented Generation (RAG)** trong lĩnh vực hành chính, pháp lý và giáo dục tại Việt Nam.

Bộ dữ liệu cung cấp các câu hỏi (`input`), câu trả lời mong đợi (`expected_output`) cùng với các đoạn văn bản ngữ cảnh (`context`) được trích xuất từ các văn bản quy phạm pháp luật, công văn, thông tư,...

## Cấu trúc dữ liệu (Dataset Structure)
Mỗi mẫu dữ liệu (sample) trong bộ dataset bao gồm các trường thông tin sau:
- `ID`: Mã định danh duy nhất cho mỗi cặp câu hỏi - câu trả lời.
- `Content`: Nội dung toàn văn của văn bản pháp luật/tài liệu gốc.
- `Source URL`: Đường dẫn nguồn tài liệu tham khảo (ví dụ: thuvienphapluat.vn).
- `Category`: Lĩnh vực của tài liệu (VD: Giáo dục, Hành chính,...).
- `Question type`: Phân loại câu hỏi (VD: Summarization, Multi-hop QA, Factoid,...).
- `Difficulty`: Độ khó của câu hỏi (Easy, Medium, Hard).
- `input`: Câu hỏi đầu vào của người dùng.
- `expected_output`: Câu trả lời tiêu chuẩn dựa trên ngữ cảnh.
- `context`: Danh sách các đoạn trích dẫn (chunks) chứa thông tin để trả lời câu hỏi.

## Hướng dẫn sử dụng (How to Use)
Bạn có thể dễ dàng tải và sử dụng bộ dữ liệu này thông qua thư viện `datasets` của Hugging Face:

```python
from datasets import load_dataset

# Tải dataset
dataset = load_dataset("nhokjack/ViAdminRAG")

# Hiển thị mẫu dữ liệu đầu tiên
print(dataset['train'][0])
```

## Mục đích sử dụng (Use Cases)
- **Huấn luyện mô hình (Training):** Tinh chỉnh (fine-tuning) các mô hình ngôn ngữ lớn (LLM) để hiểu và trả lời các câu hỏi bằng tiếng Việt trong lĩnh vực pháp lý/hành chính.
- **Đánh giá hệ thống RAG (Evaluation):** Dùng làm benchmark để đánh giá hiệu suất của các hệ thống tìm kiếm và sinh văn bản (Retriever & Generator).

## Tác giả (Author)
Bộ dữ liệu được thu thập và tổng hợp bởi **nhokjack04** phục vụ cho mục đích nghiên cứu (KLTN).
