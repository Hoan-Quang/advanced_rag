"""One rewrite before a single retrieval pass; original question drives Generation."""

def rewrite_query(client, config, question):
    settings = config.get("query_processing", {})
    if not settings.get("enabled", False):
        return question
    response = client.chat(config["generation"]["model"], [
        {"role": "system", "content": (
            "Viết lại câu hỏi thành một truy vấn tìm kiếm văn bản giáo dục tiếng Việt rõ ràng. "
            "Giữ nguyên ý định, tên riêng, con số, điều kiện và phủ định. "
            "Không trả lời câu hỏi, không tự thêm căn cứ pháp luật. Chỉ xuất một truy vấn trên một dòng."
        )}, {"role": "user", "content": question},
    ], {"temperature": 0, "seed": 42, "num_ctx": config["generation"]["num_ctx"], "num_predict": 192})
    rewritten = response["message"]["content"].strip()
    if not rewritten or len(rewritten) > 4000 or response.get("done_reason") == "length":
        raise ValueError("Query rewrite is empty, too long or incomplete")
    return " ".join(rewritten.split())
