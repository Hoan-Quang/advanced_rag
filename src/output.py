"""Readable terminal presentation; original pipeline results remain unchanged."""

from textwrap import fill, shorten


def format_answer(result, show_chunks=False, width=100):
    chunks = result.get("retrieved_chunks", [])
    lines = ["CÂU HỎI", fill(result["input"], width=width), ""]
    if "answer" in result:
        answer = result["answer"]
        # Compact citations for display only; JSON retains the original IDs.
        for number, chunk in enumerate(chunks, 1):
            answer = answer.replace(f"[{chunk['chunk_id']}]", f"[{number}]")
        lines.append("TRẢ LỜI")
        for paragraph in answer.splitlines():
            lines.append(fill(paragraph, width=width, break_long_words=False,
                              break_on_hyphens=False) if paragraph.strip() else "")
        lines.append("")
    else:
        lines.extend(["Chế độ chỉ truy xuất, chưa sinh câu trả lời.", ""])

    lines.append(f"NGUỒN TRUY XUẤT ({len(chunks)} đoạn; điểm cosine similarity)")
    for number, chunk in enumerate(chunks, 1):
        lines.append(f"[{number}] Điểm: {chunk['score']:.4f} | "
                     f"Vị trí ký tự: {chunk['start_char']}–{chunk['end_char']}")
        lines.append(f"    Nguồn: {chunk['source_url']}")
        if show_chunks:
            lines.append(f"    ID: {chunk['chunk_id']}")
            text = chunk["text"]
        else:
            text = shorten(chunk["text"], width=220, placeholder=" …")
        lines.append(fill(text, width=width, initial_indent="    ", subsequent_indent="    ",
                          break_long_words=False, break_on_hyphens=False))
        lines.append("")

    timing = [f"Truy xuất: {result['retrieval_seconds']:.2f}s"]
    if "generation_seconds" in result:
        timing.append(f"Sinh câu trả lời: {result['generation_seconds']:.2f}s")
    timing.append(f"Tổng: {result['total_seconds']:.2f}s")
    lines.extend(["THỜI GIAN", " | ".join(timing)])
    tokens = []
    for key, label in [("prompt_eval_count", "Đầu vào"), ("eval_count", "Đầu ra")]:
        if result.get(key) is not None:
            tokens.append(f"{label}: {result[key]}")
    if tokens:
        lines.append("Token — " + " | ".join(tokens))
    if result.get("done_reason") == "length":
        lines.append("Lưu ý: model đã chạm giới hạn sinh; câu trả lời có thể chưa hoàn tất.")
    if chunks and not show_chunks:
        lines.append("Dùng --show-chunks để xem toàn bộ nội dung và ID các đoạn.")
    return "\n".join(lines)
