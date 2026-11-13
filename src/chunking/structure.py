"""Prefer legal headings, paragraphs and sentence boundaries; retain source offsets."""

import re

from src.chunking.fixed import split_documents


def split_structure(documents, size, overlap):
    # Reuse input validation, but construct independent windows below.
    split_documents(documents, size, overlap)
    chunks = []
    for doc in documents:
        text = doc["content"]
        headings = [m.start() for m in re.finditer(r"(?<!\w)(?:Điều|Khoản|Chương|Mục)\s+\d+", text)]
        paragraphs = [m.end() for m in re.finditer(r"\n\s*\n", text)]
        sentences = [m.end() for m in re.finditer(r"[.!?;][\s]+", text)]
        start = 0
        while start < len(text):
            limit = min(start + size, len(text))
            end = limit
            if limit < len(text):
                minimum = start + max(overlap + 1, size * 3 // 5)
                for boundaries in (headings, paragraphs, sentences):
                    choices = [b for b in boundaries if minimum <= b <= limit]
                    if choices:
                        end = max(choices)
                        break
                else:
                    boundary = text.rfind(" ", minimum, limit)
                    if boundary >= minimum:
                        end = boundary + 1
            if text[start:end].strip():
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_structure_{size}_{overlap}_{start}_{end}",
                    "doc_id": doc["doc_id"], "text": text[start:end],
                    "start_char": start, "end_char": end, "source_url": doc["source_url"],
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks
