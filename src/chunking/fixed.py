"""Baseline fixed-width Unicode character windows with exact source offsets."""

from src.common import read_jsonl, write_jsonl


def split_documents(documents, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    chunks, seen = [], set()
    for document in documents:
        doc_id, text = document["doc_id"], document["content"]
        if doc_id in seen or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Duplicate document or empty content: {doc_id}")
        seen.add(doc_id)
        for start in range(0, len(text), size - overlap):
            end = min(start + size, len(text))
            if text[start:end].strip():
                chunks.append({
                    "chunk_id": f"{doc_id}_fixed_{size}_{overlap}_{start}_{end}",
                    "doc_id": doc_id, "text": text[start:end],
                    "start_char": start, "end_char": end,
                    "source_url": document["source_url"],
                })
            if end == len(text):
                break
    if not chunks:
        raise ValueError("No chunks generated")
    return chunks


def build_chunks(config):
    splitter = split_documents
    if config["chunking"].get("method") == "structure":
        from src.chunking.structure import split_structure
        splitter = split_structure
    chunks = splitter(read_jsonl(config["documents"]), **{
        "size": config["chunking"]["size"], "overlap": config["chunking"]["overlap"]
    })
    write_jsonl(config["chunks"], chunks)
    return chunks
