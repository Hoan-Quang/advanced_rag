"""Separate local ViAdminRAG documents and queries without using the network."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/viadminrag/ViAdminRag.json"
DEST = ROOT / "data/processed"
QUESTION_TYPES = {"Summarization": "Tóm tắt", "Multi-hop QA": "Hỏi đáp nhiều bước"}
DIFFICULTIES = {"Easy": "Dễ", "Medium": "Trung bình", "Hard": "Khó"}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty JSON array")
    required = {"Content", "Source URL", "input", "expected_output",
                "Category", "Question type", "Difficulty"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not (required | {"context"}) <= row.keys():
            raise ValueError(f"Missing fields at row {index}")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
            raise ValueError(f"Invalid text at row {index}")
        if row["Question type"] not in QUESTION_TYPES or row["Difficulty"] not in DIFFICULTIES:
            raise ValueError(f"Unknown question type or difficulty at row {index}")
        if not isinstance(row["context"], list) or not row["context"] or not all(
            isinstance(text, str) and text.strip() for text in row["context"]
        ):
            raise ValueError(f"Invalid reference context at row {index}")

    original_ids = Counter(
        row["ID"] for row in rows
        if isinstance(row.get("ID"), str) and row["ID"].strip()
    )
    content_urls = defaultdict(set)
    url_contents = defaultdict(set)
    for row in rows:
        content_urls[row["Content"]].add(row["Source URL"])
        url_contents[row["Source URL"]].add(row["Content"])

    content_ids = {content: "doc_" + digest(content) for content in content_urls}
    if len(set(content_ids.values())) != len(content_ids):
        raise ValueError("Document hash collision")
    documents = sorted([
        {"doc_id": content_ids[content], "content": content,
         "source_url": sorted(urls)[0]}
        for content, urls in content_urls.items()
    ], key=lambda doc: doc["doc_id"])

    queries = []
    for row in rows:
        original = row.get("ID")
        query_id = original if isinstance(original, str) and original_ids[original] == 1 else (
            "query_" + digest(json.dumps(row, ensure_ascii=False, sort_keys=True))
        )
        queries.append({
            "query_id": query_id,
            "input": row["input"],
            "expected_output": row["expected_output"],
            "category": row["Category"],
            "question_type": QUESTION_TYPES[row["Question type"]],
            "difficulty": DIFFICULTIES[row["Difficulty"]],
            "context": row["context"],
            "source_doc_id": content_ids[row["Content"]],
        })
    if len({query["query_id"] for query in queries}) != len(queries):
        raise ValueError("Duplicate query IDs remain; inspect duplicate source rows")
    by_id = {doc["doc_id"]: doc for doc in documents}
    for row, query in zip(rows, queries):
        if by_id[query["source_doc_id"]]["content"] != row["Content"]:
            raise ValueError(f"Invalid source mapping: {query['query_id']}")

    report = {
        "documents": len(documents), "queries": len(queries),
        "document_deduplication": "exact Content, no text normalization",
        "query_id_fallbacks": sum(q["query_id"] != r.get("ID") for q, r in zip(queries, rows)),
        "source_mapping_errors": 0,
        "urls_with_multiple_contents": {
            url: sorted(content_ids[text] for text in texts)
            for url, texts in sorted(url_contents.items()) if len(texts) > 1
        },
        "documents_with_multiple_urls": {
            content_ids[text]: sorted(urls)
            for text, urls in content_urls.items() if len(urls) > 1
        },
    }
    return documents, queries, report


def main():
    content = SOURCE.read_bytes()
    documents, queries, report = prepare(json.loads(content.decode("utf-8-sig")))
    report["source"] = SOURCE.relative_to(ROOT).as_posix()
    report["source_sha256"] = hashlib.sha256(content).hexdigest()
    DEST.mkdir(parents=True, exist_ok=True)
    write_jsonl(DEST / "documents.jsonl", documents)
    write_jsonl(DEST / "queries.jsonl", queries)
    (DEST / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
