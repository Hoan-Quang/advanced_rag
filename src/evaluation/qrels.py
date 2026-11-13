"""Conservative evidence alignment. Fuzzy candidates never become gold labels."""

import re
from collections import defaultdict
from difflib import SequenceMatcher


def normalized_with_offsets(text):
    # Normalize only whitespace, preserving a map back to original character spans.
    output, offsets = [], []
    for match in re.finditer(r"\S+", text):
        if output:
            output.append(" ")
            offsets.append((offsets[-1][1], match.start()))
        for index in range(match.start(), match.end()):
            output.append(text[index])
            offsets.append((index, index + 1))
    return "".join(output), offsets


def create_qrels(documents, queries, chunks, min_overlap=40):
    if min_overlap < 1:
        raise ValueError("min_overlap must be positive")
    docs = {doc["doc_id"]: doc for doc in documents}
    if len(docs) != len(documents):
        raise ValueError("Duplicate document IDs")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    by_doc = defaultdict(list)
    for chunk in chunks:
        text = docs[chunk["doc_id"]]["content"]
        start, end = chunk["start_char"], chunk["end_char"]
        if not 0 <= start < end <= len(text) or chunk["text"] != text[start:end]:
            raise ValueError(f"Chunk offsets/text mismatch: {chunk['chunk_id']}")
        by_doc[chunk["doc_id"]].append(chunk)
    normalized_docs = {key: normalized_with_offsets(doc["content"]) for key, doc in docs.items()}
    qrels, alignments, review, eligible = [], [], [], []
    for query in queries:
        doc_id = query["source_doc_id"]
        text, offsets = normalized_docs[doc_id]
        supporting, issues, evidence_spans = set(), [], []
        if not query["context"]:
            issues.append({"reason": "no_reference_context"})
        for context_index, evidence in enumerate(query["context"]):
            needle, _ = normalized_with_offsets(evidence)
            starts = []
            position = text.find(needle) if needle else -1
            while position >= 0:
                starts.append(position)
                position = text.find(needle, position + 1)
            detail = {"context_index": context_index, "context": evidence}
            if len(starts) != 1:
                # Similarity is a review aid, not a relevance judgment.
                candidates = sorted([
                    {"chunk_id": chunk["chunk_id"], "text": chunk["text"],
                     "similarity": SequenceMatcher(None, needle,
                         normalized_with_offsets(chunk["text"])[0], autojunk=False).ratio()}
                    for chunk in by_doc[doc_id]
                ], key=lambda row: -row["similarity"])[:3]
                issues.append(dict(detail, reason="not_found" if not starts else "multiple_occurrences",
                                   candidates=candidates))
                continue
            start = offsets[starts[0]][0]
            end = offsets[starts[0] + len(needle) - 1][1]
            hits = []
            covered = set()
            for chunk in by_doc[doc_id]:
                left, right = max(start, chunk["start_char"]), min(end, chunk["end_char"])
                if right - left >= min(min_overlap, end - start):
                    hits.append(chunk["chunk_id"])
                    covered.update(range(left, right))
            required = {i for i in range(start, end) if not docs[doc_id]["content"][i].isspace()}
            if not hits or not required <= covered:
                issues.append(dict(detail, reason="incomplete_chunk_coverage", start_char=start, end_char=end))
                continue
            supporting.update(hits)
            evidence_spans.append({"context_index": context_index, "start_char": start,
                                   "end_char": end, "chunk_ids": hits})
        if issues:
            review.append({"query_id": query["query_id"], "input": query["input"],
                           "source_doc_id": doc_id, "issues": issues})
        else:
            eligible.append(query["query_id"])
            qrels.extend({"query_id": query["query_id"], "chunk_id": chunk_id, "relevance": 1}
                         for chunk_id in sorted(supporting))
        alignments.append({"query_id": query["query_id"], "source_doc_id": doc_id,
                           "status": "needs_review" if issues else "auto_aligned",
                           "evidence_spans": evidence_spans})
    report = {
        "total_queries": len(queries), "auto_aligned_queries": len(eligible),
        "needs_review_queries": len(review), "qrel_rows": len(qrels),
        "eligible_query_ids": eligible, "complete": not review,
        "label_status": "automatic_draft_not_manually_adjudicated",
        "method": "unique whitespace-normalized evidence span in source document",
        "relevance_rule": "binary: chunk overlaps evidence by at least min(min_overlap_chars, evidence_length)",
        "min_overlap_chars": min_overlap,
        "note": "Queries with any unresolved evidence are excluded from draft qrels, not labeled irrelevant.",
    }
    return qrels, alignments, review, report
