"""Macro-averaged binary chunk relevance metrics with explicit coverage."""

import math
from collections import defaultdict


def evaluate_retrieval(queries, qrels, predictions, k, eligible_ids, allow_partial=False):
    if k < 1:
        raise ValueError("k must be positive")
    query_map = {q["query_id"]: q for q in queries}
    predicted = {p["query_id"]: p for p in predictions}
    if len(predicted) != len(predictions) or len(query_map) != len(queries):
        raise ValueError("Duplicate query IDs")
    if set(predicted) - query_map.keys():
        raise ValueError("Predictions contain unknown query IDs")
    gold = defaultdict(set)
    for row in qrels:
        if row["query_id"] not in query_map:
            raise ValueError("Qrels contain unknown query IDs")
        if row["relevance"] > 0:
            gold[row["query_id"]].add(row["chunk_id"])
    eligible = set(eligible_ids)
    if eligible - query_map.keys() or eligible != set(gold):
        raise ValueError("Eligible query IDs disagree with qrels")
    scored = set(predicted) & eligible
    if not allow_partial and scored != set(query_map):
        raise ValueError("Not all queries have labels and predictions; use --allow-partial for a diagnostic subset")
    if not scored:
        raise ValueError("No queries have both labels and predictions")
    details = []
    for query_id in sorted(scored):
        chunks = predicted[query_id]["retrieved_chunks"]
        ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate retrieved chunks: {query_id}")
        if len(ids) < k:
            raise ValueError(f"Fewer than k predictions for {query_id}; rerun retrieval with top_k >= k")
        hits = [1 if chunk_id in gold[query_id] else 0 for chunk_id in ids[:k]]
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold[query_id]))))
        details.append({
            "query_id": query_id,
            "recall": sum(hits) / len(gold[query_id]),
            "mrr": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0),
            "ndcg": sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits)) / ideal,
            "hit": int(any(hits)),
        })

    def average(rows):
        return {f"{name}@{k}": sum(row[name] for row in rows) / len(rows)
                for name in ("recall", "mrr", "ndcg", "hit")}

    groups = {}
    for field in ("question_type", "difficulty"):
        labels = defaultdict(list)
        for row in details:
            labels[query_map[row["query_id"]][field]].append(row)
        groups[field] = {label: {"count": len(rows), **average(rows)} for label, rows in labels.items()}
    return {
        "total_queries": len(queries), "scored_queries": len(details),
        "partial": scored != set(query_map),
        "missing_label_query_ids": sorted(set(query_map) - eligible),
        "missing_prediction_query_ids": sorted(set(query_map) - predicted.keys()),
        "metrics": average(details), "groups": groups, "per_query": details,
    }
