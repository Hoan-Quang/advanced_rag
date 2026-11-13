"""Lexical Generation diagnostics, not a substitute for semantic/human grading."""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import file_hash, read_jsonl


def tokens(text):
    text = re.sub(r"\[(?:doc_[^\]]+|\d+)\]", " ", text)
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def answer_scores(prediction, reference):
    pred, gold = tokens(prediction), tokens(reference)
    common = sum((Counter(pred) & Counter(gold)).values())
    f1 = 2 * common / (len(pred) + len(gold)) if pred or gold else 1.0
    return {"token_f1": f1, "exact_match": int(pred == gold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    folder = Path(args.run_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["config"]["queries"]
    if file_hash(source) != manifest["queries_sha256"]:
        raise ValueError("Queries changed since this run")
    queries = {q["query_id"]: q for q in read_jsonl(source)}
    rows = read_jsonl(folder / "predictions.jsonl")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    if not rows or any("answer" not in row for row in rows):
        raise ValueError("This run has no Generation answers")
    if set(row["query_id"] for row in rows) - queries.keys():
        raise ValueError("Unknown prediction IDs")
    partial = len(rows) != len(queries) or manifest["status"] != "complete"
    if partial and not args.allow_partial:
        raise ValueError("Incomplete answers; use --allow-partial for a diagnostic subset")
    details = [{"query_id": row["query_id"],
                **answer_scores(row["answer"], queries[row["query_id"]]["expected_output"])} for row in rows]
    summary = {"scored_queries": len(rows), "total_queries": len(queries), "partial": partial,
               "metrics": {key: sum(row[key] for row in details) / len(details) for key in ("token_f1", "exact_match")},
               "note": "Lexical overlap only; not semantic correctness or faithfulness.", "per_query": details}
    summary["groups"] = {}
    for field in ("question_type", "difficulty"):
        groups = {}
        for label in sorted({queries[row["query_id"]][field] for row in details}):
            selected = [row for row in details if queries[row["query_id"]][field] == label]
            groups[label] = {"count": len(selected), **{
                key: sum(row[key] for row in selected) / len(selected) for key in ("token_f1", "exact_match")}}
        summary["groups"][field] = groups
    (folder / "answer_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("scored_queries", "total_queries", "partial", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
