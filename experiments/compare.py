"""Compare saved metrics only when runs use the same evaluated question IDs."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="results/comparison.csv")
    args = parser.parse_args()
    rows, reference_ids, reference_queries = [], None, None
    for name in args.runs:
        folder = Path(name)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        scores = json.loads((folder / f"retrieval_metrics_at_{args.k}.json").read_text(encoding="utf-8"))
        ids = {row["query_id"] for row in scores["per_query"]}
        query_hash = manifest["queries_sha256"]
        if reference_ids is not None and (ids != reference_ids or query_hash != reference_queries):
            raise ValueError("Runs score different question sets or different query data; cannot compare directly")
        reference_ids, reference_queries = ids, query_hash
        predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        selected = [row for row in predictions if row["query_id"] in ids]
        row = {"pipeline": manifest["config"]["name"], "run": str(folder),
               "scored_queries": len(ids), "partial": scores["partial"],
               "label_status": scores["label_status"], **scores["metrics"],
               "mean_total_seconds": sum(row["total_seconds"] for row in selected) / len(selected)}
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} configurations: {output}")


if __name__ == "__main__":
    main()
