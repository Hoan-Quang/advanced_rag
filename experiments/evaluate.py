"""Evaluate saved retrieval results; incomplete labels require explicit opt-in."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, file_hash, read_jsonl
from src.evaluation.retrieval import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qrels-dir", default=str(ROOT / "data/processed/qrels/naive"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run_dir, qrels_dir = Path(args.run_dir), Path(args.qrels_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((qrels_dir / "report.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if file_hash(qrels_dir / "qrels.jsonl") != report["qrels_sha256"]:
        raise ValueError("Qrels changed without updating the report")
    for key in ("documents", "queries", "chunks"):
        if file_hash(config[key]) != report["source_hashes"][key]:
            raise ValueError(f"Qrels built for different {key}")
    if manifest["queries_sha256"] != report["source_hashes"]["queries"]:
        raise ValueError("Run used different queries")
    if manifest["index"]["chunks_sha256"] != report["source_hashes"]["chunks"]:
        raise ValueError("Run used different chunks")
    if manifest["index"]["documents_sha256"] != report["source_hashes"]["documents"]:
        raise ValueError("Run used different documents")
    if manifest["status"] != "complete" and not args.allow_partial:
        raise ValueError("Run is incomplete")
    summary = evaluate_retrieval(read_jsonl(config["queries"]), read_jsonl(qrels_dir / "qrels.jsonl"),
                                 read_jsonl(run_dir / "predictions.jsonl"), args.k,
                                 report["eligible_query_ids"], args.allow_partial)
    summary["label_status"] = report["label_status"]
    summary["qrels_sha256"] = report["qrels_sha256"]
    output = run_dir / f"retrieval_metrics_at_{args.k}.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("scored_queries", "total_queries", "partial", "label_status", "metrics")}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
