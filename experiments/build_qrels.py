"""Build auditable draft qrels from reference contexts, without model calls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, file_hash, load_config, read_jsonl, write_jsonl
from src.evaluation.qrels import create_qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/naive.yaml"))
    parser.add_argument("--output-dir", help="Default: data/processed/qrels/<config name>")
    parser.add_argument("--min-overlap", type=int, default=40)
    args = parser.parse_args()
    config = load_config(args.config)
    qrels, alignments, review, report = create_qrels(
        read_jsonl(config["documents"]), read_jsonl(config["queries"]),
        read_jsonl(config["chunks"]), args.min_overlap,
    )
    folder = Path(args.output_dir) if args.output_dir else ROOT / "data/processed/qrels" / config["name"]
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "qrels.jsonl", qrels)
    write_jsonl(folder / "alignments.jsonl", alignments)
    write_jsonl(folder / "needs_review.jsonl", review)
    report["source_hashes"] = {key: file_hash(config[key]) for key in ("documents", "queries", "chunks")}
    report["qrels_sha256"] = file_hash(folder / "qrels.jsonl")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("total_queries", "auto_aligned_queries", "needs_review_queries", "qrel_rows", "label_status")}, indent=2))
    print(f"Saved: {folder}")


if __name__ == "__main__":
    main()
