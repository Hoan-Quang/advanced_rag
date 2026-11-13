"""CLI for the Naive RAG baseline. Run from the project root."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.fixed import build_chunks
from src.common import ROOT, file_hash, load_config, read_jsonl
from src.generation import SYSTEM_PROMPT
from src.ollama_client import OllamaClient
from src.pipeline import NaivePipeline
from src.retrieval.dense import DenseRetriever, build_index


def run_queries(config, pipeline, client, args):
    rows = read_jsonl(config["queries"])
    if not rows or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Empty queries or duplicate query IDs")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    generation_model = None if args.retrieval_only else client.model_info(config["generation"]["model"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = ROOT / "results" / config["name"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": config, "system_prompt": SYSTEM_PROMPT,
        "index": pipeline.retriever.manifest, "generation_model": generation_model,
        "queries_sha256": file_hash(config["queries"]),
        "python": platform.python_version(), "platform": platform.platform(),
        "query_count": len(rows), "retrieval_only": args.retrieval_only,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "completed": 0,
    }
    manifest_path = output / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    print(f"Results: {output}", flush=True)
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                result = pipeline.answer(row["input"], args.retrieval_only)
                result.update({"query_id": row["query_id"],
                               "category": row["category"],
                               "question_type": row["question_type"],
                               "difficulty": row["difficulty"]})
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                manifest["completed"] += 1
                print(f"Completed {manifest['completed']}/{len(rows)}", flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


def main():
    parser = argparse.ArgumentParser(description="Naive RAG: fixed chunks, cosine Dense Retrieval, Ollama")
    parser.add_argument("command", choices=["chunk", "check", "index", "ask", "run"])
    parser.add_argument("--config", default=str(ROOT / "configs/naive.yaml"))
    parser.add_argument("--question", help="Question text for ask")
    parser.add_argument("--limit", type=int, help="Run only the first N questions for a smoke check")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Generation for ask/run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "chunk":
        chunks = build_chunks(config)
        print(f"Created {len(chunks)} chunks: {config['chunks']}")
        return
    client = OllamaClient(**config["ollama"])
    if args.command == "check":
        models = [client.model_info(config["embedding"]["model"])]
        if not args.retrieval_only:
            models.append(client.model_info(config["generation"]["model"]))
        print(json.dumps(models, indent=2))
        return
    if args.command == "index":
        print(json.dumps(build_index(config, client), indent=2))
        return
    if args.command == "ask" and not args.question:
        parser.error("ask requires --question")
    retriever = DenseRetriever(config, client)
    pipeline = NaivePipeline(config, retriever, client)
    if args.command == "ask":
        if not args.retrieval_only:
            client.model_info(config["generation"]["model"])
        # Escaped Unicode also works on legacy Windows console encodings.
        print(json.dumps(pipeline.answer(args.question, args.retrieval_only), ensure_ascii=True, indent=2))
    else:
        run_queries(config, pipeline, client, args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
