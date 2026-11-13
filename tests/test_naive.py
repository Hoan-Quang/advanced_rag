"""Offline checks: source coverage, index integrity and exclusion of gold evidence."""

import copy
import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from src.chunking.fixed import split_documents
from src.common import write_jsonl
from src.generation import messages_for
from src.ollama_client import OllamaClient
from src.pipeline import NaivePipeline
from src.retrieval.dense import DenseRetriever, build_index, normalize
from experiments.run import run_queries


@contextmanager
def scratch_directory():
    scratch = Path(__file__).resolve().parents[1] / ".test_tmp"
    directory = scratch / uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        if not directory.resolve().is_relative_to(scratch.resolve()):
            raise ValueError("Test cleanup escaped scratch directory")
        shutil.rmtree(directory)


class FakeOllama:
    def __init__(self):
        self.prompts = []

    def model_info(self, name):
        return {"name": name, "digest": "test-digest"}

    def embed(self, model, texts):
        return [[1, 0] if "hoc phi" in text else [0, 1] for text in texts]

    def chat(self, model, messages, options):
        self.prompts.append(messages)
        return {"done": True, "message": {"content": "Test answer"}, "eval_count": 2}


class BaselineTests(unittest.TestCase):
    def test_offsets_coverage_and_no_redundant_tail(self):
        text = "  Quy định học phí.\n" * 19
        doc = {"doc_id": "d1", "content": text, "source_url": "url"}
        chunks = split_documents([doc], 70, 15)
        covered = set()
        for chunk in chunks:
            start, end = chunk["start_char"], chunk["end_char"]
            self.assertEqual(chunk["text"], text[start:end])
            self.assertLessEqual(end - start, 70)
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(len(text))))
        self.assertEqual(chunks[-1]["end_char"], len(text))
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        exact = dict(doc, content="x" * 70)
        self.assertEqual(len(split_documents([exact], 70, 15)), 1)
        with self.assertRaises(ValueError):
            split_documents([doc], 10, 10)

    def test_invalid_vectors_and_prompt_budget(self):
        for vectors in ([[0, 0]], [[float("nan"), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                normalize(vectors, 1)
        with self.assertRaises(ValueError):
            messages_for("question", [{"chunk_id": "c1", "text": "long context"}], 3)

    def test_persisted_index_ranking_generation_and_stale_detection(self):
        with scratch_directory() as directory:
            root = Path(directory)
            docs = [
                {"doc_id": "d1", "content": "hoc phi", "source_url": "u1"},
                {"doc_id": "d2", "content": "tot nghiep", "source_url": "u2"},
            ]
            config = {
                "documents": str(root / "docs.jsonl"), "chunks": str(root / "chunks.jsonl"),
                "index_dir": str(root / "index"), "chunking": {"size": 100, "overlap": 10},
                "embedding": {"model": "embed-test", "batch_size": 1}, "retrieval": {"top_k": 1},
                "generation": {"model": "chat-test", "max_context_chars": 1000,
                               "temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 50},
            }
            write_jsonl(config["documents"], docs)
            write_jsonl(config["chunks"], split_documents(docs, 100, 10))
            client = FakeOllama()
            build_index(config, client)
            retriever = DenseRetriever(config, client)
            hits = retriever.search("hoc phi", 5)
            self.assertEqual(hits[0]["doc_id"], "d1")
            self.assertAlmostEqual(hits[0]["score"], 1)
            result = NaivePipeline(config, retriever, client).answer("hoc phi")
            self.assertEqual(result["answer"], "Test answer")
            # The non-retrieved document is not inserted into Generation.
            self.assertNotIn("tot nghiep", json.dumps(client.prompts))
            before = len(client.prompts)
            NaivePipeline(config, retriever, client).answer("hoc phi", retrieval_only=True)
            self.assertEqual(len(client.prompts), before)
            config["name"] = "naive"
            config["queries"] = str(root / "queries.jsonl")
            gold = "GOLD_SHOULD_NOT_REACH_MODEL"
            write_jsonl(config["queries"], [{
                "query_id": "q1", "input": "hoc phi", "context": [gold],
                "expected_output": gold, "source_doc_id": gold,
                "category": "test", "question_type": "test", "difficulty": "test",
            }])
            pipeline = NaivePipeline(config, retriever, client)
            with patch("experiments.run.ROOT", root):
                run_queries(config, pipeline, client, SimpleNamespace(limit=None, retrieval_only=False))
            manifest_path = next((root / "results").rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed"], 1)
            self.assertNotIn(gold, json.dumps(client.prompts))
            predictions = (manifest_path.parent / "predictions.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(gold, predictions)
            changed = copy.deepcopy(config)
            changed["chunking"]["size"] = 200
            with self.assertRaisesRegex(ValueError, "Chunking"):
                DenseRetriever(changed, client)
            Path(config["documents"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "documents changed"):
                DenseRetriever(config, client)

    def test_http_payloads_do_not_truncate_embeddings_or_stream_chat(self):
        client = OllamaClient()
        with patch.object(client, "request", return_value={"embeddings": [[1, 0]]}) as request:
            client.embed("embedding", ["text"])
            self.assertFalse(request.call_args.args[1]["truncate"])
        with patch.object(client, "request", return_value={"done": True, "message": {"content": "answer"}}) as request:
            client.chat("chat", [{"role": "user", "content": "question"}], {"seed": 42})
            self.assertFalse(request.call_args.args[1]["stream"])


if __name__ == "__main__":
    unittest.main()
