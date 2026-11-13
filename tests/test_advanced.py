"""Advanced components and combined pipeline, without downloading/running models."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app
from src.chunking.structure import split_structure
from src.common import ROOT, load_config, read_jsonl
from src.factory import build_pipeline
from src.pipeline import AdvancedPipeline, NaivePipeline
from src.reranking import rerank
from src.retrieval.hybrid import BM25, fuse
from experiments.evaluate_answers import answer_scores


class FakeDense:
    def __init__(self, config=None, client=None):
        self.manifest = {"embedding": {"name": "fake"}}
        self.chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "học phí giáo dục", "start_char": 0, "end_char": 16, "source_url": "url"},
            {"chunk_id": "c2", "doc_id": "d2", "text": "tuyển sinh đại học", "start_char": 0, "end_char": 18, "source_url": "url"},
            {"chunk_id": "c3", "doc_id": "d3", "text": "miễn học phí", "start_char": 0, "end_char": 12, "source_url": "url"},
        ]
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return [dict(c, rank=i, score=1/i) for i, c in enumerate(self.chunks[:top_k], 1)]


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, model, messages, options):
        self.calls.append(messages)
        text = "miễn học phí" if len(self.calls) == 1 else "Câu trả lời [c3]"
        return {"done": True, "message": {"content": text}, "done_reason": "stop"}


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return [4 if "miễn" in text else 1 for _, text in pairs]


class AdvancedTests(unittest.TestCase):
    def test_answer_metric_citations_and_overlap(self):
        self.assertEqual(answer_scores("Học phí [1].", "Học phí"), {"token_f1": 1.0, "exact_match": 1})
        self.assertEqual(answer_scores("a a b", "a b b")["token_f1"], 2/3)
        self.assertEqual(answer_scores("", "known")["token_f1"], 0)

    def test_structure_source_offsets_and_full_coverage_on_real_documents(self):
        documents = read_jsonl(ROOT / "data/processed/documents.jsonl")
        chunks = split_structure(documents, 1200, 200)
        for doc in documents:
            selected = [c for c in chunks if c["doc_id"] == doc["doc_id"]]
            covered = set()
            for chunk in selected:
                start, end = chunk["start_char"], chunk["end_char"]
                self.assertEqual(chunk["text"], doc["content"][start:end])
                self.assertLessEqual(end - start, 1200)
                covered.update(range(start, end))
            self.assertTrue(all(index in covered for index, char in enumerate(doc["content"]) if not char.isspace()))
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))

    def test_bm25_unicode_and_rrf_deduplication(self):
        chunks = FakeDense().chunks
        hits = BM25(chunks).search("MIỄN HỌC PHÍ", 3)
        self.assertEqual(hits[0]["chunk_id"], "c3")
        self.assertEqual(BM25(chunks).search("unknownword", 3), [])
        dense = FakeDense().search("question", 3)
        merged = fuse(dense, hits, 3)
        self.assertEqual(len({c["chunk_id"] for c in merged}), len(merged))
        self.assertTrue(all(c["score_type"] == "rrf" for c in merged))
        match = next(c for c in merged if c["chunk_id"] == "c3")
        self.assertAlmostEqual(match["score"], 1/63 + 1/61)

    def test_reranker_reorders_and_preserves_original_rank(self):
        candidates = FakeDense().search("question", 3)
        result = rerank("original", candidates, {}, 1, model=FakeCrossEncoder())
        self.assertEqual(result[0]["chunk_id"], "c3")
        self.assertEqual(result[0]["retrieval_rank"], 3)
        self.assertEqual(candidates[2]["rank"], 3)
        self.assertEqual(result[0]["score_type"], "cross_encoder")

    def test_combined_pipeline_single_retrieval_original_question_generation(self):
        config = load_config(ROOT / "configs/advanced_all.yaml")
        config["retrieval"]["top_k"] = 2
        client, model = FakeClient(), FakeCrossEncoder()
        with patch("src.factory.DenseRetriever", FakeDense), patch("src.reranking.load_reranker", return_value=model):
            pipeline = build_pipeline(config, client)
            self.assertIsInstance(pipeline, AdvancedPipeline)
            result = pipeline.answer("Câu hỏi gốc")
            self.assertEqual(result["search_query"], "miễn học phí")
            self.assertEqual(len(pipeline.retriever.dense.calls), 1)
            self.assertEqual(result["retrieved_chunks"][0]["chunk_id"], "c3")
            self.assertTrue(all(pair[0] == "Câu hỏi gốc" for pair in model.pairs))
            self.assertIn("Câu hỏi gốc", client.calls[-1][-1]["content"])
            self.assertNotIn("expected_output", str(client.calls))
            self.assertEqual(len(result["retrieved_chunks"]), 2)

    def test_each_ablation_config_builds_independently(self):
        names = ["naive", "advanced_chunking", "advanced_query", "advanced_hybrid", "advanced_rerank", "advanced_all"]
        with patch("src.factory.DenseRetriever", FakeDense), patch("src.reranking.load_reranker", return_value=FakeCrossEncoder()):
            for name in names:
                config = load_config(ROOT / f"configs/{name}.yaml")
                client = FakeClient()
                pipeline = build_pipeline(config, client)
                self.assertEqual(type(pipeline), NaivePipeline if name == "naive" else AdvancedPipeline)
                self.assertEqual(config["generation"]["model"], "qwen2.5:3b")
                if name not in {"advanced_chunking", "advanced_all"}:
                    self.assertEqual(config["index_dir"], str(ROOT / "data/indexes/naive"))
                result = pipeline.answer("Câu hỏi gốc", retrieval_only=True)
                self.assertNotIn("answer", result)
                self.assertTrue(result["retrieved_chunks"])
                self.assertEqual(len(client.calls), 1 if name in {"advanced_query", "advanced_all"} else 0)

    def test_api_pipeline_selection_rejects_arbitrary_paths(self):
        client = TestClient(app)
        response = client.get("/api/config?pipeline=advanced_all")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["query_processing"])
        self.assertEqual(client.post("/api/ask", json={"question": "x", "pipeline": "../../other"}).status_code, 422)
        with patch("app.run_question", return_value={"answer": "test"}) as run:
            response = client.post("/api/ask", json={"question": "x", "pipeline": "advanced_hybrid"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(run.call_args.args[0]["retrieval"]["method"], "hybrid")


if __name__ == "__main__":
    unittest.main()
