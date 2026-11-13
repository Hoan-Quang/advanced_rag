"""Evidence/offset edge cases and hand-calculated retrieval metrics."""

import math
import unittest

from src.chunking.fixed import split_documents
from src.evaluation.qrels import create_qrels
from src.evaluation.retrieval import evaluate_retrieval


def query(query_id, context):
    return {"query_id": query_id, "input": "Question", "source_doc_id": "d1",
            "context": [context], "question_type": "type", "difficulty": "level"}


class EvaluationTests(unittest.TestCase):
    def test_normalized_evidence_crosses_chunk_boundaries_with_source_offsets(self):
        text = "Intro. Alpha\n  beta gamma delta. Ending."
        docs = [{"doc_id": "d1", "content": text, "source_url": "url"}]
        chunks = split_documents(docs, size=18, overlap=5)
        qrels, alignment, review, report = create_qrels(docs, [query("q1", "Alpha beta gamma delta.")], chunks, 2)
        self.assertEqual(review, [])
        self.assertTrue(report["complete"])
        span = alignment[0]["evidence_spans"][0]
        self.assertEqual(text[span["start_char"]:span["end_char"]], "Alpha\n  beta gamma delta.")
        self.assertGreater(len(qrels), 1)

    def test_unmatched_repeated_or_partial_context_is_not_assigned_gold(self):
        docs = [{"doc_id": "d1", "content": "Intro. repeat. repeat. Unique fact.", "source_url": "url"}]
        chunks = split_documents(docs, 100, 0)
        queries = [query("q1", "missing fact"), query("q2", "repeat."), query("q3", "Unique fact.")]
        queries[2]["context"].append("not present")
        qrels, _, review, report = create_qrels(docs, queries, chunks, 2)
        self.assertEqual(qrels, [])
        self.assertEqual(len(review), 3)
        self.assertEqual(report["eligible_query_ids"], [])
        self.assertEqual(review[0]["issues"][0]["reason"], "not_found")
        self.assertEqual(review[1]["issues"][0]["reason"], "multiple_occurrences")
        self.assertTrue(review[0]["issues"][0]["candidates"])

    def test_stale_chunk_text_is_rejected(self):
        docs = [{"doc_id": "d1", "content": "Unique fact.", "source_url": "url"}]
        chunks = split_documents(docs, 100, 0)
        chunks[0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            create_qrels(docs, [query("q1", "Unique fact.")], chunks)

    def test_hand_calculated_metrics_and_partial_guard(self):
        queries = [query("q1", "gold"), query("q2", "gold")]
        qrels = [{"query_id": "q1", "chunk_id": key, "relevance": 1} for key in ("c1", "c2")]
        predictions = [{"query_id": "q1", "retrieved_chunks": [{"chunk_id": "wrong"}, {"chunk_id": "c1"}]}]
        with self.assertRaisesRegex(ValueError, "Not all"):
            evaluate_retrieval(queries, qrels, predictions, 2, ["q1"])
        result = evaluate_retrieval(queries, qrels, predictions, 2, ["q1"], allow_partial=True)
        self.assertTrue(result["partial"])
        self.assertEqual(result["scored_queries"], 1)
        self.assertEqual(result["metrics"]["recall@2"], 0.5)
        self.assertEqual(result["metrics"]["mrr@2"], 0.5)
        self.assertAlmostEqual(result["metrics"]["ndcg@2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        with self.assertRaisesRegex(ValueError, "Fewer than"):
            evaluate_retrieval(queries, qrels, predictions, 5, ["q1"], allow_partial=True)


if __name__ == "__main__":
    unittest.main()
