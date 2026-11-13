from time import perf_counter

from src.llm.generation import generate
from src.query.rewriter import rewrite_query
from src.retrieval.reranker import rerank
from src.pipelines.naive import NaivePipeline


class AdvancedPipeline(NaivePipeline):
    def answer(self, question, retrieval_only=False):
        start = perf_counter()
        query_start = perf_counter()
        search_query = rewrite_query(self.client, self.config, question)
        query_seconds = perf_counter() - query_start
        retrieval_start = perf_counter()
        top_k = self.config["retrieval"]["top_k"]
        use_reranker = self.config.get("reranking", {}).get("enabled", False)
        candidate_k = max(top_k, self.config["retrieval"].get("candidate_k", 20)) if use_reranker else top_k
        candidates = self.retriever.search(search_query, candidate_k)
        retrieval_seconds = perf_counter() - retrieval_start
        rerank_start = perf_counter()
        chunks = rerank(question, candidates, self.config["reranking"], top_k) if use_reranker else candidates[:top_k]
        rerank_seconds = perf_counter() - rerank_start if use_reranker else 0.0
        result = {"input": question, "search_query": search_query,
                  "query_processing_seconds": query_seconds, "retrieval_seconds": retrieval_seconds,
                  "reranking_seconds": rerank_seconds, "retrieved_chunks": chunks,
                  "candidate_chunks": candidates if use_reranker else [],
                  "score_type": "cross_encoder" if use_reranker else self.config["retrieval"].get("method", "dense")}
        if not retrieval_only:
            generation_start = perf_counter()
            result.update(generate(self.client, self.config["generation"], question, chunks))
            result["generation_seconds"] = perf_counter() - generation_start
        result["total_seconds"] = perf_counter() - start
        return result
