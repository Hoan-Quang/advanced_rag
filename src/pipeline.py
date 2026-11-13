from time import perf_counter

from src.generation import generate


class NaivePipeline:
    def __init__(self, config, retriever, client):
        self.config, self.retriever, self.client = config, retriever, client

    def answer(self, question, retrieval_only=False):
        # Only a question string crosses into the pipeline, never gold QA fields.
        start = perf_counter()
        chunks = self.retriever.search(question, self.config["retrieval"]["top_k"])
        retrieval_time = perf_counter() - start
        result = {"input": question, "retrieved_chunks": chunks,
                  "retrieval_seconds": retrieval_time}
        if not retrieval_only:
            generation_start = perf_counter()
            result.update(generate(self.client, self.config["generation"], question, chunks))
            result["generation_seconds"] = perf_counter() - generation_start
        result["total_seconds"] = perf_counter() - start
        return result
