from src.pipelines.naive import NaivePipeline
from src.pipelines.advanced import AdvancedPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def build_pipeline(config, client):
    retriever = DenseRetriever(config, client)
    if config["retrieval"].get("method", "dense") == "hybrid":
        retriever = HybridRetriever(retriever, config["retrieval"])
    advanced = (config["retrieval"].get("method") == "hybrid"
                or config.get("query_processing", {}).get("enabled")
                or config.get("reranking", {}).get("enabled")
                or config["chunking"].get("method") == "structure")
    return (AdvancedPipeline if advanced else NaivePipeline)(config, retriever, client)
