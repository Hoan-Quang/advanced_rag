"""Application boundary for interactive requests; no model calls at import time."""

from copy import deepcopy
from threading import Lock

from src.llm.ollama_client import OllamaClient
from src.pipelines.factory import build_pipeline

_request_lock = Lock()


class BusyError(RuntimeError):
    pass


def check_connection(config, retrieval_only=False):
    client = OllamaClient(config["ollama"]["base_url"], timeout=5)
    models = [client.model_info(config["embedding"]["model"])]
    if not retrieval_only or config.get("query_processing", {}).get("enabled", False):
        models.append(client.model_info(config["generation"]["model"]))
    return models


def run_question(config, question, top_k, retrieval_only=False):
    question = question.strip()
    if not question:
        raise ValueError("Bạn chưa nhập câu hỏi.")
    if not _request_lock.acquire(blocking=False):
        raise BusyError("Hệ thống đang xử lý một câu hỏi khác. Vui lòng thử lại sau.")
    try:
        settings = deepcopy(config)
        settings["retrieval"]["top_k"] = top_k
        client = OllamaClient(**settings["ollama"])
        if not retrieval_only or settings.get("query_processing", {}).get("enabled", False):
            client.model_info(settings["generation"]["model"])
        return build_pipeline(settings, client).answer(question, retrieval_only)
    finally:
        _request_lock.release()
