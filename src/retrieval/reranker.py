"""CPU cross-encoder. Download is explicit, never triggered by a UI request."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.utils.common import ROOT, file_hash


@lru_cache(maxsize=1)
def load_reranker(model_path, max_length, checkpoint_hash=None):
    path = ROOT / model_path
    if not (path / "config.json").is_file():
        raise RuntimeError("Reranker chưa được tải. Chạy python experiments/prepare_reranker.py trước.")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError("Cài thư viện: python -m pip install -r requirements-advanced.txt") from error
    return CrossEncoder(str(path), device="cpu", max_length=max_length, local_files_only=True,
                        trust_remote_code=False)


def rerank(question, candidates, settings, top_k, model=None):
    if model is None:
        checkpoint = ROOT / settings["model_path"] / "config.json"
        fingerprint = file_hash(checkpoint) if checkpoint.is_file() else None
        model = load_reranker(settings["model_path"], settings.get("max_length", 1024), fingerprint)
    scores = np.asarray(model.predict([(question, c["text"]) for c in candidates],
                                     batch_size=settings.get("batch_size", 2), show_progress_bar=False)).reshape(-1)
    if len(scores) != len(candidates) or not np.isfinite(scores).all():
        raise ValueError("Invalid reranker scores")
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [dict(candidates[index], retrieval_rank=candidates[index]["rank"],
                 retrieval_score=candidates[index]["score"], score=float(scores[index]),
                 score_type="cross_encoder", rank=rank)
            for rank, index in enumerate(order, 1)]


def reranker_metadata(config):
    settings = config.get("reranking", {})
    if not settings.get("enabled"):
        return None
    import json
    source = ROOT / settings["model_path"] / "source.json"
    if not source.is_file():
        raise RuntimeError("Missing reranker provenance. Run python experiments/prepare_reranker.py")
    return json.loads(source.read_text(encoding="utf-8"))
