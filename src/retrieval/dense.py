"""Exact cosine search over normalized vectors; suitable for this small corpus."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common import file_hash, read_jsonl, write_jsonl


def normalize(vectors, count):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("Invalid embedding dimensions or non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero-length embedding")
    return array / norms


def build_index(config, client):
    model = client.model_info(config["embedding"]["model"])
    chunks = read_jsonl(config["chunks"])
    if not chunks:
        raise ValueError("Empty chunks file; run chunk first")
    batches = []
    batch_size = config["embedding"]["batch_size"]
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batches.append(normalize(client.embed(model["name"], [c["text"] for c in batch]), len(batch)))
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)}", flush=True)
    vectors = np.concatenate(batches)
    folder = Path(config["index_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must not leave a usable manifest for mixed old/new files.
    (folder / "manifest.json").unlink(missing_ok=True)
    write_jsonl(folder / "chunks.jsonl", chunks)
    np.save(folder / "vectors.npy", vectors, allow_pickle=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding": model, "chunking": config["chunking"],
        "documents_sha256": file_hash(config["documents"]),
        "chunks_sha256": file_hash(folder / "chunks.jsonl"),
        "vectors_sha256": file_hash(folder / "vectors.npy"),
        "count": len(chunks), "dimensions": vectors.shape[1], "metric": "cosine",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class DenseRetriever:
    def __init__(self, config, client):
        self.client = client
        folder = Path(config["index_dir"])
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["embedding"] != client.model_info(config["embedding"]["model"]):
            raise ValueError("Embedding model changed; rebuild index")
        if self.manifest["chunking"] != config["chunking"]:
            raise ValueError("Chunking config changed; run chunk and index again")
        for key, path in [("documents", config["documents"]), ("chunks", folder / "chunks.jsonl"),
                          ("vectors", folder / "vectors.npy")]:
            if self.manifest[key + "_sha256"] != file_hash(path):
                raise ValueError(f"{key} changed; rebuild index")
        self.chunks = read_jsonl(folder / "chunks.jsonl")
        self.vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        expected = (self.manifest["count"], self.manifest["dimensions"])
        if self.vectors.shape != expected or len(self.chunks) != expected[0]:
            raise ValueError("Index dimensions do not match manifest")

    def search(self, question, top_k):
        if not isinstance(question, str) or not question.strip() or top_k < 1:
            raise ValueError("Non-empty question and positive top_k required")
        query = normalize(self.client.embed(self.manifest["embedding"]["name"], [question]), 1)
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimensions differ from index")
        scores = self.vectors @ query[0]
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [dict(self.chunks[i], rank=rank, score=float(scores[i]))
                for rank, i in enumerate(indices, 1)]
