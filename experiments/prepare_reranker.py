"""Explicitly download and save the multilingual CPU reranker."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import ROOT


def main():
    from sentence_transformers import CrossEncoder
    destination = ROOT / ".models/bge-reranker-v2-m3"
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=1024,
                         cache_folder=str(ROOT / ".models/cache"), trust_remote_code=False)
    model.save(str(destination))
    metadata = {"model": "BAAI/bge-reranker-v2-m3", "device": "cpu", "max_length": 1024,
                "revision": getattr(model.model.config, "_commit_hash", None)}
    (destination / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
