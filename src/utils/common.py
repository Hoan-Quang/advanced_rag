import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    def merge(base, override):
        result = dict(base)
        for key, value in override.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result

    def read_config(source, seen):
        source = Path(source).resolve()
        if source in seen:
            raise ValueError("Circular configuration inheritance")
        with source.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        parent = values.pop("extends", None)
        return merge(read_config(source.parent / parent, seen | {source}), values) if parent else values

    config = read_config(path, set())
    for key in ("documents", "queries", "chunks", "index_dir"):
        config[key] = str(ROOT / config[key])
    size, overlap = config["chunking"]["size"], config["chunking"]["overlap"]
    if not isinstance(size, int) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("Require 0 <= chunk overlap < size")
    if config["retrieval"]["top_k"] < 1 or config["embedding"]["batch_size"] < 1:
        raise ValueError("top_k and batch_size must be positive")
    if config["chunking"].get("method", "fixed") != "fixed":
        raise ValueError("Only fixed chunking is supported in the current experiment")
    if config["retrieval"].get("method", "dense") not in {"dense", "hybrid"}:
        raise ValueError("Unknown retrieval method")
    if config["retrieval"].get("candidate_k", config["retrieval"]["top_k"]) < config["retrieval"]["top_k"]:
        raise ValueError("candidate_k must be >= top_k")
    return config
