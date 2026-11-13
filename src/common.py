import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("documents", "queries", "chunks", "index_dir"):
        config[key] = str(ROOT / config[key])
    size, overlap = config["chunking"]["size"], config["chunking"]["overlap"]
    if not isinstance(size, int) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("Require 0 <= chunk overlap < size")
    if config["retrieval"]["top_k"] < 1 or config["embedding"]["batch_size"] < 1:
        raise ValueError("top_k and batch_size must be positive")
    return config
