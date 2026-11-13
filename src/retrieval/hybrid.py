"""BM25 with Unicode word tokens and reciprocal rank fusion (RRF)."""

import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tokens = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.tokens]
        self.average = sum(self.lengths) / max(1, len(chunks))
        df = Counter(token for tokens in self.tokens for token in tokens)
        self.idf = {token: math.log(1 + (len(chunks) - count + .5) / (count + .5)) for token, count in df.items()}

    def search(self, question, top_k):
        scores = []
        for index, terms in enumerate(self.tokens):
            score = 0.0
            for term in set(tokenize(question)):
                frequency = terms[term]
                if frequency:
                    denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(1, self.average))
                    score += self.idf[term] * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return [dict(self.chunks[index], score=score, rank=rank, score_type="bm25")
                for rank, (index, score) in enumerate(scores[:top_k], 1)]


def fuse(dense, sparse, top_k, rrf_k=60):
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    records = {}
    for name, hits in (("dense", dense), ("bm25", sparse)):
        for rank, hit in enumerate(hits, 1):
            entry = records.setdefault(hit["chunk_id"], dict(hit, score=0.0, score_type="rrf"))
            entry["score"] += 1 / (rrf_k + rank)
            entry[name + "_rank"] = rank
            entry[name + "_score"] = hit["score"]
    ranked = sorted(records.values(), key=lambda hit: (-hit["score"], hit["chunk_id"]))[:top_k]
    return [dict(hit, rank=index) for index, hit in enumerate(ranked, 1)]


class HybridRetriever:
    def __init__(self, dense, config):
        self.dense, self.config = dense, config
        self.manifest, self.chunks = dense.manifest, dense.chunks
        self.sparse = BM25(self.chunks)

    def search(self, question, top_k):
        pool = max(top_k, self.config.get("hybrid_pool_k", 20))
        return fuse(self.dense.search(question, pool), self.sparse.search(question, pool),
                    top_k, self.config.get("rrf_k", 60))
