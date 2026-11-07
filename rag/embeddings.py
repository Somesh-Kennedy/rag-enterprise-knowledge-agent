"""Embedding models: a real sentence-transformer by default, and a tiny hashing model for offline tests."""
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding. Needs no model download; only matches shared words."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9$%]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings(backend: str, model: str) -> Embeddings:
    if backend == "hashing":
        return HashingEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
