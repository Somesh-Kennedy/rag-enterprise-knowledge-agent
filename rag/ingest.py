"""Index the knowledge base:  python -m rag.ingest"""
import time

from rag.config import Settings
from rag.loader import load_documents, split_documents
from rag.store import KnowledgeStore


def ingest(settings: Settings | None = None, embeddings=None) -> KnowledgeStore:
    settings = settings or Settings.from_env()
    docs = load_documents(settings.kb_dir)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store = KnowledgeStore(settings, embeddings)
    start = time.perf_counter()
    store.build(chunks)
    store.save()
    model = "hashing" if settings.embed_backend == "hashing" else settings.embed_model
    print(f"indexed {len(docs)} documents as {len(chunks)} chunks in {time.perf_counter() - start:.1f}s "
          f"({settings.vector_backend}, {model})")
    return store


if __name__ == "__main__":
    ingest()
