"""Vector store wrapper: FAISS on disk by default, Pinecone when VECTOR_BACKEND=pinecone.

Both go through LangChain's VectorStore interface. `search` always returns cosine similarity (higher is better),
so the "not in the knowledge base" threshold means the same thing on either backend.
"""
import os

from langchain_core.documents import Document

from rag.config import Settings
from rag.embeddings import get_embeddings


class KnowledgeStore:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings or get_embeddings(settings.embed_backend, settings.embed_model)
        self.store = None

    # -- building -------------------------------------------------------------------------
    def build(self, chunks: list[Document]) -> None:
        if self.settings.vector_backend == "pinecone":
            self.store = self._pinecone(chunks)
        else:
            from langchain_community.vectorstores import FAISS

            self.store = FAISS.from_documents(chunks, self.embeddings, normalize_L2=True)

    def save(self) -> None:
        if self.settings.vector_backend == "faiss" and self.store is not None:
            self.settings.index_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.settings.index_dir))

    def load(self) -> bool:
        if self.settings.vector_backend == "pinecone":
            from langchain_pinecone import PineconeVectorStore

            self.store = PineconeVectorStore(index_name=self.settings.pinecone_index, embedding=self.embeddings)
            return True
        from langchain_community.vectorstores import FAISS

        if not (self.settings.index_dir / "index.faiss").exists():
            return False
        self.store = FAISS.load_local(
            str(self.settings.index_dir), self.embeddings, allow_dangerous_deserialization=True
        )
        return True

    def add(self, chunks: list[Document]) -> None:
        self.store.add_documents(chunks)

    def doc_ids(self) -> set[str]:
        """Document ids in the local index (FAISS only; Pinecone would need a metadata query)."""
        if hasattr(self.store, "docstore"):
            return {d.metadata["doc_id"] for d in self.store.docstore._dict.values()}
        return set()

    # -- searching ------------------------------------------------------------------------
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        hits = self.store.similarity_search_with_score(query, k=k)
        if self.settings.vector_backend == "pinecone":
            return [(doc, float(score)) for doc, score in hits]          # a cosine Pinecone index already gives similarity
        # FAISS returns the squared L2 distance between unit vectors, so cosine = 1 - dist / 2
        return [(doc, 1.0 - float(dist) / 2.0) for doc, dist in hits]

    def _pinecone(self, chunks: list[Document]):
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        name = self.settings.pinecone_index
        if name not in [i["name"] for i in client.list_indexes()]:
            dim = len(self.embeddings.embed_query("dimension probe"))
            client.create_index(
                name=name, dimension=dim, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_REGION", "us-east-1")),
            )
        return PineconeVectorStore.from_documents(chunks, self.embeddings, index_name=name)
