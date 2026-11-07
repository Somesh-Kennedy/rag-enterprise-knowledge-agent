"""FastAPI service: POST /query, GET/POST /documents, GET /health.   Run: uvicorn rag.api:app"""
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.agent import KnowledgeAgent, build_llm
from rag.config import Settings
from rag.ingest import ingest
from rag.loader import split_documents, to_document
from rag.store import KnowledgeStore


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=10)


class DocumentIn(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,60}$")
    title: str = Field(min_length=1, max_length=200)
    department: str = "General"
    text: str = Field(min_length=10, max_length=20000)


def create_app(settings: Settings | None = None, embeddings=None, llm="auto") -> FastAPI:
    """`llm="auto"` uses Groq when GROQ_API_KEY is set; pass None (extractive) or any object with .invoke() to override."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="Enterprise Knowledge Agent", version="1.0.0")
    state: dict = {}

    def agent() -> KnowledgeAgent:
        if "agent" not in state:
            store = KnowledgeStore(settings, embeddings)
            if not store.load():
                store = ingest(settings, embeddings)          # first start: build the index from kb/
            model = build_llm(settings.llm_model) if llm == "auto" else llm
            state["agent"] = KnowledgeAgent(store, model, settings.top_k, settings.min_score)
            state["doc_ids"] = store.doc_ids()
        return state["agent"]

    @app.get("/health")
    def health():
        agent()
        return {"status": "ok", "vector_backend": settings.vector_backend, "documents": len(state["doc_ids"]),
                "answers": "generated" if state["agent"].llm else "extractive"}

    @app.post("/query")
    def query(body: QueryIn):
        result = agent().answer(body.question, body.k)
        return {**result, "sources": [asdict(s) for s in result["sources"]]}

    @app.get("/documents")
    def documents():
        agent()
        return {"count": len(state["doc_ids"]), "ids": sorted(state["doc_ids"])}

    @app.post("/documents", status_code=201)
    def add_document(body: DocumentIn):
        a = agent()
        if body.id in state["doc_ids"]:
            raise HTTPException(409, f"document {body.id} already exists")
        doc = to_document(body.model_dump(), source="api")
        a.store.add(split_documents([doc], settings.chunk_size, settings.chunk_overlap))
        a.store.save()
        state["doc_ids"].add(body.id)
        return {"id": body.id, "documents": len(state["doc_ids"])}

    return app


app = create_app()
