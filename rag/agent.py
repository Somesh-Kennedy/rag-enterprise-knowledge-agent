"""The question-answering agent: retrieve chunks, then answer only from them and cite the sources."""
import os
from dataclasses import dataclass

from rag.store import KnowledgeStore

NOT_FOUND = "I could not find this in the knowledge base."

PROMPT = """You answer employee questions using only the context below.
- Use only facts stated in the context. If the context does not answer the question, reply exactly: {not_found}
- Be brief (one or two sentences) and cite the source ids you used in square brackets, e.g. [hr-pto].

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class Source:
    doc_id: str
    title: str
    department: str
    score: float
    snippet: str


class KnowledgeAgent:
    def __init__(self, store: KnowledgeStore, llm=None, top_k: int = 4, min_score: float = 0.25):
        self.store, self.llm, self.top_k, self.min_score = store, llm, top_k, min_score

    def retrieve(self, question: str, k: int | None = None) -> list[Source]:
        hits = self.store.search(question, k or self.top_k)
        return [
            Source(d.metadata["doc_id"], d.metadata["title"], d.metadata["department"], round(score, 4), d.page_content)
            for d, score in hits
        ]

    def answer(self, question: str, k: int | None = None) -> dict:
        sources = self.retrieve(question, k)
        usable = [s for s in sources if s.score >= self.min_score]
        if not usable:
            return {"answer": NOT_FOUND, "mode": "no_match", "sources": []}
        if self.llm is None:
            # No LLM configured: return the best passage itself, clearly marked as extractive.
            best = usable[0]
            return {"answer": f"{best.snippet} [{best.doc_id}]", "mode": "extractive", "sources": usable}
        context = "\n\n".join(f"[{s.doc_id}] {s.snippet}" for s in usable)
        reply = self.llm.invoke(PROMPT.format(not_found=NOT_FOUND, context=context, question=question))
        text = getattr(reply, "content", str(reply)).strip()
        return {"answer": text, "mode": "generated", "sources": usable}


def build_llm(model: str):
    """A Groq-hosted chat model if GROQ_API_KEY is set, otherwise None (extractive answers)."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=0)
