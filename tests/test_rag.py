"""Offline tests: they use the hashing embedding and fake LLMs, so no model download or API key is needed."""
import dataclasses

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from eval.run_eval import rank_of
from rag.agent import NOT_FOUND, KnowledgeAgent, Source
from rag.api import create_app
from rag.config import ROOT, Settings
from rag.embeddings import HashingEmbeddings
from rag.loader import load_documents, parse_markdown, split_documents
from rag.store import KnowledgeStore


@pytest.fixture()
def settings(tmp_path):
    return dataclasses.replace(Settings(), embed_backend="hashing", index_dir=tmp_path / "index", kb_dir=ROOT / "kb")


@pytest.fixture()
def store(settings):
    s = KnowledgeStore(settings, HashingEmbeddings())
    s.build(split_documents(load_documents(settings.kb_dir), settings.chunk_size, settings.chunk_overlap))
    return s


class FakeLLM:
    def __init__(self, reply="Employees get 20 days. [hr-pto]"):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Reply", (), {"content": self.reply})()


# ---------------------------------------------------------------- loading and chunking
def test_parse_markdown_reads_front_matter():
    item = parse_markdown("---\nid: x-1\ntitle: A title\ndepartment: IT\n---\nBody text here.\n", "fallback")
    assert item == {"id": "x-1", "title": "A title", "department": "IT", "text": "Body text here."}


def test_parse_markdown_without_front_matter_uses_defaults():
    item = parse_markdown("Just text", "some-doc")
    assert item["id"] == "some-doc" and item["title"] == "Some Doc" and item["department"] == "General"


def test_knowledge_base_has_more_than_50_documents_with_unique_ids(settings):
    docs = load_documents(settings.kb_dir)
    assert len(docs) > 50
    assert len({d.metadata["doc_id"] for d in docs}) == len(docs)


def test_long_documents_are_split_and_keep_metadata():
    doc = Document(page_content="word " * 400, metadata={"doc_id": "long-1", "title": "Long"})
    chunks = split_documents([doc], 500, 60)
    assert len(chunks) > 1
    assert all(c.metadata["doc_id"] == "long-1" for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------- vector store
def test_exact_chunk_text_scores_cosine_one(store):
    doc, score = store.search(load_documents(store.settings.kb_dir)[0].page_content, 1)[0]
    assert score == pytest.approx(1.0, abs=1e-3)


def test_store_score_equals_true_cosine_similarity(store):
    emb = HashingEmbeddings()
    query, doc_text = "annual budget finalized in november", load_documents(store.settings.kb_dir)
    doc, score = store.search(query, 1)[0]
    q, d = emb.embed_query(query), emb.embed_query(doc.page_content)
    assert score == pytest.approx(sum(a * b for a, b in zip(q, d)), abs=1e-4)


def test_index_round_trips_through_disk(settings, store):
    store.save()
    reloaded = KnowledgeStore(settings, HashingEmbeddings())
    assert reloaded.load() is True
    assert reloaded.search("paid time off days per year", 1)[0][0].metadata["doc_id"] == "hr-pto"


def test_load_returns_false_when_no_index_exists(settings):
    assert KnowledgeStore(settings, HashingEmbeddings()).load() is False


# ---------------------------------------------------------------- agent
def test_retrieve_finds_the_right_document(store):
    agent = KnowledgeAgent(store, None, top_k=3)
    assert agent.retrieve("How many days of paid time off per year?")[0].doc_id == "hr-pto"


def test_extractive_answer_cites_the_best_source(store):
    out = KnowledgeAgent(store, None).answer("How many days of paid time off per year?")
    assert out["mode"] == "extractive"
    assert "20 days" in out["answer"] and out["answer"].endswith("[hr-pto]")


def test_question_outside_the_knowledge_base_is_refused(store):
    out = KnowledgeAgent(store, FakeLLM()).answer("zebra quantum telescope galaxies")
    assert out == {"answer": NOT_FOUND, "mode": "no_match", "sources": []}


def test_generated_answer_sends_only_retrieved_context_to_the_llm(store):
    llm = FakeLLM()
    out = KnowledgeAgent(store, llm).answer("How many days of paid time off per year?")
    assert out["mode"] == "generated" and out["answer"] == "Employees get 20 days. [hr-pto]"
    prompt = llm.prompts[0]
    assert "[hr-pto]" in prompt and "using only the context" in prompt and NOT_FOUND in prompt


def test_source_is_a_plain_dataclass():
    assert dataclasses.asdict(Source("a", "T", "D", 0.5, "snip"))["doc_id"] == "a"


def test_rank_of_deduplicates_chunks_of_the_same_document():
    sources = [Source("a", "", "", 0, ""), Source("a", "", "", 0, ""), Source("b", "", "", 0, "")]
    assert rank_of(sources, "b") == 2 and rank_of(sources, "zzz") is None


# ---------------------------------------------------------------- API
@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings, HashingEmbeddings(), llm=None))


def test_health_reports_documents_and_answer_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] > 50 and body["answers"] == "extractive"


def test_query_returns_answer_and_sources(client):
    body = client.post("/query", json={"question": "How many days of paid time off per year?"}).json()
    assert body["sources"][0]["doc_id"] == "hr-pto" and "20 days" in body["answer"]


def test_query_validates_input(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "valid question here", "k": 99}).status_code == 422


def test_documents_lists_ids(client):
    body = client.get("/documents").json()
    assert body["count"] == len(body["ids"]) and "hr-pto" in body["ids"]


def test_ingesting_a_document_makes_it_searchable(client):
    new = {"id": "hr-pets", "title": "Office pets", "department": "HR",
           "text": "Dogs are welcome in the Dallas office on Fridays if registered with facilities."}
    assert client.post("/documents", json=new).status_code == 201
    body = client.post("/query", json={"question": "Are dogs welcome in the office on Fridays?"}).json()
    assert body["sources"][0]["doc_id"] == "hr-pets"


def test_duplicate_and_invalid_documents_are_rejected(client):
    dup = {"id": "hr-pto", "title": "x", "text": "Some replacement text for the policy."}
    assert client.post("/documents", json=dup).status_code == 409
    bad = {"id": "Bad ID!", "title": "x", "text": "Some replacement text for the policy."}
    assert client.post("/documents", json=bad).status_code == 422
