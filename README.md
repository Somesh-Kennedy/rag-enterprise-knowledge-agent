# RAG Enterprise Knowledge Agent

A retrieval-augmented question-answering service for internal company documents. Ask a question in plain words and get
a short answer **with the source documents cited**, or a clear "I could not find this" when the knowledge base does not
cover it. Built with **LangChain**, a vector store (**FAISS** locally, **Pinecone** optional), **FastAPI**, and an
LLM served by **Groq** (`openai/gpt-oss-120b`).

> **The data is synthetic.** The 54 policy documents describe an invented company ("Harbor & Pine Logistics"). Nothing
> here comes from a real employer, and no real customer or employee data is used.

```
question ─► embed (MiniLM) ─► vector search (top-k) ─► score filter ─► LLM answers only from the chunks, cites [doc-id]
                                                          │
                                                          └─ nothing close enough ─► "I could not find this in the knowledge base."
```

## What it does
- **Indexing pipeline** (`python -m rag.ingest`): reads `kb/*.md` (front matter: id, title, department), splits into
  chunks, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and saves a FAISS index. Switch to Pinecone with
  `VECTOR_BACKEND=pinecone`.
- **Grounded answers**: the prompt allows only facts from the retrieved context and asks for `[doc-id]` citations.
- **Refuses out-of-scope questions** using a cosine-similarity threshold (`MIN_SCORE`, default 0.25), so the LLM is not
  even called when nothing relevant was found.
- **No-LLM fallback**: without a `GROQ_API_KEY` the service returns the best passage itself (`mode: "extractive"`).
- **FastAPI service** with input validation: `POST /query`, `GET /documents`, `POST /documents` (add a document at
  runtime; duplicates return 409), `GET /health`.

## Results
Measured on this repo's 54 questions (one per document, written in different words from the document), using the real
embedding model on CPU. Re-run with `python -m eval.run_eval`.

| Metric | Result |
|---|---|
| Right document ranked 1st (hit@1) | **90.7%** (49 of 54) |
| Right document in the top 3 / top 5 | **100%** / **100%** |
| Mean reciprocal rank | **0.944** |
| Retrieval time per query | about 8 ms |
| Out-of-scope questions refused (10 off-topic questions) | **10 of 10** |
| Lowest in-scope top score / highest out-of-scope top score | 0.299 / 0.154 (threshold 0.25 sits between) |

The five questions where the right document is not first are all vocabulary gaps, for example "vacation days" versus a
document titled "Paid time off policy". A keyword (BM25) hybrid or a cross-encoder reranker is the obvious next step.

Answer quality with the LLM (`python -m eval.run_eval --generate`), answers by `openai/gpt-oss-120b` on Groq and graded
by a different model (`qwen/qwen3.8-27b`) against the reference fact:

| Metric | Result |
|---|---|
| Answers judged correct by the second model | **54 of 54** |
| Reference fact appears word for word in the answer (strict string match) | 27 of 54 (the rest are paraphrases, e.g. "twenty" vs "20") |
| Answer cites the expected document id | 52 of 54 |
| Out-of-scope questions refused end to end | 10 of 10 |

**Read these numbers carefully.** The documents and questions were written together, the corpus is small, and the 10
out-of-scope questions are easy. This is a regression check for the pipeline, not a benchmark. The Pinecone backend uses
the same LangChain interface but has not been run against a live Pinecone index.

## Quick start
```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add GROQ_API_KEY for generated answers

python -m rag.ingest                                  # build the index
uvicorn rag.api:app --reload                          # http://127.0.0.1:8000/docs
```
```bash
curl -s localhost:8000/query -H "content-type: application/json" \
     -d '{"question": "How often do I get a new work computer?"}'
```

## Layout
```
kb/                 54 synthetic policy documents (HR, IT, Security, Finance, Facilities, Legal, Engineering)
scripts/            kb_data.py (source text and eval questions) and generate_kb.py (writes kb/ and eval/questions.json)
rag/                config, loader (front matter + chunking), embeddings, store (FAISS/Pinecone), agent, ingest, api
eval/               run_eval.py and the question set
tests/              20 offline tests (hashing embeddings and fake LLMs: no model download or API key needed)
```

## Tests
```bash
pytest -q
```
Cover loading and chunking, FAISS save/load, that the reported score is the true cosine similarity, extractive and
generated answers, refusal of out-of-scope questions, and every API endpoint including validation and duplicates.

## Limitations
- Small synthetic corpus; real enterprise documents (PDF, tables, scans) need parsing this project does not do.
- No authentication or per-department access control on the API.
- FAISS is in-process and file-based; Pinecone support is untested against a live index.
- Chunks are whole short documents here; long documents would need more careful chunking and evaluation.
