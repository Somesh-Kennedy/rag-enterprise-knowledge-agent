# RAG Enterprise Knowledge Agent

A retrieval-augmented question-answering service for internal company documents. Ask a question in plain words and get
a short answer **with the source documents cited**, or a clear "I could not find this" when the knowledge base does not
cover it. Built with **LangChain**, a vector store (**FAISS** locally, **Pinecone** optional), **FastAPI**, and an
LLM served by **Groq** (`openai/gpt-oss-120b`).

> **The data is synthetic.** The 54 policy documents describe an invented company ("Harbor & Pine Logistics"). Nothing
> here comes from a real employer, and no real customer or employee data is used.

```
INDEXING   kb/*.md ─► parse front matter ─► split into chunks ─► embed (MiniLM, 384-dim) ─► FAISS index on disk

QUERYING   question ─► embed ─► top-k nearest chunks ─► drop any below MIN_SCORE ─┬─► none left: "I could not find this"
                                                                                  └─► LLM answers from the chunks only,
                                                                                      citing [doc-id]
```

## Contents
[How it works](#how-it-works) · [Worked example](#worked-example) · [API](#api) · [Results](#results) ·
[Configuration](#configuration) · [Quick start](#quick-start) · [Layout](#layout) · [Tests](#tests) · [Limitations](#limitations)

## How it works

**1. Documents.** Each file in `kb/` is Markdown with a small header (`id`, `title`, `department`) and a short policy
text, for example `hr-pto`: "Full-time employees earn 20 days of paid time off per year...". There are 54 documents
across HR (11), IT (11), Finance (9), Security (9), Legal (5), Engineering (5) and Facilities (4).

**2. Chunking.** `rag/loader.py` puts the title at the start of the document text (it carries a lot of the topic signal
for a short policy), then splits with LangChain's `RecursiveCharacterTextSplitter` (500 characters, 60 overlap). The
longest document here is 242 characters, so every document is a single chunk (54 documents, 54 chunks). The splitter
matters for longer documents and has a test for that case.

**3. Embeddings.** `sentence-transformers/all-MiniLM-L6-v2` turns text into a 384-number vector that captures meaning, so
"vacation" and "paid time off" land near each other. It runs on CPU, with no API key and no data leaving the machine.

**4. Vector store.** FAISS holds the vectors and finds the nearest ones. Vectors are normalized to length 1, and FAISS
returns the *squared* distance between them. `rag/store.py` converts that to cosine similarity (`1 - distance / 2`) so
the score means the same thing on FAISS and Pinecone. This conversion was wrong in my first version (it treated the value
as a plain similarity), which made the refusal threshold meaningless; a test now checks the score against the true cosine.
With `VECTOR_BACKEND=pinecone` the same LangChain interface writes to a Pinecone serverless index instead.

**5. Refusing.** `rag/agent.py` keeps only the chunks whose similarity is at least `MIN_SCORE` (0.25). If none qualify the
service answers "I could not find this in the knowledge base." and the LLM is never called. The threshold sits between the
lowest in-scope top score (0.299) and the highest off-topic top score (0.154) in the evaluation set.

**6. Answering.** The remaining chunks (top 4 by default) go into a prompt that allows only facts from the context, asks for
one or two sentences, and asks for `[doc-id]` citations. With no `GROQ_API_KEY` the service skips the LLM and returns the
best passage itself, marked `mode: "extractive"`, so it still works offline.

## Worked example
Real output from `POST /query` in extractive mode (no LLM), with `k = 2`:

```json
{"question": "How many vacation days do I get each year?", "k": 2}
```
```json
{
  "answer": "Company holidays\nThe company observes 11 paid holidays each year ... [hr-holidays]",
  "mode": "extractive",
  "sources": [
    {"doc_id": "hr-holidays", "title": "Company holidays",   "score": 0.5269},
    {"doc_id": "hr-sick",     "title": "Sick leave policy",  "score": 0.4695}
  ]
}
```
This is one of the five known misses: the words "vacation days" are closer to "holidays" and "sick leave" than to the
document titled "Paid time off policy" (`hr-pto`), which ranks third. With the default `top_k = 4` the right document is
still inside the context handed to the LLM, which is why generated answers are judged correct for all 54 questions even
though 5 documents are not ranked first. The extractive fallback has no such safety net.

An off-topic question is refused:
```json
{"question": "What is the capital of France?"}  ->  {"answer": "I could not find this in the knowledge base.", "mode": "no_match", "sources": []}
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /query` | `{"question": "...", "k": 1-10}` returns `answer`, `mode` (`generated`, `extractive` or `no_match`) and `sources` (id, title, department, score, snippet). Question is 3 to 500 characters. |
| `GET /documents` | Count and ids of the indexed documents. |
| `POST /documents` | Add a document at runtime (`id` lowercase letters, digits and dashes; `title`; `department`; `text`). It is embedded, added to the index and saved. A duplicate id returns 409; invalid input returns 422. |
| `GET /health` | Status, vector backend, number of documents, and whether answers are `generated` or `extractive`. |

Interactive docs are at `/docs` when the server is running.

## Results
Measured on this repo's 54 questions (one per document, written in different words from the document), using the real
embedding model on CPU. Re-run with `python -m eval.run_eval`.

| Metric | Result |
|---|---|
| Right document ranked 1st (hit@1) | **90.7%** (49 of 54) |
| Right document in the top 3 / top 5 | **100%** / **100%** |
| Mean reciprocal rank | **0.944** |
| Retrieval time per query | about 12 ms |
| Out-of-scope questions refused (10 off-topic questions) | **10 of 10** |
| Lowest in-scope top score / highest out-of-scope top score | 0.299 / 0.154 (threshold 0.25 sits between) |

The five questions where the right document is not first are all vocabulary gaps: "vacation days" vs "Paid time off
policy", "everybody expected in the office" vs the remote-work policy, "first review in the first three months" vs the
probation policy, "months appraisals happen" vs the review policy, and "guest walk around alone" vs the badge policy. A
keyword (BM25) hybrid or a cross-encoder reranker is the obvious next step.

Answer quality with the LLM (`python -m eval.run_eval --generate`): answers written by `openai/gpt-oss-120b` on Groq and
graded by a different model (`qwen/qwen3.8-27b`) against the reference fact for each question.

| Metric | Result |
|---|---|
| Answers judged correct by the second model | **54 of 54** |
| Reference fact appears word for word in the answer (strict string match) | 27 of 54 (the rest are paraphrases, e.g. "twenty" vs "20") |
| Answer cites the expected document id | 52 of 54 |
| Out-of-scope questions refused end to end | 10 of 10 |

**Read these numbers carefully.** The documents and questions were written together, the corpus is small, and the 10
out-of-scope questions are easy. The grader is another LLM, not a person. This is a regression check for the pipeline, not
a benchmark. The Pinecone backend uses the same LangChain interface but has not been run against a live Pinecone index.

## Configuration
Environment variables (a `.env` file is loaded; see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | (none) | Enables generated answers. Without it the service answers extractively. |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq chat model used to answer. |
| `JUDGE_MODEL` | (answer model) | Different Groq model used only by the evaluation script to grade answers. |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `pinecone` (also needs `PINECONE_API_KEY`). |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 60 | Splitter settings in characters. |
| `TOP_K` | 4 | Chunks retrieved per question. |
| `MIN_SCORE` | 0.25 | Cosine similarity below which a chunk is ignored. |

## Quick start
```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add GROQ_API_KEY for generated answers

python -m rag.ingest                                  # build the index
uvicorn rag.api:app --reload                          # http://127.0.0.1:8000/docs
python -m eval.run_eval                               # retrieval metrics (add --generate for answer quality)
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
eval/               run_eval.py, the 54 questions (with reference facts) and the 10 off-topic questions
tests/              20 offline tests (hashing embeddings and fake LLMs: no model download or API key needed)
```

## Tests
```bash
pytest -q
```
Cover loading and chunking, FAISS save/load, that the reported score is the true cosine similarity, extractive and
generated answers, refusal of out-of-scope questions, and every API endpoint including validation and duplicates. They use
a small hashing embedding and fake LLMs, so they run in seconds without a download or a key.

## Limitations
- Small synthetic corpus; real enterprise documents (PDF, tables, scans) need parsing this project does not do.
- No authentication or per-department access control on the API.
- FAISS is in-process and file-based; Pinecone support is untested against a live index.
- Chunks are whole short documents here; long documents would need more careful chunking and evaluation.
- Retrieval uses embeddings only, so synonyms the model does not know can miss (see the worked example).
