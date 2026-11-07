"""Measure retrieval (and, with a Groq key, answer quality) on the 54 evaluation questions.

    python -m eval.run_eval                 # retrieval only: hit@k, MRR, out-of-scope rejection
    python -m eval.run_eval --generate      # also answer with the LLM and grade the answers

Writes eval/results.json. The knowledge base and the questions are synthetic, and the questions were written
alongside the documents, so treat the numbers as a regression check for this pipeline, not as a benchmark.
"""
import argparse
import json
import os
import time
from pathlib import Path

from rag.agent import NOT_FOUND, KnowledgeAgent, build_llm
from rag.config import Settings
from rag.ingest import ingest
from rag.store import KnowledgeStore

HERE = Path(__file__).resolve().parent

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the 2022 football World Cup?",
    "How do I bake sourdough bread?",
    "What is the boiling point of water at sea level?",
    "Explain how photosynthesis works.",
    "What's the best programming language for beginners?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the current price of Bitcoin?",
    "How tall is Mount Everest?",
    "Recommend a good movie for tonight.",
]

JUDGE = """You grade an answer to an employee question.
Question: {question}
Reference fact (must be reflected in a correct answer): {fact}
Answer given: {answer}
Reply with one word, YES if the answer states or clearly implies the reference fact and does not contradict it, otherwise NO."""


def rank_of(sources, doc_id):
    seen = []
    for s in sources:
        if s.doc_id not in seen:
            seen.append(s.doc_id)
    return seen.index(doc_id) + 1 if doc_id in seen else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--generate", action="store_true", help="also generate and grade answers (needs GROQ_API_KEY)")
    args = ap.parse_args()

    settings = Settings.from_env()
    store = KnowledgeStore(settings)
    if not store.load():
        store = ingest(settings)
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    # ---- retrieval ----
    agent = KnowledgeAgent(store, None, args.k, settings.min_score)
    ranks, misses, start = [], [], time.perf_counter()
    for q in questions:
        r = rank_of(agent.retrieve(q["question"], args.k), q["doc_id"])
        ranks.append(r)
        if r is None or r > 1:
            misses.append({"question": q["question"], "expected": q["doc_id"], "rank": r})
    elapsed = time.perf_counter() - start
    n = len(questions)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    results = {
        "questions": n,
        "documents": len(store.doc_ids()),
        "embed_model": settings.embed_model,
        "hit@1": round(hit(1), 3), "hit@3": round(hit(3), 3), "hit@5": round(hit(5), 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "avg_retrieval_ms": round(elapsed / n * 1000, 1),
        "not_top1": misses,
    }

    # ---- out-of-scope questions should be refused, not answered ----
    refused = sum(1 for q in OUT_OF_SCOPE if not [s for s in agent.retrieve(q, 1) if s.score >= settings.min_score])
    results["out_of_scope_rejected"] = f"{refused}/{len(OUT_OF_SCOPE)}"
    results["min_score"] = settings.min_score

    # ---- generation (optional) ----
    if args.generate:
        llm = build_llm(settings.llm_model)
        if llm is None:
            raise SystemExit("--generate needs GROQ_API_KEY")
        judge_name = os.environ.get("JUDGE_MODEL", settings.llm_model)       # grade with a different model than the one answering
        judge = build_llm(judge_name)
        gen = KnowledgeAgent(store, llm, settings.top_k, settings.min_score)
        fact_in_answer = cited = judged = 0
        wrong = []
        for q in questions:
            out = gen.answer(q["question"])
            ans = out["answer"]
            ok_fact = q["fact"].lower() in ans.lower()
            fact_in_answer += ok_fact
            cited += f"[{q['doc_id']}]" in ans
            verdict = judge.invoke(JUDGE.format(question=q["question"], fact=q["fact"], answer=ans)).content.strip().upper()
            correct = verdict.startswith("YES")
            judged += correct
            if not correct:
                wrong.append({"question": q["question"], "answer": ans, "fact": q["fact"]})
            time.sleep(0.3)
        oos_refused = sum(1 for q in OUT_OF_SCOPE if gen.answer(q)["answer"].startswith(NOT_FOUND[:20]))
        results["generation"] = {
            "llm": settings.llm_model,
            "judge": judge_name,
            "answers_judged_correct": f"{judged}/{n}",
            "exact_fact_string_in_answer": f"{fact_in_answer}/{n}",
            "cites_expected_doc": f"{cited}/{n}",
            "out_of_scope_refused": f"{oos_refused}/{len(OUT_OF_SCOPE)}",
            "judged_wrong": wrong,
        }

    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("not_top1",)}, indent=2))
    print(f"questions where the right document was not ranked first: {len(misses)}")


if __name__ == "__main__":
    main()
