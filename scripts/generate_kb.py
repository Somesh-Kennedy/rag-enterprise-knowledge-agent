"""Writes the synthetic knowledge base (kb/*.md) and the evaluation questions (eval/questions.json)."""
import json
from pathlib import Path

from kb_data import DOCS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    kb = ROOT / "kb"
    kb.mkdir(exist_ok=True)
    for old in kb.glob("*.md"):
        old.unlink()
    questions = []
    for doc_id, dept, title, text, question, fact in DOCS:
        (kb / f"{doc_id}.md").write_text(
            f"---\nid: {doc_id}\ntitle: {title}\ndepartment: {dept}\n---\n{text}\n", encoding="utf-8"
        )
        questions.append({"question": question, "doc_id": doc_id, "fact": fact})
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"wrote {len(DOCS)} documents and {len(questions)} questions")


if __name__ == "__main__":
    main()
