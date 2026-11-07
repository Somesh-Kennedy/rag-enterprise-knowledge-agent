"""Loads markdown documents (with a small front-matter block) and splits them into chunks."""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_markdown(text: str, fallback_id: str) -> dict:
    """Return {id, title, department, text} from a file with optional `key: value` front matter."""
    meta = {"id": fallback_id, "title": fallback_id.replace("-", " ").title(), "department": "General"}
    body = text
    if text.startswith("---\n"):
        head, _, rest = text[4:].partition("\n---\n")
        for line in head.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in meta and value.strip():
                meta[key.strip()] = value.strip()
        body = rest
    return {**meta, "text": body.strip()}


def load_documents(kb_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        item = parse_markdown(path.read_text(encoding="utf-8"), path.stem)
        docs.append(to_document(item, source=path.name))
    return docs


def to_document(item: dict, source: str) -> Document:
    # The title is part of the embedded text: it carries a lot of the topic signal for short policies.
    return Document(
        page_content=f"{item['title']}\n{item['text']}",
        metadata={"doc_id": item["id"], "title": item["title"], "department": item["department"], "source": source},
    )


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks
