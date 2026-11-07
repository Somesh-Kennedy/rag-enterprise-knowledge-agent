"""Settings, read from environment variables (a .env file is loaded if present)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    kb_dir: Path = ROOT / "kb"
    index_dir: Path = ROOT / ".index"
    vector_backend: str = "faiss"                      # faiss | pinecone
    embed_backend: str = "huggingface"                 # huggingface | hashing (offline, used by the tests)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pinecone_index: str = "enterprise-kb"
    chunk_size: int = 500
    chunk_overlap: int = 60
    top_k: int = 4
    min_score: float = 0.25                            # cosine similarity below this means "not in the knowledge base"
    llm_model: str = "llama-3.3-70b-versatile"         # served by Groq when GROQ_API_KEY is set

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            kb_dir=Path(e("KB_DIR", str(ROOT / "kb"))),
            index_dir=Path(e("INDEX_DIR", str(ROOT / ".index"))),
            vector_backend=e("VECTOR_BACKEND", "faiss"),
            embed_backend=e("EMBED_BACKEND", "huggingface"),
            embed_model=e("EMBED_MODEL", cls.embed_model),
            pinecone_index=e("PINECONE_INDEX", cls.pinecone_index),
            chunk_size=int(e("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(e("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(e("TOP_K", cls.top_k)),
            min_score=float(e("MIN_SCORE", cls.min_score)),
            llm_model=e("LLM_MODEL", cls.llm_model),
        )
