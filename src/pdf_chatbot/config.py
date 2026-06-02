from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    top_k: int
    provider: str
    openai_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    gemini_api_key: str | None
    gemini_model: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        return cls(
            storage_dir=Path(os.getenv("PDF_CHAT_STORAGE", "storage")),
            top_k=int(os.getenv("TOP_K", "6")),
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        )
