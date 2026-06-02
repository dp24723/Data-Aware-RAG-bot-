from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from pdf_chatbot.answer import answer_question
from pdf_chatbot.config import Settings
from pdf_chatbot.store import VectorStore

app = FastAPI(title="PDF Chatbot")
settings = Settings.from_env()
store = VectorStore(settings.storage_dir)


class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": store.exists(), "provider": settings.provider}


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for file in files:
            target = folder / file.filename
            with target.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        count = store.build(
            folder,
            provider=settings.provider,
            openai_api_key=settings.openai_api_key,
            openai_embedding_model=settings.openai_embedding_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )
    return {"chunks": count}


@app.post("/ask")
def ask(req: QuestionRequest):
    result = answer_question(
        req.question,
        store,
        provider=settings.provider,
        openai_api_key=settings.openai_api_key,
        openai_chat_model=settings.openai_chat_model,
        openai_embedding_model=settings.openai_embedding_model,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_chat_model=settings.ollama_chat_model,
        ollama_embedding_model=settings.ollama_embedding_model,
        top_k=req.top_k or settings.top_k,
    )
    return {
        "answer": result.answer,
        "sources": [source.__dict__ for source in result.sources],
    }
