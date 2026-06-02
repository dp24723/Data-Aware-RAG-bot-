from pathlib import Path
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from pdf_chatbot.chunker import chunk_page
from pdf_chatbot.loader import iter_files, load_pages
from pdf_chatbot.models import Chunk, SearchHit


class VectorStore:
    def __init__(self, storage_dir: Path | str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.chunks_path = self.storage_dir / "chunks.jsonl"
        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.tfidf_path = self.storage_dir / "tfidf.joblib"
        self.meta_path = self.storage_dir / "index_meta.json"

    def exists(self) -> bool:
        return self.chunks_path.exists() and (self.embeddings_path.exists() or self.tfidf_path.exists())

    def reset(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.chunks_path, self.embeddings_path, self.tfidf_path, self.meta_path]:
            if path.exists():
                path.unlink()

    def build(
        self,
        path: Path | str,
        *,
        provider: str = "tfidf",
        openai_api_key: str | None = None,
        openai_embedding_model: str = "text-embedding-3-small",
        ollama_base_url: str = "http://localhost:11434",
        ollama_embedding_model: str = "nomic-embed-text",
    ) -> int:
        path = Path(path)
        chunks: list[Chunk] = []
        for file_path in iter_files(path):
            for page in load_pages(file_path):
                chunks.extend(chunk_page(page))

        if not chunks:
            raise ValueError("No readable PDF/text content found.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._write_chunks(chunks)

        provider = provider.lower().strip()
        texts = [c.text for c in chunks]

        if provider == "openai" and openai_api_key:
            vectors = embed_openai(texts, api_key=openai_api_key, model=openai_embedding_model)
            np.save(self.embeddings_path, vectors)
            meta = {"retrieval": "openai_embeddings", "embedding_model": openai_embedding_model, "chunks": len(chunks)}
        elif provider == "ollama":
            try:
                vectors = embed_ollama(texts, base_url=ollama_base_url, model=ollama_embedding_model)
                np.save(self.embeddings_path, vectors)
                meta = {"retrieval": "ollama_embeddings", "embedding_model": ollama_embedding_model, "chunks": len(chunks)}
            except Exception as exc:
                # Keep the app useful even if the embedding model has not been pulled yet.
                self._build_tfidf(texts)
                meta = {
                    "retrieval": "tfidf_fallback",
                    "embedding_model": None,
                    "chunks": len(chunks),
                    "note": f"Ollama embeddings were not available: {exc}",
                }
        else:
            self._build_tfidf(texts)
            meta = {"retrieval": "tfidf", "embedding_model": None, "chunks": len(chunks)}

        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return len(chunks)

    def _build_tfidf(self, texts: list[str]) -> None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, self.tfidf_path)
        if self.embeddings_path.exists():
            self.embeddings_path.unlink()

    def chunks(self) -> list[Chunk]:
        if not self.chunks_path.exists():
            return []
        rows = [json.loads(line) for line in self.chunks_path.read_text(encoding="utf-8").splitlines()]
        return [Chunk(**row) for row in rows]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        provider: str = "tfidf",
        openai_api_key: str | None = None,
        openai_embedding_model: str = "text-embedding-3-small",
        ollama_base_url: str = "http://localhost:11434",
        ollama_embedding_model: str = "nomic-embed-text",
    ) -> list[SearchHit]:
        if not self.exists():
            raise FileNotFoundError("No index found. Upload documents and build the index first.")

        chunks = self.chunks()
        provider = provider.lower().strip()

        if self.embeddings_path.exists() and provider == "openai" and openai_api_key:
            doc_vectors = np.load(self.embeddings_path)
            query_vector = embed_openai([query], api_key=openai_api_key, model=openai_embedding_model)[0]
            scores = cosine_np(query_vector, doc_vectors)
        elif self.embeddings_path.exists() and provider == "ollama":
            doc_vectors = np.load(self.embeddings_path)
            query_vector = embed_ollama([query], base_url=ollama_base_url, model=ollama_embedding_model)[0]
            scores = cosine_np(query_vector, doc_vectors)
        elif self.tfidf_path.exists():
            payload = joblib.load(self.tfidf_path)
            query_vec = payload["vectorizer"].transform([query])
            scores = cosine_similarity(query_vec, payload["matrix"]).ravel()
        else:
            return []

        order = np.argsort(scores)[::-1][: max(1, top_k)]
        hits: list[SearchHit] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            c = chunks[int(idx)]
            hits.append(SearchHit(c.chunk_id, c.source, c.page_number, c.text, score))
        return hits

    def _write_chunks(self, chunks: list[Chunk]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def embed_openai(texts: list[str], *, api_key: str, model: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return normalize(np.array(vectors, dtype="float32"))


def embed_ollama(texts: list[str], *, base_url: str, model: str) -> np.ndarray:
    import requests

    vectors: list[list[float]] = []
    url = base_url.rstrip("/") + "/api/embeddings"
    for text in texts:
        response = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
        response.raise_for_status()
        vectors.append(response.json()["embedding"])
    return normalize(np.array(vectors, dtype="float32"))


def normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def cosine_np(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    q = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    docs = doc_vectors / np.clip(np.linalg.norm(doc_vectors, axis=1, keepdims=True), 1e-12, None)
    return docs @ q
