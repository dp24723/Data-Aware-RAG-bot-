import re
from hashlib import sha1

from pdf_chatbot.models import Chunk, Page


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_page(page: Page, chunk_size: int = 1100, overlap: int = 180) -> list[Chunk]:
    sentences = split_sentences(page.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def add_chunk(parts: list[str]) -> None:
        if not parts:
            return
        text = clean_text(" ".join(parts))
        if not text:
            return
        digest = sha1(f"{page.source}:{page.page_number}:{len(chunks)}:{text[:80]}".encode()).hexdigest()[:12]
        chunks.append(Chunk(chunk_id=digest, source=page.source, page_number=page.page_number, text=text))

    for sentence in sentences:
        if len(sentence) > chunk_size:
            add_chunk(current)
            current, current_len = [], 0
            start = 0
            while start < len(sentence):
                end = min(start + chunk_size, len(sentence))
                text = sentence[start:end]
                digest = sha1(f"{page.source}:{page.page_number}:{start}:{text[:80]}".encode()).hexdigest()[:12]
                chunks.append(Chunk(digest, page.source, page.page_number, clean_text(text)))
                if end == len(sentence):
                    break
                start = max(0, end - overlap)
            continue

        if current_len + len(sentence) + 1 > chunk_size:
            add_chunk(current)
            tail = " ".join(current)[-overlap:] if current else ""
            current = [tail, sentence] if tail else [sentence]
            current_len = sum(len(x) + 1 for x in current)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    add_chunk(current)
    return chunks
