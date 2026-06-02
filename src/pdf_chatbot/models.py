from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    source: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page_number: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source: str
    page_number: int
    text: str
    score: float


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[SearchHit]
