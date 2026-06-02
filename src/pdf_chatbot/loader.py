from pathlib import Path

from pypdf import PdfReader

from pdf_chatbot.models import Page

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[Page] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(Page(source=path.name, page_number=idx, text=text))
        return pages

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Page(source=path.name, page_number=1, text=text)] if text.strip() else []

    raise ValueError(f"Unsupported file type: {path.suffix}")


def iter_files(path: Path):
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file())

    for file_path in files:
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path
