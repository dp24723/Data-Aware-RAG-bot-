from pdf_chatbot.chunker import chunk_page
from pdf_chatbot.models import Page


def test_chunker_keeps_page_metadata():
    page = Page(source="sample.pdf", page_number=2, text="This is sentence one. " * 120)
    chunks = chunk_page(page, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].text
