from pdf_chatbot.store import VectorStore


def test_store_tfidf_fallback(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("The project uses thermal risk analysis and material compatibility checks.")

    store = VectorStore(tmp_path / "storage")
    count = store.build(docs, provider="tfidf")
    hits = store.search("thermal risk", top_k=3, provider="tfidf")

    assert count >= 1
    assert hits
    assert "thermal" in hits[0].text.lower()
