from pdf_chatbot.answer import _extractive_fallback
from pdf_chatbot.models import SearchHit


def test_fallback_mentions_source():
    hits = [SearchHit("1", "a.pdf", 1, "The phone number is 555-1234.", 0.8)]
    answer = _extractive_fallback("phone number", hits)

    assert "a.pdf" in answer
    assert "555-1234" in answer
