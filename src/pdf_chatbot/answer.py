from pdf_chatbot.models import ChatAnswer, SearchHit
from pdf_chatbot.store import VectorStore


SYSTEM_PROMPT = """You are a helpful PDF assistant.
Answer using only the provided document context.
Write naturally in your own words, like a real assistant, not like raw copied chunks.
If the answer is a direct fact, give the direct fact first.
If the question asks for summary, explain clearly with short bullets.
If the answer is not present in the context, say that the document does not provide enough information.
Do not invent facts.
Mention source page numbers briefly when useful.
"""


def answer_question(
    question: str,
    store: VectorStore,
    *,
    provider: str,
    openai_api_key: str | None,
    openai_chat_model: str,
    openai_embedding_model: str,
    gemini_api_key: str | None,
    gemini_model: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    ollama_embedding_model: str,
    top_k: int,
) -> ChatAnswer:
    hits = store.search(
        question,
        top_k=top_k,
        provider=provider,
        openai_api_key=openai_api_key,
        openai_embedding_model=openai_embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_embedding_model=ollama_embedding_model,
    )
    if not hits:
        return ChatAnswer("I could not find relevant information in the indexed document.", [])

    context = format_context(hits)
    provider = provider.lower().strip()

    if provider == "openai" and openai_api_key:
        return ChatAnswer(_answer_openai(question, context, api_key=openai_api_key, model=openai_chat_model), hits)

    if provider == "gemini" and gemini_api_key:
        return ChatAnswer(_answer_gemini(question, context, api_key=gemini_api_key, model=gemini_model), hits)

    if provider == "ollama":
        try:
            return ChatAnswer(_answer_ollama(question, context, base_url=ollama_base_url, model=ollama_chat_model), hits)
        except Exception as exc:
            return ChatAnswer(
                _extractive_fallback(question, hits)
                + f"\n\nLocal AI answer failed. Make sure Ollama is running and the model is pulled. Error: {exc}",
                hits,
            )

    return ChatAnswer(_extractive_fallback(question, hits), hits)


def _answer_openai(question: str, context: str, *, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.output_text.strip()


def _answer_gemini(question: str, context: str, *, api_key: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"{SYSTEM_PROMPT}\n\nDocument context:\n{context}\n\nQuestion: {question}"
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return "I could not generate an answer from the selected model."


def _answer_ollama(question: str, context: str, *, base_url: str, model: str) -> str:
    import requests

    prompt = f"Document context:\n{context}\n\nQuestion: {question}"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }
    response = requests.post(base_url.rstrip("/") + "/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for idx, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Source {idx}: {hit.source}, page {hit.page_number}, score {hit.score:.3f}]\n{hit.text}"
        )
    return "\n\n".join(blocks)


def _extractive_fallback(question: str, hits: list[SearchHit]) -> str:
    top = hits[0]
    return (
        "I found the most relevant text below, but the local AI model is not available yet. "
        "Start Ollama and rebuild the index to get a natural ChatGPT-style answer.\n\n"
        f"Source: {top.source}, page {top.page_number}\n\n{top.text[:1200]}"
    )
