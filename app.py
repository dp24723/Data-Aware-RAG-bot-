from pathlib import Path
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from pdf_chatbot.answer import answer_question
from pdf_chatbot.config import Settings
from pdf_chatbot.store import VectorStore


st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot")
st.caption("Upload PDFs, build an index, and ask questions in natural language.")


def load_secret_into_env(key: str) -> None:
    """Load Streamlit Cloud secrets into env without breaking local runs."""
    if os.getenv(key):
        return

    try:
        value = st.secrets.get(key, None)
    except Exception:
        value = None

    if value is not None and str(value).strip():
        os.environ[key] = str(value)


for secret_key in [
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "PDF_CHAT_STORAGE",
    "TOP_K",
]:
    load_secret_into_env(secret_key)


settings = Settings.from_env()
store = VectorStore(settings.storage_dir)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if settings.provider == "gemini":
        if settings.gemini_api_key:
            st.info(f"Using Gemini model: {settings.gemini_model}")
        else:
            st.warning("GEMINI_API_KEY is missing. Add it in Streamlit secrets or .env.")
    elif settings.provider == "ollama":
        st.info(f"Using local Ollama model: {settings.ollama_chat_model}")
    elif settings.provider == "openai":
        if settings.openai_api_key:
            st.info(f"Using OpenAI model: {settings.openai_chat_model}")
        else:
            st.warning("LLM_PROVIDER is openai, but OPENAI_API_KEY is missing.")
    else:
        st.warning(f"Unknown provider: {settings.provider}")

    col1, col2 = st.columns(2)

    with col1:
        build_clicked = st.button("Build index", use_container_width=True)

    with col2:
        reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        store.reset()
        st.session_state.messages = []
        st.success("Index cleared.")

    if build_clicked:
        if not uploaded_files:
            st.error("Upload at least one file first.")
        else:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp)

                    for item in uploaded_files:
                        target = folder / item.name
                        target.write_bytes(item.getbuffer())

                    with st.spinner("Reading documents and building index..."):
                        count = store.build(
                            folder,
                            provider=settings.provider,
                            openai_api_key=settings.openai_api_key,
                            openai_embedding_model=settings.openai_embedding_model,
                            ollama_base_url=settings.ollama_base_url,
                            ollama_embedding_model=settings.ollama_embedding_model,
                        )

                st.success(f"Indexed {count} chunks.")

            except Exception as exc:
                st.error("I could not build the index for these documents.")
                st.caption(str(exc))

    st.divider()
    st.write("Provider:", settings.provider)
    st.write("Index ready:", "Yes" if store.exists() else "No")
    st.write("Top K:", settings.top_k)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask something about your PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources = []

    if not store.exists():
        answer = "Please upload at least one document and click **Build index** first."

    else:
        try:
            with st.spinner("Thinking..."):
                result = answer_question(
                    question,
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
                    top_k=settings.top_k,
                )

            answer = result.answer
            sources = result.sources

        except Exception as exc:
            answer = (
                "I could not generate an AI answer right now. "
                "Please check the Gemini API key, model name, or quota."
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant", avatar="🤖"):
                st.error(answer)
                st.caption(str(exc))

            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for idx, source in enumerate(sources, start=1):
                    st.markdown(
                        f"**{idx}. {source.source}, page {source.page_number}** "
                        f"· score `{source.score:.3f}`"
                    )
                    st.write(source.text)