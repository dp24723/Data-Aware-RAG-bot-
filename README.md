# Data-Aware RAG

PDF chatbot with document upload, text extraction, retrieval, source display, and AI-generated answers.

## Local setup

```bash
conda create -n pdfchat python=3.11 -y
conda activate pdfchat
pip install -e ".[dev]"
cp .env.example .env
streamlit run app.py
```

Set `GEMINI_API_KEY` in `.env` for AI answers.

## Streamlit Cloud deployment

Use `app.py` as the entry file. Add these in Streamlit Cloud secrets:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K = "6"
```

Do not commit `.env` or `.streamlit/secrets.toml`.

## Tests

```bash
pytest -q
```
# Data-Aware-RAG-bot-
