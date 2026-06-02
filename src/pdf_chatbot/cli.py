from pathlib import Path
import argparse

from pdf_chatbot.answer import answer_question
from pdf_chatbot.config import Settings
from pdf_chatbot.store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdf-chatbot")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("path", type=Path)

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)

    sub.add_parser("reset")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    store = VectorStore(settings.storage_dir)

    if args.command == "reset":
        store.reset()
        print("Index cleared")
        return

    if args.command == "ingest":
        count = store.build(
            args.path,
            provider=settings.provider,
            openai_api_key=settings.openai_api_key,
            openai_embedding_model=settings.openai_embedding_model,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )
        print(f"Indexed {count} chunks")
        return

    if args.command == "ask":
        result = answer_question(
            args.question,
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
            top_k=args.top_k or settings.top_k,
        )
        print(result.answer)
        if result.sources:
            print("\nSources:")
            for source in result.sources[:3]:
                print(f"- {source.source}, page {source.page_number}, score={source.score:.3f}")


if __name__ == "__main__":
    main()
