"""Factory functions that build LangChain LLM/embedding clients from a
runtime-supplied API key (never from disk) plus static Settings."""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings


def build_chat_model(settings: Settings, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )


def build_embeddings(settings: Settings, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )


def validate_api_key(settings: Settings, api_key: str) -> bool:
    """Perform a cheap, real call to confirm the API key is valid."""
    if not api_key or not api_key.strip():
        return False
    try:
        embeddings = build_embeddings(settings, api_key)
        embeddings.embed_query("supportpearlz key validation ping")
        return True
    except Exception:  # noqa: BLE001 - any failure means "not valid/usable"
        return False
