"""Centralized, validated application configuration.

All tunable parameters live here so business logic never contains magic
numbers. The API key itself is intentionally excluded from this settings
object -- it is supplied at runtime from Streamlit session state (see
app.py) and never read from disk for the interactive application.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


class Settings(BaseSettings):
    """Validated, environment-overridable configuration for SupportPearlz."""

    model_config = SettingsConfigDict(
        env_prefix="SUPPORTPEARLZ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / embedding provider -----------------------------------
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    # --- Ingestion / chunking ----------------------------------------
    knowledge_base_dir: Path = Field(default=DEFAULT_KNOWLEDGE_BASE_DIR)
    chunk_size: int = Field(default=900, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Vector store --------------------------------------------------
    vector_store_dir: Path = Field(default=DEFAULT_VECTOR_STORE_DIR)
    collection_name: str = Field(default="supportpearlz_kb")

    # --- Retrieval -----------------------------------------------------
    retrieval_strategy: str = Field(default="similarity")  # "similarity" | "mmr"
    retrieval_k: int = Field(default=4, gt=0)
    retrieval_fetch_k: int = Field(default=10, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Context / generation -------------------------------------------
    max_context_characters: int = Field(default=8000, gt=0)
    history_window_turns: int = Field(default=4, ge=0)

    # --- Logging ---------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=DEFAULT_LOG_DIR)

    # --- Evaluation --------------------------------------------------------
    eval_results_dir: Path = Field(default=DEFAULT_EVAL_RESULTS_DIR)

    # Optional fallback API key for non-interactive contexts (e.g. the
    # evaluation CLI script). The Streamlit UI never reads this value.
    fallback_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")


def get_settings() -> Settings:
    """Return a freshly validated Settings instance."""
    return Settings()
