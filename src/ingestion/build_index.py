"""Offline ingestion entry point: documents -> chunks -> persisted vector store.

This module is the *only* path that re-embeds the knowledge base. Querying
(src/chains/rag_chain.py) never triggers ingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import IngestionSummary, load_knowledge_base
from src.retrieval.vector_store import VectorStoreManager
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class BuildResult:
    ingestion: IngestionSummary
    chunk_count: int
    index_metadata: dict

    def as_dict(self) -> dict:
        return {
            "files_discovered": self.ingestion.files_discovered,
            "files_loaded": self.ingestion.files_loaded,
            "files_skipped": self.ingestion.files_skipped,
            "skipped_files": [asdict(s) for s in self.ingestion.skipped_files],
            "duplicate_count": self.ingestion.duplicate_count,
            "documents_produced": self.ingestion.documents_produced,
            "chunk_count": self.chunk_count,
            "index_metadata": self.index_metadata,
        }


def build_knowledge_base_index(
    settings: Settings,
    embeddings: Embeddings,
    progress_callback=None,
    vector_store_manager: VectorStoreManager | None = None,
) -> BuildResult:
    """Run the full offline ingestion pipeline and persist the vector store.

    If a `vector_store_manager` is supplied (e.g. a cached instance already
    held by the Streamlit app), it is reused so callers keep a single
    consistent in-memory reference to the store after rebuilding. Otherwise a
    fresh manager is constructed (e.g. for the standalone evaluation CLI).
    """

    def report(message: str) -> None:
        logger.info(message)
        if progress_callback is not None:
            progress_callback(message)

    report(f"Scanning knowledge base directory: {settings.knowledge_base_dir}")
    documents, ingestion_summary = load_knowledge_base(settings.knowledge_base_dir)
    report(
        f"Loaded {ingestion_summary.files_loaded}/{ingestion_summary.files_discovered} files "
        f"into {ingestion_summary.documents_produced} documents "
        f"({ingestion_summary.files_skipped} skipped, {ingestion_summary.duplicate_count} duplicates)."
    )

    report("Splitting documents into chunks...")
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    report(f"Produced {len(chunks)} chunks.")

    report("Embedding chunks and persisting vector store (this may take a moment)...")
    manager = vector_store_manager or VectorStoreManager(settings.vector_store_dir, settings.collection_name, embeddings)
    index_metadata = manager.rebuild(chunks)
    report("Vector store persisted successfully.")

    return BuildResult(ingestion=ingestion_summary, chunk_count=len(chunks), index_metadata=index_metadata)
