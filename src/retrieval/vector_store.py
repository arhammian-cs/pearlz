"""Persistent Chroma vector store management.

Ingestion (build/rebuild) and querying are deliberately kept as separate
entry points: `load_or_build` never re-embeds an existing, valid index, and
`rebuild` is the only path that re-embeds the full knowledge base.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_INDEX_METADATA_FILENAME = "index_meta.json"


class VectorStoreManager:
    """Wraps a persistent Chroma collection with load/build/rebuild semantics."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._store: Chroma | None = None

    @property
    def _meta_path(self) -> Path:
        return self._persist_dir / _INDEX_METADATA_FILENAME

    def exists(self) -> bool:
        """Whether a persisted index already exists on disk."""
        return self._meta_path.exists()

    def _open(self) -> Chroma:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def load_or_build(self) -> "VectorStoreManager":
        """Load the existing persistent index if one exists.

        This deliberately never embeds anything itself -- ingestion only
        happens through the explicit `rebuild()` path (triggered by the
        "Build / Rebuild Knowledge Base" UI action or the CLI). If no index
        exists yet, an empty collection is opened so querying degrades
        gracefully to the relevance-gate refusal path instead of crashing.
        """
        if self.exists():
            logger.info("Loading existing persistent vector store from %s", self._persist_dir)
        else:
            logger.warning(
                "No persistent vector store found at %s; the knowledge base must be built first.",
                self._persist_dir,
            )
        self._store = self._open()
        return self

    def rebuild(self, chunks: list[Document]) -> dict:
        """Re-embed all chunks from scratch and persist the new collection.

        Drops and recreates the Chroma *collection* rather than deleting the
        persist directory on disk -- removing the directory while a client
        holds open file handles is unreliable on Windows.
        """
        logger.info("Rebuilding vector store with %d chunks.", len(chunks))
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._store = None

        client = chromadb.PersistentClient(path=str(self._persist_dir))
        try:
            client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            batch_size = 100
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                self._store.add_documents(batch)
                logger.info("Embedded batch %d-%d of %d", start, start + len(batch), len(chunks))

        doc_types = sorted({c.metadata.get("doc_type", "general") for c in chunks})
        sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
        metadata = {
            "chunk_count": len(chunks),
            "document_count": len(sources),
            "doc_types": doc_types,
            "sources": sources,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        self._meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Vector store persisted to %s", self._persist_dir)
        return metadata

    def get_metadata(self) -> dict | None:
        if not self._meta_path.exists():
            return None
        return json.loads(self._meta_path.read_text(encoding="utf-8"))

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self.load_or_build()
        return self._store
