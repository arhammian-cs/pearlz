"""Configurable retrieval with a deterministic relevance gate.

Retrieval strategy (similarity vs. MMR) is configuration-driven. Scores are
always produced so the relevance gate and UI can reason about retrieval
quality -- for MMR, scores are approximated from a similarity pass over the
same candidate pool used for MMR selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.chains.schemas import RetrievedChunk
from src.retrieval.vector_store import VectorStoreManager
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    max_score: float
    is_relevant: bool


class Retriever:
    """Thin retrieval layer over a VectorStoreManager with a relevance gate."""

    def __init__(self, vector_store_manager: VectorStoreManager, settings: Settings) -> None:
        self._manager = vector_store_manager
        self._settings = settings

    def retrieve(self, query: str) -> RetrievalResult:
        store = self._manager.store
        k = self._settings.retrieval_k

        try:
            if self._settings.retrieval_strategy == "mmr":
                chunks = self._retrieve_mmr(store, query, k)
            else:
                chunks = self._retrieve_similarity(store, query, k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for query %r: %s", query, exc, exc_info=True)
            return RetrievalResult(chunks=[], max_score=0.0, is_relevant=False)

        max_score = max((c.score for c in chunks), default=0.0)
        is_relevant = max_score >= self._settings.relevance_threshold and len(chunks) > 0
        logger.info(
            "Retrieved %d chunks for query %r (max_score=%.3f, relevant=%s)",
            len(chunks),
            query,
            max_score,
            is_relevant,
        )
        return RetrievalResult(chunks=chunks, max_score=max_score, is_relevant=is_relevant)

    def _retrieve_similarity(self, store, query: str, k: int) -> list[RetrievedChunk]:
        results = store.similarity_search_with_relevance_scores(query, k=k)
        return self._to_retrieved_chunks(results)

    def _retrieve_mmr(self, store, query: str, k: int) -> list[RetrievedChunk]:
        fetch_k = max(self._settings.retrieval_fetch_k, k)
        candidate_pool = store.similarity_search_with_relevance_scores(query, k=fetch_k)
        score_by_content = {doc.page_content: score for doc, score in candidate_pool}

        selected_docs = store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=self._settings.mmr_lambda,
        )
        results = [(doc, score_by_content.get(doc.page_content, 0.0)) for doc in selected_docs]
        return self._to_retrieved_chunks(results)

    @staticmethod
    def _to_retrieved_chunks(results) -> list[RetrievedChunk]:
        chunks = []
        for index, (doc, score) in enumerate(results, start=1):
            chunks.append(
                RetrievedChunk(
                    label=f"S{index}",
                    content=doc.page_content,
                    score=max(0.0, min(1.0, float(score))),
                    document=doc.metadata.get("source", "unknown"),
                    doc_type=doc.metadata.get("doc_type", "general"),
                    location=doc.metadata.get("location", ""),
                )
            )
        return chunks
