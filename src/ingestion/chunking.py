"""Configurable, metadata-preserving chunking for ingested documents."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Row-level documents (e.g. pricing/SKU rows) are already atomic units.
# Splitting them further could separate a SKU from its price, so they are
# never re-chunked regardless of configured chunk size.
_NO_SPLIT_DOC_TYPES = {"pricing_guide"}

_SECTION_AWARE_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into retrieval-sized chunks, preserving all metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SECTION_AWARE_SEPARATORS,
    )

    chunks: list[Document] = []
    for document in documents:
        if document.metadata.get("doc_type") in _NO_SPLIT_DOC_TYPES:
            chunks.append(document)
            continue

        split_docs = splitter.split_documents([document])
        for index, split_doc in enumerate(split_docs):
            split_doc.metadata = {**document.metadata, "chunk_index": index}
        chunks.extend(split_docs)

    logger.info(
        "Chunking complete: %d documents -> %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
