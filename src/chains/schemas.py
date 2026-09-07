"""Pydantic v2 schemas shared across the RAG pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    HIGH = "high"
    PARTIAL = "partial"
    NONE = "none"


class LLMStructuredAnswer(BaseModel):
    """Raw structured output requested directly from the chat model.

    The model may only cite the bracket labels (e.g. "S1") that were present
    in the context it was given -- those labels are validated against the
    actual retrieved chunks downstream, never trusted blindly.
    """

    answer: str = Field(description="The support answer grounded in the provided context.")
    sources: list[str] = Field(
        default_factory=list,
        description="Citation labels (e.g. 'S1') actually used to support the answer.",
    )
    confidence: Confidence
    answered: bool = Field(description="Whether the question was actually answered.")


class RetrievedChunk(BaseModel):
    """A single retrieved chunk, kept for UI/debug/evaluation purposes."""

    label: str
    content: str
    score: float
    document: str
    doc_type: str
    location: str


class SourceInfo(BaseModel):
    """A validated, de-duplicated citation shown to the end user."""

    label: str
    document: str
    doc_type: str
    location: str


class RagResponse(BaseModel):
    """Final structured response returned by the RAG pipeline to the UI."""

    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    confidence: Confidence
    answered: bool
    rewritten_query: str | None = None
    refusal_reason: str | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_seconds: float = 0.0
    escalation_required: bool = False
    citation_fabrication_detected: bool = False
