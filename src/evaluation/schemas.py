"""Pydantic v2 schemas for the SupportPearlz evaluation engine.

Shared by both the CLI (evaluation/run_eval.py) and the Streamlit
"Evaluation" tab so both surfaces consume identical, real, typed results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.chains.schemas import Confidence, SourceInfo

CaseStatus = Literal["pass", "partial", "fail"]


class TurnRecord(BaseModel):
    """One executed turn (a case may have multiple turns for follow-ups)."""

    question: str
    rewritten_query: str | None = None
    answer: str
    confidence: Confidence
    answered: bool
    sources: list[SourceInfo] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    max_retrieval_score: float = 0.0


class CaseResult(BaseModel):
    """The scored outcome of a single evaluation test case."""

    id: str
    category: str
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    turns: list[TurnRecord]
    expect_answered: bool
    final_confidence: Confidence
    final_answered: bool
    latency_seconds: float
    status: CaseStatus
    failure_stage: str | None = None
    keyword_ratio: float
    source_match: bool
    groundedness: bool
    citation_accuracy: bool
    repeat_of: str | None = None


class ConsistencyCheck(BaseModel):
    base_id: str
    repeat_id: str
    same_answered_flag: bool
    same_confidence: bool


class CategoryBreakdown(BaseModel):
    """Pass/partial/fail rollup for a named slice of the case set."""

    label: str
    total: int
    pass_count: int
    partial_count: int
    fail_count: int
    pass_rate: float


class EvalSummary(BaseModel):
    """The complete, real, machine-readable evaluation run result."""

    run_at: str
    total_cases: int
    pass_count: int
    partial_count: int
    fail_count: int
    pass_rate: float
    overall_score: float
    refusal_rate: float
    hallucination_count: int
    groundedness_rate: float
    citation_accuracy_rate: float
    avg_retrieved_chunk_count: float
    avg_max_retrieval_score: float
    confidence_distribution: dict[str, int]
    category_breakdown: list[CategoryBreakdown]
    answerable_breakdown: CategoryBreakdown
    unanswerable_breakdown: CategoryBreakdown
    follow_up_breakdown: CategoryBreakdown
    adversarial_breakdown: CategoryBreakdown
    out_of_scope_breakdown: CategoryBreakdown
    consistency_checks: list[ConsistencyCheck]
    cases: list[CaseResult]
