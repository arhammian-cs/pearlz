"""The single, shared SupportPearlz evaluation engine.

Both `evaluation/run_eval.py` (CLI) and the Streamlit "Evaluation" tab call
into `run_evaluation()` (or the convenience wrapper `build_and_run_evaluation`
for standalone CLI use) so there is exactly one implementation of the
scoring logic -- no duplicated evaluation code between the two surfaces.

Every metric here is computed from real responses produced by the actual
RagChain against the actual persistent vector store; nothing is fabricated.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from src.chains.llm_factory import build_chat_model, build_embeddings
from src.chains.memory import ConversationMemory
from src.chains.rag_chain import RagChain
from src.chains.schemas import RagResponse
from src.config import Settings
from src.evaluation.schemas import (
    CaseResult,
    CategoryBreakdown,
    ConsistencyCheck,
    EvalSummary,
    TurnRecord,
)
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStoreManager
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[str], None]

TEST_QUESTIONS_FILENAME = "test_questions.json"
_RESULTS_FILE_PREFIX = "eval_"
_LATEST_RESULTS_FILENAME = "latest.json"


class EvaluationNotReadyError(RuntimeError):
    """Raised when evaluation is requested but no persistent index exists."""


def _default_test_questions_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "evaluation" / TEST_QUESTIONS_FILENAME


def _score_case(case: dict, final_response: RagResponse) -> dict:
    """Score a single case's final-turn response against its expectations."""
    expect_answered = case.get("expect_answered", True)
    expected_keywords = [k.lower() for k in case.get("expected_keywords", [])]
    forbidden_keywords = [k.lower() for k in case.get("forbidden_keywords", [])]
    expected_doc_types = case.get("expected_source_doc_types", [])

    answer_lower = final_response.answer.lower()

    forbidden_hit = next((kw for kw in forbidden_keywords if kw in answer_lower), None)
    keyword_hits = [kw for kw in expected_keywords if kw in answer_lower]
    keyword_ratio = (len(keyword_hits) / len(expected_keywords)) if expected_keywords else 1.0

    source_doc_types = {s.doc_type for s in final_response.sources}
    source_match = (not expected_doc_types) or bool(source_doc_types.intersection(expected_doc_types))

    answered_flag_ok = final_response.answered == expect_answered
    groundedness = (final_response.answered and bool(final_response.sources)) or not final_response.answered
    citation_accuracy = not final_response.citation_fabrication_detected

    failure_stage = None
    if forbidden_hit is not None:
        status = "fail"
        failure_stage = f"forbidden_content_present ('{forbidden_hit}')"
    elif not answered_flag_ok:
        status = "fail"
        failure_stage = "answered_flag_mismatch"
    elif expect_answered and not source_match:
        status = "partial"
        failure_stage = "source_mismatch"
    elif expect_answered and keyword_ratio < 1.0:
        status = "partial" if keyword_ratio > 0 else "fail"
        failure_stage = "keyword_mismatch"
    elif not citation_accuracy:
        status = "partial"
        failure_stage = "citation_fabrication_detected"
    else:
        status = "pass"

    return {
        "status": status,
        "failure_stage": failure_stage,
        "keyword_ratio": round(keyword_ratio, 2),
        "source_match": source_match,
        "groundedness": groundedness,
        "citation_accuracy": citation_accuracy,
    }


def _run_case(case: dict, rag_chain: RagChain, settings: Settings) -> CaseResult:
    memory = ConversationMemory(history_window_turns=settings.history_window_turns)
    final_response: RagResponse | None = None
    turn_records: list[TurnRecord] = []

    for turn_question in case["turns"]:
        response = rag_chain.answer(turn_question, memory)
        turn_records.append(
            TurnRecord(
                question=turn_question,
                rewritten_query=response.rewritten_query,
                answer=response.answer,
                confidence=response.confidence,
                answered=response.answered,
                sources=response.sources,
                retrieved_chunk_count=len(response.retrieved_chunks),
                max_retrieval_score=max((c.score for c in response.retrieved_chunks), default=0.0),
            )
        )
        final_response = response

    assert final_response is not None  # every case has at least one turn
    scoring = _score_case(case, final_response)

    return CaseResult(
        id=case["id"],
        category=case["category"],
        tags=case.get("tags", []),
        notes=case.get("notes"),
        turns=turn_records,
        expect_answered=case.get("expect_answered", True),
        final_confidence=final_response.confidence,
        final_answered=final_response.answered,
        latency_seconds=round(final_response.latency_seconds, 3),
        repeat_of=case.get("repeat_of"),
        **scoring,
    )


def _breakdown(label: str, cases: list[CaseResult]) -> CategoryBreakdown:
    total = len(cases)
    pass_count = sum(1 for c in cases if c.status == "pass")
    partial_count = sum(1 for c in cases if c.status == "partial")
    fail_count = sum(1 for c in cases if c.status == "fail")
    pass_rate = round(pass_count / total, 3) if total else 0.0
    return CategoryBreakdown(
        label=label,
        total=total,
        pass_count=pass_count,
        partial_count=partial_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
    )


def _consistency_checks(cases: list[CaseResult]) -> list[ConsistencyCheck]:
    by_id = {c.id: c for c in cases}
    checks = []
    for case in cases:
        if not case.repeat_of:
            continue
        base = by_id.get(case.repeat_of)
        if base is None:
            continue
        checks.append(
            ConsistencyCheck(
                base_id=base.id,
                repeat_id=case.id,
                same_answered_flag=base.final_answered == case.final_answered,
                same_confidence=base.final_confidence == case.final_confidence,
            )
        )
    return checks


def run_evaluation(
    settings: Settings,
    rag_chain: RagChain,
    vector_store_manager: VectorStoreManager,
    test_questions_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EvalSummary:
    """Run every test case through the given (already-configured) RagChain.

    `rag_chain` and `vector_store_manager` are accepted as parameters rather
    than constructed here so that callers -- in particular the Streamlit
    app -- can pass in the exact same cached instances used for normal chat,
    guaranteeing evaluation exercises the same LLM, embeddings, vector
    store, retriever, prompts, and citation-validation logic as the rest of
    the application.
    """

    def report(message: str) -> None:
        logger.info(message)
        if progress_callback is not None:
            progress_callback(message)

    if not vector_store_manager.exists():
        raise EvaluationNotReadyError(
            "No persistent vector store found. Build the knowledge base index first."
        )

    path = test_questions_path or _default_test_questions_path()
    cases_raw = json.loads(path.read_text(encoding="utf-8"))
    report(f"Loaded {len(cases_raw)} test cases from {path.name}.")

    results: list[CaseResult] = []
    for index, case in enumerate(cases_raw, start=1):
        report(f"Running case {index}/{len(cases_raw)}: {case['id']} ({case['category']})...")
        results.append(_run_case(case, rag_chain, settings))

    total = len(results)
    pass_count = sum(1 for r in results if r.status == "pass")
    partial_count = sum(1 for r in results if r.status == "partial")
    fail_count = sum(1 for r in results if r.status == "fail")
    refusal_count = sum(1 for r in results if not r.final_answered)
    hallucination_count = sum(1 for r in results if not r.citation_accuracy)

    all_scores = [t.max_retrieval_score for r in results for t in r.turns if t.retrieved_chunk_count > 0]
    all_chunk_counts = [t.retrieved_chunk_count for r in results for t in r.turns]

    confidence_distribution: dict[str, int] = {}
    for r in results:
        key = r.final_confidence.value
        confidence_distribution[key] = confidence_distribution.get(key, 0) + 1

    categories = sorted({r.category for r in results})
    category_breakdown = [_breakdown(category, [r for r in results if r.category == category]) for category in categories]

    summary = EvalSummary(
        run_at=datetime.now(timezone.utc).isoformat(),
        total_cases=total,
        pass_count=pass_count,
        partial_count=partial_count,
        fail_count=fail_count,
        pass_rate=round(pass_count / total, 3) if total else 0.0,
        overall_score=round(((pass_count + 0.5 * partial_count) / total) * 100, 1) if total else 0.0,
        refusal_rate=round(refusal_count / total, 3) if total else 0.0,
        hallucination_count=hallucination_count,
        groundedness_rate=round(sum(1 for r in results if r.groundedness) / total, 3) if total else 0.0,
        citation_accuracy_rate=round(sum(1 for r in results if r.citation_accuracy) / total, 3) if total else 0.0,
        avg_retrieved_chunk_count=round(sum(all_chunk_counts) / len(all_chunk_counts), 2) if all_chunk_counts else 0.0,
        avg_max_retrieval_score=round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0,
        confidence_distribution=confidence_distribution,
        category_breakdown=category_breakdown,
        answerable_breakdown=_breakdown("answerable", [r for r in results if r.expect_answered]),
        unanswerable_breakdown=_breakdown("unanswerable", [r for r in results if not r.expect_answered]),
        follow_up_breakdown=_breakdown("follow_up", [r for r in results if r.category == "follow_up"]),
        adversarial_breakdown=_breakdown("adversarial", [r for r in results if r.category == "adversarial"]),
        out_of_scope_breakdown=_breakdown("out_of_scope", [r for r in results if "out_of_scope" in r.tags]),
        consistency_checks=_consistency_checks(results),
        cases=results,
    )

    report(f"Evaluation complete: {pass_count}/{total} passed ({summary.overall_score}% overall score).")
    return summary


def build_and_run_evaluation(
    settings: Settings,
    api_key: str,
    progress_callback: ProgressCallback | None = None,
) -> EvalSummary:
    """Standalone entry point that constructs its own RAG components.

    Used by the CLI (`evaluation/run_eval.py`), which has no pre-existing
    Streamlit-cached RagChain to reuse. Uses the same factory functions
    (`build_chat_model`, `build_embeddings`) as the Streamlit app, so the
    resulting LLM/embedding configuration is identical even though the
    object instances are process-local to the CLI run.
    """
    embeddings = build_embeddings(settings, api_key)
    manager = VectorStoreManager(settings.vector_store_dir, settings.collection_name, embeddings)
    retriever = Retriever(manager, settings)
    chat_model = build_chat_model(settings, api_key)
    rag_chain = RagChain(chat_model, retriever, settings)
    return run_evaluation(settings, rag_chain, manager, progress_callback=progress_callback)


def save_summary(summary: EvalSummary, results_dir: Path) -> Path:
    """Persist a run to a timestamped file and refresh results/latest.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump_json(indent=2)

    timestamped_path = results_dir / f"{_RESULTS_FILE_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    timestamped_path.write_text(payload, encoding="utf-8")

    latest_path = results_dir / _LATEST_RESULTS_FILENAME
    latest_path.write_text(payload, encoding="utf-8")

    logger.info("Evaluation results written to %s", timestamped_path)
    return timestamped_path


def load_latest_summary(results_dir: Path) -> EvalSummary | None:
    """Load the most recent evaluation results from disk, if any exist.

    Performs no LLM calls -- safe to call unconditionally (e.g. on tab
    render) without triggering a real evaluation run.
    """
    latest_path = results_dir / _LATEST_RESULTS_FILENAME
    if not latest_path.exists():
        return None
    return EvalSummary.model_validate_json(latest_path.read_text(encoding="utf-8"))
