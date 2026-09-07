"""The composed SupportPearlz RAG pipeline.

Pipeline stages (see module docstring in each collaborator for detail):
question + history -> condense -> retrieve -> relevance gate ->
labelled context -> grounded prompt -> LLM structured output ->
citation validation -> RagResponse.

Each stage is delegated to a small, focused collaborator (Retriever,
ConversationMemory, prompts, schemas) rather than inlined into one giant
function.
"""

from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel

from src.chains.memory import ConversationMemory
from src.chains.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, REFUSAL_ANSWER_TEMPLATE
from src.chains.schemas import Confidence, LLMStructuredAnswer, RagResponse, RetrievedChunk, SourceInfo
from src.config import Settings
from src.retrieval.retriever import Retriever
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _build_labelled_context(chunks: list[RetrievedChunk], max_characters: int) -> str:
    """Render retrieved chunks into a labelled context block, budget-capped."""
    blocks = []
    used_characters = 0
    for chunk in chunks:
        header = f"[{chunk.label}] Source: {chunk.document} ({chunk.location})"
        block = f"{header}\n{chunk.content}"
        if used_characters + len(block) > max_characters and blocks:
            break
        blocks.append(block)
        used_characters += len(block)
    return "\n\n".join(blocks)


def _validate_citations(
    structured_answer: LLMStructuredAnswer,
    chunks: list[RetrievedChunk],
) -> tuple[list[SourceInfo], bool]:
    """Keep only citation labels that were actually present in the context.

    Returns the validated, de-duplicated source list plus a flag indicating
    whether any fabricated (hallucinated) citation labels were removed.
    """
    chunk_by_label = {chunk.label: chunk for chunk in chunks}
    validated: list[SourceInfo] = []
    seen: set[tuple[str, str]] = set()
    had_fabricated = False

    for label in structured_answer.sources:
        chunk = chunk_by_label.get(label)
        if chunk is None:
            had_fabricated = True
            logger.warning("Discarding fabricated citation label from model output: %r", label)
            continue
        key = (chunk.document, chunk.location)
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            SourceInfo(
                label=chunk.label,
                document=chunk.document,
                doc_type=chunk.doc_type,
                location=chunk.location,
            )
        )
    return validated, had_fabricated


def _refusal_response(reason: str, rewritten_query: str | None, latency: float) -> RagResponse:
    return RagResponse(
        answer=REFUSAL_ANSWER_TEMPLATE,
        sources=[],
        confidence=Confidence.NONE,
        answered=False,
        rewritten_query=rewritten_query,
        refusal_reason=reason,
        retrieved_chunks=[],
        latency_seconds=latency,
        escalation_required=True,
    )


class RagChain:
    """Orchestrates condensation, retrieval, generation, and validation."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        retriever: Retriever,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._condense_chain = CONDENSE_PROMPT | chat_model
        self._answer_chain = ANSWER_PROMPT | chat_model.with_structured_output(LLMStructuredAnswer)

    def _condense_question(self, question: str, memory: ConversationMemory) -> str:
        if memory.is_empty():
            return question
        history_text = memory.format_for_condensation()
        try:
            result = self._condense_chain.invoke({"history": history_text, "question": question})
            rewritten = (result.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Query condensation failed, falling back to original question: %s", exc)
            return question
        if not rewritten:
            return question
        logger.info("Query condensation: original=%r rewritten=%r", question, rewritten)
        return rewritten

    def answer(self, question: str, memory: ConversationMemory) -> RagResponse:
        start_time = time.monotonic()

        rewritten_query = self._condense_question(question, memory)
        retrieval = self._retriever.retrieve(rewritten_query)

        if not retrieval.is_relevant:
            latency = time.monotonic() - start_time
            reason = (
                "No knowledge base content met the relevance threshold "
                f"({retrieval.max_score:.2f} < {self._settings.relevance_threshold:.2f})."
                if retrieval.chunks
                else "No knowledge base content was retrieved for this query."
            )
            response = _refusal_response(reason, rewritten_query, latency)
            memory.add_turn(question, response.answer)
            return response

        context = _build_labelled_context(retrieval.chunks, self._settings.max_context_characters)

        try:
            structured_answer = self._answer_chain.invoke({"context": context, "question": rewritten_query})
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed: %s", exc, exc_info=True)
            latency = time.monotonic() - start_time
            response = _refusal_response(
                "The support assistant could not generate a response due to a temporary error.",
                rewritten_query,
                latency,
            )
            memory.add_turn(question, response.answer)
            return response

        validated_sources, had_fabricated = _validate_citations(structured_answer, retrieval.chunks)
        confidence = structured_answer.confidence
        answered = structured_answer.answered

        if had_fabricated:
            # A fabricated citation means the grounding claim is untrustworthy;
            # degrade confidence rather than silently presenting it as valid.
            confidence = Confidence.PARTIAL if confidence == Confidence.HIGH else Confidence.NONE
            if not validated_sources:
                answered = False

        if answered and not validated_sources:
            # An answer with zero real citations cannot be considered grounded.
            confidence = Confidence.NONE
            answered = False

        latency = time.monotonic() - start_time
        response = RagResponse(
            answer=structured_answer.answer if answered else REFUSAL_ANSWER_TEMPLATE,
            sources=validated_sources,
            confidence=confidence,
            answered=answered,
            rewritten_query=rewritten_query,
            refusal_reason=None if answered else "The retrieved documentation did not sufficiently support an answer.",
            retrieved_chunks=retrieval.chunks,
            latency_seconds=latency,
            escalation_required=not answered,
            citation_fabrication_detected=had_fabricated,
        )
        memory.add_turn(question, response.answer)
        logger.info(
            "Answered query: original=%r rewritten=%r confidence=%s answered=%s sources=%d latency=%.2fs",
            question,
            rewritten_query,
            response.confidence,
            response.answered,
            len(response.sources),
            response.latency_seconds,
        )
        return response
