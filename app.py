"""SupportPearlz -- Streamlit frontend.

This module only coordinates the UI: session state, the API-key gate, and
the chat/knowledge-base/retrieval/evaluation tabs. All RAG logic lives in
`src/`.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from src.chains.llm_factory import build_chat_model, build_embeddings, validate_api_key
from src.chains.memory import ConversationMemory
from src.chains.rag_chain import RagChain
from src.chains.schemas import Confidence, RagResponse
from src.config import Settings, get_settings
from src.evaluation.evaluator import (
    EvaluationNotReadyError,
    load_latest_summary,
    run_evaluation,
    save_summary,
)
from src.evaluation.schemas import CaseResult, EvalSummary
from src.ingestion.build_index import build_knowledge_base_index
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStoreManager
from src.utils.logging_setup import configure_logging, get_logger

SETTINGS: Settings = get_settings()
configure_logging(SETTINGS.log_dir, SETTINGS.log_level)
logger = get_logger(__name__)

st.set_page_config(page_title="SupportPearlz", layout="wide", initial_sidebar_state="expanded")

CONFIDENCE_COLORS = {
    Confidence.HIGH: "#16a34a",
    Confidence.PARTIAL: "#d97706",
    Confidence.NONE: "#dc2626",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "api_key": None,
        "session_id": str(uuid.uuid4()),
        "chat_turns": [],
        "memory": ConversationMemory(history_window_turns=SETTINGS.history_window_turns),
        "eval_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.api_key = None
    st.session_state.chat_turns = []
    st.session_state.memory = ConversationMemory(history_window_turns=SETTINGS.history_window_turns)
    get_embeddings.clear()
    get_chat_model.clear()
    get_vector_manager.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Cached resource builders (keyed on the session-provided API key)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str):
    return build_embeddings(SETTINGS, api_key)


@st.cache_resource(show_spinner=False)
def get_chat_model(api_key: str):
    return build_chat_model(SETTINGS, api_key)


@st.cache_resource(show_spinner=False)
def get_vector_manager(api_key: str) -> VectorStoreManager:
    return VectorStoreManager(SETTINGS.vector_store_dir, SETTINGS.collection_name, get_embeddings(api_key))


def get_rag_chain(api_key: str) -> RagChain:
    manager = get_vector_manager(api_key)
    retriever = Retriever(manager, SETTINGS)
    return RagChain(get_chat_model(api_key), retriever, SETTINGS)


# ---------------------------------------------------------------------------
# API key gate
# ---------------------------------------------------------------------------

def render_api_key_gate() -> None:
    _, center, _ = st.columns([1, 1.3, 1])
    with center:
        st.markdown(
            "<h1 style='text-align:center;margin-bottom:0;'>SupportPearlz</h1>"
            "<p style='text-align:center;color:#6b7280;margin-top:4px;'>"
            "AI Customer Support Knowledge Agent</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### Enter your API Key")
        with st.form("api_key_form"):
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Continue", use_container_width=True)

        if submitted:
            if not api_key_input or not api_key_input.strip():
                st.error("Please enter an API key.")
            else:
                with st.spinner("Validating API key..."):
                    is_valid = validate_api_key(SETTINGS, api_key_input.strip())
                if is_valid:
                    st.session_state.api_key = api_key_input.strip()
                    st.session_state.authenticated = True
                    logger.info("Session authenticated (key redacted).")
                    st.rerun()
                else:
                    st.error("Invalid API key. Please check your key and try again.")

        st.markdown(
            "<p style='text-align:center;color:#9ca3af;font-size:0.85em;margin-top:16px;'>"
            "Your API key is used only for this application session.<br/>"
            "It is never written to disk or logged.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _rebuild_index(api_key: str) -> None:
    manager = get_vector_manager(api_key)
    embeddings = get_embeddings(api_key)
    progress_box = st.sidebar.empty()
    messages: list[str] = []

    def on_progress(message: str) -> None:
        messages.append(message)
        progress_box.info("\n\n".join(messages[-6:]))

    with st.spinner("Building knowledge base index..."):
        result = build_knowledge_base_index(
            SETTINGS, embeddings, progress_callback=on_progress, vector_store_manager=manager
        )
    progress_box.empty()
    st.sidebar.success(
        f"Index built: {result.chunk_count} chunks from {result.ingestion.documents_produced} documents "
        f"({result.ingestion.files_skipped} files skipped)."
    )
    if result.ingestion.skipped_files:
        with st.sidebar.expander(f"{result.ingestion.files_skipped} skipped file(s)"):
            for skipped in result.ingestion.skipped_files:
                st.write(f"- {skipped.path}: {skipped.reason}")


def render_sidebar(api_key: str) -> None:
    manager = get_vector_manager(api_key)
    metadata = manager.get_metadata()

    with st.sidebar:
        st.markdown("## SupportPearlz")
        st.caption("Grounded answers from Pearlz documentation")
        st.markdown("---")

        st.markdown("#### Knowledge Base")
        if metadata:
            st.success("Persistent knowledge base loaded")
            col_a, col_b = st.columns(2)
            col_a.metric("Documents", metadata.get("document_count", 0))
            col_b.metric("Chunks", metadata.get("chunk_count", 0))
            st.caption(f"Doc types: {', '.join(metadata.get('doc_types', []))}")
        else:
            st.warning("No knowledge base index found yet. Build it below.")

        st.markdown("#### Configuration")
        st.write(f"LLM model: `{SETTINGS.llm_model}`")
        st.write(f"Embedding model: `{SETTINGS.embedding_model}`")
        st.write(f"Retrieval strategy: `{SETTINGS.retrieval_strategy}` (k={SETTINGS.retrieval_k})")
        st.write(f"Relevance threshold: `{SETTINGS.relevance_threshold}`")

        st.markdown("---")
        st.markdown("#### Actions")
        if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
            _rebuild_index(api_key)
            st.rerun()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_turns = []
            st.session_state.memory.reset()
            st.rerun()
        if st.button("Logout / Clear API Key", use_container_width=True, type="primary"):
            _logout()


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def _render_confidence_badge(confidence: Confidence) -> str:
    color = CONFIDENCE_COLORS[confidence]
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:12px;font-size:0.75em;font-weight:600;letter-spacing:0.03em;'>"
        f"{confidence.value.upper()} CONFIDENCE</span>"
    )


def _render_sources(response: RagResponse) -> None:
    if not response.sources:
        return
    st.markdown("**Sources**")
    for source in response.sources:
        with st.expander(f"{source.document} — {source.location}"):
            st.caption(f"Document type: {source.doc_type}")


def render_chat_tab(api_key: str) -> None:
    st.markdown("### Chat")
    st.caption("Ask about warranty, shipping, refunds, installation, troubleshooting, pricing, and more.")

    for turn in st.session_state.chat_turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            response: RagResponse = turn["response"]
            st.write(response.answer)
            st.markdown(_render_confidence_badge(response.confidence), unsafe_allow_html=True)
            st.caption(f"Latency: {response.latency_seconds:.2f}s")
            _render_sources(response)

    question = st.chat_input("Type your question...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating a grounded answer..."):
                rag_chain = get_rag_chain(api_key)
                response = rag_chain.answer(question, st.session_state.memory)
            st.write(response.answer)
            st.markdown(_render_confidence_badge(response.confidence), unsafe_allow_html=True)
            st.caption(f"Latency: {response.latency_seconds:.2f}s")
            _render_sources(response)

        st.session_state.chat_turns.append(
            {"question": question, "response": response, "timestamp": time.time()}
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Knowledge Base tab
# ---------------------------------------------------------------------------

def render_knowledge_base_tab(api_key: str) -> None:
    st.markdown("### Knowledge Base Overview")
    manager = get_vector_manager(api_key)
    metadata = manager.get_metadata()

    kb_dir = SETTINGS.knowledge_base_dir
    files = sorted(p for p in kb_dir.rglob("*") if p.is_file()) if kb_dir.exists() else []
    file_rows = [{"file": p.name, "type": p.suffix.lstrip(".").upper(), "size_kb": round(p.stat().st_size / 1024, 1)} for p in files]

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### Source Documents")
        if file_rows:
            st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No documents found in data/knowledge_base/.")

    with col_right:
        st.markdown("#### Vector Store Status")
        if metadata:
            st.success("Persistent knowledge base loaded")
            st.write(f"Collection: `{SETTINGS.collection_name}`")
            st.write(f"Documents indexed: {metadata.get('document_count', 0)}")
            st.write(f"Chunks indexed: {metadata.get('chunk_count', 0)}")
            st.write(f"Built at: {metadata.get('built_at', 'unknown')}")

            doc_type_counts = pd.Series(metadata.get("doc_types", [])).value_counts().reset_index()
            if not doc_type_counts.empty:
                doc_type_counts.columns = ["doc_type", "count"]
                fig = px.bar(doc_type_counts, x="doc_type", y="count", title="Document Types in Index")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No knowledge base index found yet. Use the sidebar to build it.")


# ---------------------------------------------------------------------------
# Retrieval Insights tab
# ---------------------------------------------------------------------------

def render_retrieval_tab() -> None:
    st.markdown("### Retrieval Insights")
    st.caption("Details for the most recent question in this session.")

    if not st.session_state.chat_turns:
        st.info("Ask a question in the Chat tab to see retrieval details here.")
        return

    latest = st.session_state.chat_turns[-1]
    response: RagResponse = latest["response"]

    st.write(f"**Original question:** {latest['question']}")
    st.write(f"**Rewritten (standalone) query:** {response.rewritten_query or '(no rewriting needed)'}")
    st.write(f"**Relevance threshold:** {SETTINGS.relevance_threshold}")

    if response.retrieved_chunks:
        chunk_rows = [
            {
                "label": c.label,
                "document": c.document,
                "location": c.location,
                "doc_type": c.doc_type,
                "score": round(c.score, 3),
            }
            for c in response.retrieved_chunks
        ]
        df = pd.DataFrame(chunk_rows)
        fig = px.bar(
            df.sort_values("score"),
            x="score",
            y="label",
            orientation="h",
            color="doc_type",
            hover_data=["document", "location"],
            title="Retrieved Chunk Similarity Scores",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("View retrieved chunk text"):
            for c in response.retrieved_chunks:
                st.markdown(f"**[{c.label}] {c.document} ({c.location})** — score {c.score:.3f}")
                st.text(c.content[:800])
    else:
        st.info("No chunks were retrieved for the most recent question (refusal path).")


# ---------------------------------------------------------------------------
# Analytics (session-derived, real data only)
# ---------------------------------------------------------------------------

def render_analytics_tab() -> None:
    st.markdown("### Response Analytics")
    turns = st.session_state.chat_turns
    if not turns:
        st.info("No questions asked yet in this session.")
        return

    rows = [
        {
            "turn": i + 1,
            "confidence": t["response"].confidence.value,
            "answered": t["response"].answered,
            "latency": t["response"].latency_seconds,
            "source_count": len(t["response"].sources),
        }
        for i, t in enumerate(turns)
    ]
    df = pd.DataFrame(rows)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Confidence Distribution")
        confidence_counts = df["confidence"].value_counts().reset_index()
        confidence_counts.columns = ["confidence", "count"]
        fig = px.bar(confidence_counts, x="confidence", y="count", color="confidence", title="Confidence Levels This Session")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Answered vs. Refused")
        status_counts = df["answered"].map({True: "answered", False: "refused"}).value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", title="Answer Status This Session")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Response Latency")
    fig = px.line(df, x="turn", y="latency", markers=True, title="Latency per Turn (seconds)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Source Distribution")
    all_sources = [s.document for t in turns for s in t["response"].sources]
    if all_sources:
        source_counts = pd.Series(all_sources).value_counts().reset_index()
        source_counts.columns = ["document", "count"]
        fig = px.bar(source_counts, x="document", y="count", title="Documents Cited This Session")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sources have been cited yet this session.")


# ---------------------------------------------------------------------------
# Evaluation tab -- connected to the real evaluation engine
# (src/evaluation/evaluator.py), the same one used by evaluation/run_eval.py
# ---------------------------------------------------------------------------

def _run_full_evaluation(api_key: str) -> None:
    """Execute the real evaluation suite against the live RAG pipeline.

    Reuses the same cached RagChain / VectorStoreManager instances used by
    the Chat tab, so evaluation exercises the exact same LLM, embeddings,
    vector store, retriever, prompts, and citation-validation logic.
    """
    manager = get_vector_manager(api_key)
    rag_chain = get_rag_chain(api_key)

    progress_box = st.empty()
    messages: list[str] = []

    def on_progress(message: str) -> None:
        messages.append(message)
        progress_box.info("\n\n".join(messages[-8:]))

    try:
        with st.spinner("Running evaluation against the live RAG pipeline..."):
            summary = run_evaluation(SETTINGS, rag_chain, manager, progress_callback=on_progress)
    except EvaluationNotReadyError as exc:
        progress_box.empty()
        st.error(str(exc))
        return

    progress_box.empty()
    save_summary(summary, SETTINGS.eval_results_dir)
    st.session_state.eval_summary = summary
    st.success(
        f"Evaluation complete: {summary.pass_count}/{summary.total_cases} passed "
        f"(overall score {summary.overall_score:.1f}%)."
    )


def _breakdown_rows(summary: EvalSummary) -> pd.DataFrame:
    rows = [
        {
            "segment": f"category: {b.label}",
            "total": b.total,
            "pass": b.pass_count,
            "partial": b.partial_count,
            "fail": b.fail_count,
            "pass_rate": b.pass_rate,
        }
        for b in summary.category_breakdown
    ]
    for label, breakdown in [
        ("answerable questions", summary.answerable_breakdown),
        ("unanswerable / refusal", summary.unanswerable_breakdown),
        ("follow-up", summary.follow_up_breakdown),
        ("adversarial / injection", summary.adversarial_breakdown),
        ("out-of-scope", summary.out_of_scope_breakdown),
    ]:
        if breakdown.total == 0:
            continue
        rows.append(
            {
                "segment": label,
                "total": breakdown.total,
                "pass": breakdown.pass_count,
                "partial": breakdown.partial_count,
                "fail": breakdown.fail_count,
                "pass_rate": breakdown.pass_rate,
            }
        )
    return pd.DataFrame(rows)


def _case_summary_row(case: CaseResult) -> dict:
    final_turn = case.turns[-1]
    return {
        "id": case.id,
        "category": case.category,
        "question": final_turn.question,
        "expected_behavior": "answered" if case.expect_answered else "refused",
        "actual_behavior": "answered" if case.final_answered else "refused",
        "status": case.status,
        "confidence": case.final_confidence.value,
        "sources": ", ".join(f"{s.document} ({s.location})" for s in final_turn.sources) or "-",
        "citation_valid": case.citation_accuracy,
        "failure_reason": case.failure_stage or "-",
    }


def render_evaluation_tab(api_key: str) -> None:
    st.markdown("### Evaluation")
    st.caption(
        "Runs evaluation/test_questions.json through the live RAG pipeline used by the Chat tab. "
        "Every metric below is computed from real responses -- nothing here is hard-coded."
    )

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            _run_full_evaluation(api_key)
    with col_refresh:
        if st.button("Refresh / Load Last Results", use_container_width=True):
            loaded = load_latest_summary(SETTINGS.eval_results_dir)
            if loaded is None:
                st.warning("No saved evaluation results found on disk yet.")
            else:
                st.session_state.eval_summary = loaded

    if st.session_state.eval_summary is None:
        # Loading previously saved results from disk is not an LLM call, so
        # it is safe to do this once automatically for a better first view.
        st.session_state.eval_summary = load_latest_summary(SETTINGS.eval_results_dir)

    summary: EvalSummary | None = st.session_state.eval_summary
    if summary is None:
        st.info(
            "No evaluation results yet. Click **Run Evaluation** to test the live pipeline "
            "(requires a built knowledge base index), or run `python evaluation/run_eval.py` "
            "from the terminal."
        )
        return

    st.caption(f"Last run: {summary.run_at}")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Total Cases", summary.total_cases)
    col_b.metric("Passed", summary.pass_count)
    col_c.metric("Failed", summary.fail_count)
    col_d.metric("Overall Score", f"{summary.overall_score:.1f}%")
    col_e.metric("Citation Accuracy", f"{summary.citation_accuracy_rate * 100:.0f}%")

    col_f, col_g, col_h, col_i = st.columns(4)
    col_f.metric("Refusal Rate", f"{summary.refusal_rate * 100:.0f}%")
    col_g.metric("Groundedness", f"{summary.groundedness_rate * 100:.0f}%")
    col_h.metric("Avg Retrieved Chunks", f"{summary.avg_retrieved_chunk_count:.1f}")
    col_i.metric("Avg Top Retrieval Score", f"{summary.avg_max_retrieval_score:.2f}")

    st.markdown("#### Pass / Partial / Fail Distribution")
    status_counts = pd.DataFrame(
        [{"status": s, "count": sum(1 for c in summary.cases if c.status == s)} for s in ["pass", "partial", "fail"]]
    )
    fig = px.bar(status_counts, x="status", y="count", color="status", title="Case Outcomes")
    st.plotly_chart(fig, use_container_width=True)

    breakdown_df = _breakdown_rows(summary)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Confidence Distribution")
        conf_df = pd.DataFrame(
            [{"confidence": k, "count": v} for k, v in summary.confidence_distribution.items()]
        )
        if not conf_df.empty:
            fig = px.pie(conf_df, names="confidence", values="count", title="Final Confidence Levels")
            st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown("#### Category / Segment Performance")
        fig = px.bar(breakdown_df, x="segment", y="pass_rate", title="Pass Rate by Segment", range_y=[0, 1])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segment Performance Detail")
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    if summary.consistency_checks:
        st.markdown("#### Consistency Checks (Repeated / Rephrased Questions)")
        st.dataframe(
            pd.DataFrame([c.model_dump() for c in summary.consistency_checks]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Detailed Test Results")
    st.dataframe(pd.DataFrame([_case_summary_row(c) for c in summary.cases]), use_container_width=True, hide_index=True)

    failed_cases = [c for c in summary.cases if c.status in ("fail", "partial")]
    if failed_cases:
        st.markdown("#### Inspect Failed / Partial Cases")
        for case in failed_cases:
            final_turn = case.turns[-1]
            with st.expander(f"[{case.status.upper()}] {case.id} — {final_turn.question}"):
                st.write(f"**Category:** {case.category}")
                st.write(f"**Failure stage:** {case.failure_stage or 'n/a'}")
                st.write(f"**Expected:** {'answered' if case.expect_answered else 'refused'} — "
                         f"**Actual:** {'answered' if case.final_answered else 'refused'}")
                st.write(f"**Confidence:** {case.final_confidence.value}")
                st.write(f"**Keyword match ratio:** {case.keyword_ratio}")
                st.write(f"**Source match:** {case.source_match}")
                st.write(f"**Citation accuracy:** {case.citation_accuracy}")
                st.write("**Final answer:**")
                st.write(final_turn.answer)
                if final_turn.sources:
                    st.write("**Sources cited:**")
                    for s in final_turn.sources:
                        st.write(f"- {s.document} ({s.location})")
                if case.notes:
                    st.caption(f"Test intent: {case.notes}")
    else:
        st.success("No failed or partial cases in the last run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    if not st.session_state.authenticated:
        render_api_key_gate()
        return

    api_key = st.session_state.api_key
    render_sidebar(api_key)

    st.markdown(
        "<h1 style='margin-bottom:0;'>SupportPearlz</h1>"
        "<p style='color:#6b7280;margin-top:4px;'>AI Customer Support Knowledge Agent — "
        "grounded answers from Pearlz documentation.</p>",
        unsafe_allow_html=True,
    )

    tab_chat, tab_kb, tab_retrieval, tab_analytics, tab_eval = st.tabs(
        ["Chat", "Knowledge Base", "Retrieval Insights", "Analytics", "Evaluation"]
    )
    with tab_chat:
        render_chat_tab(api_key)
    with tab_kb:
        render_knowledge_base_tab(api_key)
    with tab_retrieval:
        render_retrieval_tab()
    with tab_analytics:
        render_analytics_tab()
    with tab_eval:
        render_evaluation_tab(api_key)


if __name__ == "__main__":
    main()
