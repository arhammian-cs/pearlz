https://pearlz-3.streamlit.app/


# SupportPearlz

AI Customer Support Knowledge Agent for the fictional Pearlz Home Systems —
a Retrieval-Augmented Generation (RAG) app built with LangChain, Chroma, and
Streamlit.

## Requirements

- Python 3.10+
- An OpenAI API key (entered through the app UI at runtime)

## Installation

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

On first launch you'll see an API-key entry screen. Enter your OpenAI API
key (it is validated live, kept only in the Streamlit session for that run,
and never written to disk or logged).

## Building the knowledge base index

Sample documents already exist in `data/knowledge_base/` (PDF, DOCX,
Markdown, TXT, CSV). The vector index is **not** built automatically:

1. Log in with your API key.
2. In the sidebar, click **Build / Rebuild Knowledge Base**.
3. Subsequent app launches will reuse the persisted index in
   `data/vector_store/` without re-embedding, until you rebuild it again.

## Running the evaluation harness

Requires a persisted vector store (build it via the UI first). Both the CLI
and the Streamlit **Evaluation** tab call the same evaluation engine
(`src/evaluation/evaluator.py`), so they always produce identical results
for the same knowledge base and model configuration.

Via the UI: open the **Evaluation** tab and click **Run Evaluation** (uses
the session's API key; no LLM calls happen until you click it). Use
**Refresh / Load Last Results** to view a previous run without calling the
LLM again.

Via the CLI:

```bash
python evaluation/run_eval.py --api-key sk-...
# or, with OPENAI_API_KEY set in the environment / a local .env file:
python evaluation/run_eval.py
```

Results are written to `evaluation/results/` (one timestamped file per run,
plus `latest.json`) and are what the app's **Evaluation** tab displays.

## Project structure

```
app.py                     Streamlit UI (chat, knowledge base, retrieval, analytics, evaluation)
src/config.py               Validated settings (chunking, retrieval, models, logging)
src/ingestion/               Document loaders, chunking, offline index builder
src/retrieval/               Persistent Chroma vector store + retriever/relevance gate
src/chains/                  Prompts, Pydantic schemas, conversational memory, RAG chain
src/evaluation/               Shared evaluation engine + schemas (used by CLI and the UI)
src/utils/                    Centralized logging
data/knowledge_base/         Fictional Pearlz support documents
data/vector_store/           Persisted Chroma index (created after first build)
evaluation/                  Test question set + evaluation harness + results
```

## Configuration

All tunable parameters (chunk size/overlap, retrieval k, relevance
threshold, models, etc.) live in `src/config.py` and can be overridden via
environment variables prefixed `SUPPORTPEARLZ_` (see `.env.example`). The
API key itself is never read from `.env` by the Streamlit app — only the
evaluation CLI script falls back to `OPENAI_API_KEY` from the environment.
