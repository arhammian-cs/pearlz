"""Command-line evaluation harness for the SupportPearlz RAG pipeline.

This is a thin CLI wrapper: all scoring/aggregation logic lives in
`src/evaluation/evaluator.py`, the same engine used by the Streamlit
"Evaluation" tab, so both surfaces always produce identical results for the
same knowledge base and model configuration.

Loads evaluation/test_questions.json, runs each case (including multi-turn
follow-up cases) through the real RagChain against the persistent vector
store, and writes machine-readable results to evaluation/results/.

Requires a persisted vector store (build it first via the Streamlit
"Build / Rebuild Knowledge Base" control) and an OpenAI API key available
as the OPENAI_API_KEY environment variable or passed via --api-key.

Usage:
    python evaluation/run_eval.py --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.evaluation.evaluator import (  # noqa: E402
    EvaluationNotReadyError,
    build_and_run_evaluation,
    save_summary,
)
from src.utils.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SupportPearlz evaluation harness.")
    parser.add_argument("--api-key", dest="api_key", default=None, help="OpenAI API key (overrides OPENAI_API_KEY env var).")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_dir, settings.log_level)

    api_key = args.api_key or settings.fallback_api_key
    if not api_key:
        raise SystemExit("An API key is required: pass --api-key or set OPENAI_API_KEY.")

    try:
        summary = build_and_run_evaluation(settings, api_key, progress_callback=print)
    except EvaluationNotReadyError as exc:
        raise SystemExit(str(exc)) from exc

    save_summary(summary, settings.eval_results_dir)
    print(json.dumps(summary.model_dump(exclude={"cases"}), indent=2, default=str))


if __name__ == "__main__":
    main()
