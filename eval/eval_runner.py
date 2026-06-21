"""
eval_runner.py — RAG pipeline evaluation harness.

Usage
-----
Run from the project root (so that `app.*` imports resolve):

    python eval/eval_runner.py                          # uses defaults from .env
    python eval/eval_runner.py --provider groq          # override LLM backend
    python eval/eval_runner.py --questions eval/test_questions.json
    python eval/eval_runner.py --output eval/results.json

The script:
  1. Loads question/ground-truth pairs from a JSON file.
  2. Passes each question through the full RAG pipeline (RAGChain.ask).
  3. Computes all metrics in eval/metrics.py for each question.
  4. Prints a per-question table and aggregate summary to stdout.
  5. Optionally saves results to a JSON file for further analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Bootstrap — make sure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.rag.chain import RAGChain  # noqa: E402  (after sys.path fix)
from eval.metrics import (           # noqa: E402
    answer_relevance,
    context_recall,
    exact_match,
    f1_token_overlap,
    faithfulness,
)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _context_text(chunks: List[Dict[str, Any]]) -> str:
    """Flatten retrieved chunks into a single string for metric computation."""
    return "\n\n".join(c["text"] for c in chunks)


def evaluate_single(
    chain: RAGChain,
    question: str,
    ground_truth: str,
    provider: str,
    model_name: str,
    top_k: int,
) -> Dict[str, Any]:
    """Run one question through the RAG pipeline and compute all metrics."""
    t0 = time.perf_counter()
    answer, chunks = chain.ask(
        query=question,
        provider=provider,
        model_name=model_name,
        top_k=top_k,
    )
    latency_s = round(time.perf_counter() - t0, 3)

    context = _context_text(chunks)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "num_chunks_retrieved": len(chunks),
        "latency_s": latency_s,
        "metrics": {
            "context_recall":    round(context_recall(ground_truth, context), 4),
            "answer_relevance":  round(answer_relevance(question, answer), 4),
            "faithfulness":      round(faithfulness(answer, context), 4),
            "exact_match":       round(exact_match(answer, ground_truth), 4),
            "f1_token_overlap":  round(f1_token_overlap(answer, ground_truth), 4),
        },
        "sources": [
            {"file": c["source_file"], "chunk_id": c["chunk_id"], "page": c["page"]}
            for c in chunks
        ],
    }


def run_evaluation(
    questions_path: str,
    provider: str,
    model_name: str | None,
    top_k: int,
    output_path: str | None,
) -> None:
    """Load questions, evaluate each one, print results, optionally save."""

    # --- load questions -------------------------------------------------------
    questions_file = Path(questions_path)
    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    with open(questions_file) as f:
        items: List[Dict[str, str]] = json.load(f)

    if not items:
        logger.error("Questions file is empty.")
        sys.exit(1)

    # Validate schema
    for i, item in enumerate(items):
        if "question" not in item or "ground_truth" not in item:
            logger.error(
                f"Item {i} is missing 'question' or 'ground_truth' keys."
            )
            sys.exit(1)

    # --- set up chain ---------------------------------------------------------
    logger.info(f"Initialising RAGChain (provider={provider}, model={model_name or 'default'}) ...")
    chain = RAGChain()

    # --- run ------------------------------------------------------------------
    results = []
    metric_keys = ["context_recall", "answer_relevance", "faithfulness", "exact_match", "f1_token_overlap"]

    print("\n" + "=" * 72)
    print(f"  RAG Evaluation  |  {len(items)} questions  |  provider={provider}")
    print("=" * 72)

    for idx, item in enumerate(items, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        logger.info(f"[{idx}/{len(items)}] {q[:80]}")

        try:
            result = evaluate_single(chain, q, gt, provider, model_name, top_k)
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            result = {
                "question": q,
                "ground_truth": gt,
                "answer": None,
                "error": str(e),
                "metrics": {k: None for k in metric_keys},
            }

        results.append(result)

        # Per-question output
        m = result["metrics"]
        print(f"\n[Q{idx}] {q}")
        print(f"  Answer    : {str(result.get('answer', 'ERROR'))[:120]}")
        print(f"  Latency   : {result.get('latency_s', '-')}s  |  Chunks: {result.get('num_chunks_retrieved', '-')}")
        print(
            f"  Metrics   : recall={m['context_recall']}  relevance={m['answer_relevance']}  "
            f"faithful={m['faithfulness']}  EM={m['exact_match']}  F1={m['f1_token_overlap']}"
        )

    # --- aggregate summary ----------------------------------------------------
    valid = [r for r in results if r.get("answer") is not None]
    print("\n" + "=" * 72)
    print("  AGGREGATE SUMMARY")
    print("=" * 72)
    if valid:
        for key in metric_keys:
            scores = [r["metrics"][key] for r in valid if r["metrics"].get(key) is not None]
            avg = round(sum(scores) / len(scores), 4) if scores else "n/a"
            print(f"  {key:<22} avg = {avg}")
    else:
        print("  No successful evaluations.")
    print("=" * 72 + "\n")

    # --- save output ----------------------------------------------------------
    if output_path:
        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "provider": provider,
            "model": model_name or "default",
            "top_k": top_k,
            "num_questions": len(items),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.success(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline against a question/ground-truth dataset."
    )
    parser.add_argument(
        "--questions",
        default="eval/test_questions.json",
        help="Path to the JSON file with question/ground_truth pairs (default: eval/test_questions.json)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "ollama"),
        choices=["ollama", "groq"],
        help="LLM backend to use (default: value of LLM_PROVIDER env var or 'ollama')",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g. llama3.1:8b or llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K_RETRIEVAL", 5)),
        help="Number of chunks to retrieve per question (default: TOP_K_RETRIEVAL env var or 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON (e.g. eval/results.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        questions_path=args.questions,
        provider=args.provider,
        model_name=args.model,
        top_k=args.top_k,
        output_path=args.output,
    )