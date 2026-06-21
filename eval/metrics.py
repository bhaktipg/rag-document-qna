"""
metrics.py — Lightweight RAG evaluation metrics (no LLM-as-judge required).

All metrics operate on plain text and return a float in [0, 1].

Metrics implemented
-------------------
- context_recall        : fraction of ground-truth tokens found in retrieved context
- answer_relevance      : token-overlap between the answer and the question
- faithfulness          : fraction of answer tokens that appear in the retrieved context
- exact_match           : 1.0 if answer == ground truth (normalised), else 0.0
- f1_token_overlap      : harmonic mean of precision & recall at token level vs ground truth
"""

from __future__ import annotations

import re
import string
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lower-case and split on whitespace/punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _token_set(text: str) -> set:
    return set(_tokenise(text))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def context_recall(ground_truth: str, context: str) -> float:
    """Fraction of ground-truth tokens present in the retrieved context.

    A score of 1.0 means every word in the expected answer was surfaced
    by the retriever. A score of 0.0 means none were.
    """
    gt_tokens = _token_set(ground_truth)
    if not gt_tokens:
        return 0.0
    ctx_tokens = _token_set(context)
    return len(gt_tokens & ctx_tokens) / len(gt_tokens)


def answer_relevance(question: str, answer: str) -> float:
    """Token overlap between the answer and the question.

    A rough proxy for whether the answer addresses the question at all.
    """
    q_tokens = _token_set(question)
    a_tokens = _token_set(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens | a_tokens)


def faithfulness(answer: str, context: str) -> float:
    """Fraction of answer tokens that appear in the retrieved context.

    Measures whether the model hallucinated content not in the sources.
    A score of 1.0 means the answer only uses words from the context.
    """
    a_tokens = _token_set(answer)
    if not a_tokens:
        return 0.0
    ctx_tokens = _token_set(context)
    return len(a_tokens & ctx_tokens) / len(a_tokens)


def exact_match(answer: str, ground_truth: str) -> float:
    """1.0 if the normalised answer exactly matches the ground truth."""
    def _normalise(text: str) -> str:
        return " ".join(_tokenise(text))

    return 1.0 if _normalise(answer) == _normalise(ground_truth) else 0.0


def f1_token_overlap(answer: str, ground_truth: str) -> float:
    """Harmonic mean of token-level precision and recall vs ground truth.

    This is the standard SQuAD-style F1 metric.
    """
    a_tokens = _tokenise(answer)
    gt_tokens = _tokenise(ground_truth)
    if not a_tokens or not gt_tokens:
        return 0.0

    common = set(a_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(a_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)