"""Answer graders: EM + lenient EM + token F1 + judge stub.

``cohen_kappa`` is the human-agreement hook: pass two parallel label lists
(e.g. judge scores vs human scores binarized) to measure the judge before
trusting ``judge_score`` in analysis.
"""

from __future__ import annotations

import re
import string

_WS_PUNCT = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = text.lower()
    text = _WS_PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, reference: str) -> int:
    """Strict EM: 1 on exact string equality, else 0."""
    return int(prediction == reference)


def lenient_exact_match(prediction: str, reference: str) -> int:
    """Lenient EM: 1 when normalized strings match, else 0."""
    return int(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 over normalized text (0.0 when either side empty)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    counts: dict[str, int] = {}
    for tok in ref_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    for tok in pred_tokens:
        if counts.get(tok, 0) > 0:
            counts[tok] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_stub(prediction: str, reference: str, judge_model: str = "") -> dict:
    """LLM-judge placeholder: returns no score until a judge is wired.

    Pilot only - the sweep leaves ``judge_score`` NULL and records which
    (absent) judge would have scored it, so later slices can backfill.
    """
    return {
        "judge_score": None,
        "judge_model": judge_model or "judge-stub/unwired",
        "note": "judge not wired in pilot; score left NULL for backfill",
        "graded": False,
    }


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa hook for judge-vs-human agreement (-1..1).

    Takes two parallel discrete label lists (binarize scores first).
    Returns 0.0 on degenerate inputs (no agreement possible to measure).
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    if len(classes) < 2:
        return 0.0
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum(
        (sum(1 for a in labels_a if a == c) / n) * (sum(1 for b in labels_b if b == c) / n)
        for c in classes
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def grade(prediction: str, reference: str) -> dict:
    """All cheap graders for one attempt (judge stays a stub)."""
    judged = judge_stub(prediction, reference)
    return {
        "exact_match": exact_match(prediction, reference),
        "lenient_em": lenient_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "judge_score": judged["judge_score"],
        "judge_model": judged["judge_model"],
    }
