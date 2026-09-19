"""Stable labels from live-repeat draws (costfinal-10).

The sweep exposed small-model label noise: L0 flip rate 0.88 across 3 live
repeats at requested temperature 0 (paraphrase-level sampling variation -
same facts, different wording). A single draw per (query, route) pair is
therefore a noisy correctness label, so every live local-tier pair is
re-run :data:`N_REPEATS` times and graded by majority.

Tie rule (documented per the task spec): a pair is stable-correct only on a
strict majority (> half) of correct repeat draws; **ties count as
incorrect**. With 3 complete repeats a binary tie is impossible - the rule
bites only for incomplete pairs (missing draws), which are conservatively
labelled incorrect and flagged, never silently dropped.

Correctness per draw is the same gate the headroom analysis uses:
``token_f1 >= 0.5`` (:data:`costsmart.eval.metrics.CORRECT_THRESHOLD`).
"""

from __future__ import annotations

from . import graders
from .metrics import CORRECT_THRESHOLD
from ..telemetry.schema import N_REPEATS


def majority_correct(votes: list[bool], n_expected: int = N_REPEATS) -> dict:
    """Majority verdict over per-draw correctness votes.

    ``stable_correct`` requires a strict majority of the *expected* draws
    (quota = ``n_expected // 2 + 1``, i.e. >= 2 of 3): anything short of
    quota - an even split, a shortfall, or a losing vote - **counts as
    incorrect** per the spec tie rule. A missing draw can only hurt, never
    help, except when the observed correct votes already meet quota (2-0
    with one draw missing is unoverturnable and counts as correct).

    ``tie`` marks labels decided by the conservative rule rather than a
    full majority: incomplete pairs, or an exact even split among observed
    votes. Complete odd triples never tie.
    """
    votes = [bool(v) for v in votes]
    n_correct = sum(votes)
    n_total = len(votes)
    complete = n_total >= n_expected
    quota = n_expected // 2 + 1
    stable = n_correct >= quota
    tie = (not complete) or (n_total > 0 and n_correct * 2 == n_total)
    return {
        "stable_correct": stable,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_expected": n_expected,
        "quota": quota,
        "complete": complete,
        "tie": tie,
    }


def grade_repeats(predictions: list[str], reference: str) -> list[dict]:
    """Deterministic grades + correctness flag for each repeat prediction."""
    out = []
    for prediction in predictions:
        graded = graders.grade(prediction, reference)
        out.append({
            "prediction": prediction,
            "exact_match": graded["exact_match"],
            "lenient_em": graded["lenient_em"],
            "token_f1": graded["token_f1"],
            "correct": bool(graded["token_f1"] >= CORRECT_THRESHOLD),
        })
    return out


def stable_label_for_pair(
    repeat_rows: list[dict],
    base_row: dict | None = None,
    n_expected: int = N_REPEATS,
) -> dict:
    """Stable label + representative row for one (query, route) pair.

    ``repeat_rows`` are telemetry repeat rows (graded, with ``prediction`` /
    ``token_f1`` / ``reference``). The representative is the first repeat on
    the majority side, so ``is_correct(representative) == stable_correct``
    holds by construction for complete pairs and the whole existing
    headroom pipeline (metrics + stats + make_plots) runs unchanged on
    representative rows. Incomplete/tied pairs fall back to the base
    single-run row as representative, labelled incorrect per the tie rule.
    """
    ordered = sorted(repeat_rows, key=lambda r: r.get("repeat_idx", 0))
    votes = [float(r.get("token_f1") or 0.0) >= CORRECT_THRESHOLD for r in ordered]
    verdict = majority_correct(votes, n_expected)
    majority_side = True if verdict["stable_correct"] else None
    representative = None
    if verdict["complete"]:
        # First repeat on the majority side (stable-correct => a correct
        # draw; stable-incorrect => an incorrect draw).
        want = verdict["stable_correct"]
        for row, vote in zip(ordered, votes):
            if vote == want:
                representative = row
                break
        majority_side = verdict["stable_correct"]
    if representative is None:
        # Tie or incomplete: fall back to the base single-run row (may be
        # None when the caller only has repeats - then use the first repeat
        # and let stable_correct=False carry the conservative label).
        representative = base_row if base_row is not None else (ordered[0] if ordered else None)
    return {
        "query_id": (ordered[0].get("query_id") if ordered
                     else (base_row or {}).get("query_id", "")),
        "route_id": (ordered[0].get("route_id") if ordered
                     else (base_row or {}).get("route_id", "")),
        "stable_correct": verdict["stable_correct"],
        "majority_side": majority_side,
        "n_correct": verdict["n_correct"],
        "n_total": verdict["n_total"],
        "n_expected": n_expected,
        "complete": verdict["complete"],
        "tie": verdict["tie"],
        "representative": representative,
    }


def flip_stats(per_pair_predictions: list[list[str]]) -> dict:
    """String-level and label-level flip statistics over repeat draws.

    ``string_flip`` mirrors the l0_stability.py definition (repeat
    prediction strings not all identical). ``label_flip`` is what the
    oracle consumes (correctness votes not unanimous). Distribution counts
    pairs by number of distinct prediction strings and by vote split.
    """
    n = len(per_pair_predictions)
    string_flipped = sum(1 for reps in per_pair_predictions if len(set(reps)) > 1)
    distinct_hist: dict[int, int] = {}
    for reps in per_pair_predictions:
        d = len(set(reps))
        distinct_hist[d] = distinct_hist.get(d, 0) + 1
    return {
        "n_pairs": n,
        "string_flipped": string_flipped,
        "string_flip_rate": (string_flipped / n) if n else 0.0,
        "distinct_predictions_hist": distinct_hist,
    }


def label_flip_stats(vote_lists: list[list[bool]]) -> dict:
    """Correctness-level flip statistics (the noise the oracle sees)."""
    n = len(vote_lists)
    flipped = sum(1 for votes in vote_lists if len(set(bool(v) for v in votes)) > 1)
    split_hist: dict[str, int] = {}
    for votes in vote_lists:
        k = sum(1 for v in votes if v)
        key = f"{k}-{len(votes) - k}"
        split_hist[key] = split_hist.get(key, 0) + 1
    unanimous_correct = sum(1 for votes in vote_lists if votes and all(votes))
    unanimous_incorrect = sum(1 for votes in vote_lists if votes and not any(votes))
    return {
        "n_pairs": n,
        "label_flipped": flipped,
        "label_flip_rate": (flipped / n) if n else 0.0,
        "vote_split_hist": split_hist,
        "unanimous_correct": unanimous_correct,
        "unanimous_incorrect": unanimous_incorrect,
    }
