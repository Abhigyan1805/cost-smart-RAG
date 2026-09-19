"""Routing metrics stubs: accuracy, regret, escalation precision/recall.

Each takes parallel lists over the eval set so the sweep can aggregate
attempt rows (grouped by query) into router-level numbers.
"""

from __future__ import annotations


def routing_accuracy(predicted_routes: list[str], oracle_routes: list[str]) -> float:
    """Fraction of queries where the router picked the oracle route."""
    if not predicted_routes or len(predicted_routes) != len(oracle_routes):
        return 0.0
    return sum(1 for p, o in zip(predicted_routes, oracle_routes) if p == o) / len(oracle_routes)


def regret(
    chosen_costs: list[float],
    oracle_costs: list[float],
    chosen_scores: list[float] | None = None,
    oracle_scores: list[float] | None = None,
) -> dict:
    """Cost regret (and optional quality gap) vs the oracle routing.

    Returns totals + mean; quality gap entries are None when scores absent.
    """
    if not chosen_costs or len(chosen_costs) != len(oracle_costs):
        return {"total_regret": 0.0, "mean_regret": 0.0, "mean_quality_gap": None}
    gaps = [c - o for c, o in zip(chosen_costs, oracle_costs)]
    out: dict = {
        "total_regret": sum(gaps),
        "mean_regret": sum(gaps) / len(gaps),
        "mean_quality_gap": None,
    }
    if chosen_scores is not None and oracle_scores is not None and len(chosen_scores) == len(gaps):
        qgaps = [o - c for c, o in zip(chosen_scores, oracle_scores)]
        out["mean_quality_gap"] = sum(qgaps) / len(qgaps)
    return out


def escalation_precision_recall(
    escalated: list[bool], should_have_escalated: list[bool]
) -> dict:
    """P/R of the escalate-to-bigger-model decision (stubs -> real formula)."""
    if not escalated or len(escalated) != len(should_have_escalated):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for e, s in zip(escalated, should_have_escalated) if e and s)
    fp = sum(1 for e, s in zip(escalated, should_have_escalated) if e and not s)
    fn = sum(1 for e, s in zip(escalated, should_have_escalated) if not e and s)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
