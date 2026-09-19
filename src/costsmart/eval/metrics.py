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


# ---------------------------------------------------------------------------
# Headroom 2x2 (costheadroom-09): cheapest-local vs strongest-cloud.
# ---------------------------------------------------------------------------

#: Correctness threshold on the score field (same 0.5 gate as the oracle
#: baseline in :mod:`costsmart.eval.baselines`).
CORRECT_THRESHOLD = 0.5

#: Default headroom pair: cheapest local tier vs strongest cloud tier.
CHEAP_ROUTE = "L0"
STRONG_ROUTE = "C4"


def is_correct(attempt: dict, threshold: float = CORRECT_THRESHOLD,
               field: str = "token_f1") -> bool:
    """True when an attempt row scores at/above the correctness threshold."""
    try:
        return float(attempt.get(field) or 0.0) >= threshold
    except (TypeError, ValueError):
        return False


def _attempt_cost(attempt: dict) -> float:
    """Total cost of one attempt row (metered cloud + amortized local)."""
    try:
        cloud = float(attempt.get("cloud_spend_usd") or 0.0)
    except (TypeError, ValueError):
        cloud = 0.0
    try:
        amort = float(attempt.get("amortized_usd") or 0.0)
    except (TypeError, ValueError):
        amort = 0.0
    return cloud + amort


def build_contingency_table(
    attempts: list[dict],
    cheap_route: str = CHEAP_ROUTE,
    strong_route: str = STRONG_ROUTE,
    threshold: float = CORRECT_THRESHOLD,
    field: str = "token_f1",
) -> dict:
    """2x2 correctness contingency of cheap-local vs strong-cloud rows.

    Cells (counts + rates over queries with BOTH routes present):
    ``both_correct``, ``cheap_only`` (local right / cloud wrong),
    ``strong_only`` (local wrong / cloud right), ``neither``.
    Queries missing either route are counted in ``skipped`` (never silently
    dropped into a cell).
    """
    by_query: dict[str, dict] = {}
    for attempt in attempts:
        if attempt.get("route_id") in (cheap_route, strong_route):
            by_query.setdefault(attempt.get("query_id", ""), {})[attempt["route_id"]] = attempt
    cells = {"both_correct": 0, "cheap_only": 0, "strong_only": 0, "neither": 0}
    skipped = 0
    for routes in by_query.values():
        if cheap_route not in routes or strong_route not in routes:
            skipped += 1
            continue
        cheap_ok = is_correct(routes[cheap_route], threshold, field)
        strong_ok = is_correct(routes[strong_route], threshold, field)
        if cheap_ok and strong_ok:
            cells["both_correct"] += 1
        elif cheap_ok:
            cells["cheap_only"] += 1
        elif strong_ok:
            cells["strong_only"] += 1
        else:
            cells["neither"] += 1
    n = sum(cells.values())
    out: dict = {"n": n, "skipped": skipped, **cells,
                 "cheap_route": cheap_route, "strong_route": strong_route,
                 "threshold": threshold, "field": field}
    for key in cells:
        out[f"{key}_rate"] = (cells[key] / n) if n else 0.0
    return out


def routable_fraction(
    attempts: list[dict] | dict,
    cheap_route: str = CHEAP_ROUTE,
    strong_route: str = STRONG_ROUTE,
    threshold: float = CORRECT_THRESHOLD,
    field: str = "token_f1",
) -> dict:
    """Fraction of queries the cheap tier already answers correctly.

    ``attempts`` is either raw attempt rows or a prebuilt
    :func:`build_contingency_table` dict. The estimate is
    ``(both_correct + cheap_only) / n`` with ``n`` the paired-query count.
    """
    table = attempts if isinstance(attempts, dict) else build_contingency_table(
        attempts, cheap_route, strong_route, threshold, field)
    n = table["n"]
    k = table["both_correct"] + table["cheap_only"]
    return {"estimate": (k / n) if n else 0.0, "k": k, "n": n,
            "cheap_route": table["cheap_route"], "strong_route": table["strong_route"]}


def oracle_cheap_cost(attempts: list[dict], query_id: str,
                      threshold: float = CORRECT_THRESHOLD,
                      field: str = "token_f1") -> tuple[float, str]:
    """Cheapest correct route cost for one query (oracle headroom policy).

    Returns ``(cost, route_id)``. When no attempted route is correct, quality
    is zero whatever we pick, so the cheapest route wins (equal quality,
    minimal cost); the caller records this via ``route_id`` (never the
    strong route unless it is itself cheapest).
    """
    cands = [a for a in attempts if a.get("query_id") == query_id]
    if not cands:
        return 0.0, ""
    correct = [a for a in cands if is_correct(a, threshold, field)]
    pool = correct or cands
    best = min(pool, key=_attempt_cost)
    return _attempt_cost(best), best.get("route_id", "")


def max_savings_at_no_quality_loss(
    attempts: list[dict],
    strong_route: str = STRONG_ROUTE,
    threshold: float = CORRECT_THRESHOLD,
    field: str = "token_f1",
) -> dict:
    """Maximum cost saving routable at zero quality loss vs all-strong.

    Baseline: every query served by ``strong_route``. Oracle: every query
    served by its cheapest correct route (same correctness as the baseline
    wherever the baseline is correct). ``saving_fraction`` is the headroom
    upper bound a perfect router could capture; ``n_quality_loss_queries``
    counts queries where even the oracle is wrong while strong is right
    (zero here means the saving is truly lossless).
    """
    by_query: dict[str, list[dict]] = {}
    for attempt in attempts:
        by_query.setdefault(attempt.get("query_id", ""), []).append(attempt)
    strong_cost = oracle_cost = 0.0
    n = n_routed_cheap = n_no_correct = n_quality_loss = 0
    strong_rows: list[float] = []
    oracle_rows: list[float] = []
    for query_id, cands in by_query.items():
        strong = [a for a in cands if a.get("route_id") == strong_route]
        if not strong:
            continue
        n += 1
        s_cost = _attempt_cost(strong[0])
        o_cost, o_route = oracle_cheap_cost(attempts, query_id, threshold, field)
        strong_cost += s_cost
        oracle_cost += o_cost
        strong_rows.append(s_cost)
        oracle_rows.append(o_cost)
        if o_route != strong_route:
            n_routed_cheap += 1
        if not any(is_correct(a, threshold, field) for a in cands):
            n_no_correct += 1
        if is_correct(strong[0], threshold, field) and not any(
                is_correct(a, threshold, field) and _attempt_cost(a) <= s_cost
                for a in cands):
            n_quality_loss += 1
    saving = (strong_cost - oracle_cost) / strong_cost if strong_cost else 0.0
    return {"strong_route": strong_route, "n": n,
            "strong_cost": strong_cost, "oracle_cost": oracle_cost,
            "saving_fraction": saving,
            "n_routed_off_strong": n_routed_cheap,
            "n_no_correct_route": n_no_correct,
            "n_quality_loss_queries": n_quality_loss,
            "per_query_strong": strong_rows, "per_query_oracle": oracle_rows}
