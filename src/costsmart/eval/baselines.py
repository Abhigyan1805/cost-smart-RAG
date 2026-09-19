"""Routing baseline stubs for the oracle sweep.

Baselines pick a route per query without a learned router:
always-L0/C0/C1/C2, uniform-random (seeded), and the oracle (cheapest route
that answers correctly - computed post-hoc from attempt rows).
"""

from __future__ import annotations

import random

ROUTES = ("L0", "C0", "C1", "C2")


def always_l0(_query_id: str) -> str:
    return "L0"


def always_c0(_query_id: str) -> str:
    return "C0"


def always_c1(_query_id: str) -> str:
    return "C1"


def always_c2(_query_id: str) -> str:
    return "C2"


def random_route(query_id: str, seed: int = 0) -> str:
    """Seeded uniform-random route, stable per (seed, query_id)."""
    rng = random.Random(f"{seed}:{query_id}")
    return rng.choice(list(ROUTES))


def oracle_route(query_id: str, attempts: list[dict]) -> str:
    """Cheapest correct route for a query from recorded attempts.

    Correct = token_f1 >= 0.5 (pilot threshold); cost = cloud + amortized.
    Falls back to the cheapest attempted route when none is correct.
    """
    cands = [a for a in attempts if a.get("query_id") == query_id]
    if not cands:
        return "L0"
    correct = [a for a in cands if (a.get("token_f1") or 0) >= 0.5]
    pool = correct or cands
    best = min(pool, key=lambda a: (a.get("cloud_spend_usd") or 0) + (a.get("amortized_usd") or 0))
    return best.get("route_id", "L0")


BASELINES = {
    "always-L0": always_l0,
    "always-C0": always_c0,
    "always-C1": always_c1,
    "always-C2": always_c2,
    "random": random_route,
    "oracle": oracle_route,
}
