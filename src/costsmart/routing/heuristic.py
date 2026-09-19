"""v0 heuristic router: hand-written rules, runs standalone.

No dependencies, no retrieval input, no model weights: given only the raw
query string it classifies the query as a simple factual lookup (route to the
small local model) or a complex multi-step query (route to the large cloud
model). Later router versions (v1+) may call this as a fast-path fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LOCAL = "local"  #: small quantized model (Ollama).
CLOUD = "cloud"  #: large cloud LLM.

#: Relative cost units used for demo accounting (cloud ~= 20x local).
COST = {LOCAL: 1.0, CLOUD: 20.0}

_COMPLEX_CUES = (
    "compare", "contrast", "versus", " vs ", "timeline", "step-by-step",
    "step by step", "why", "how", "explain", "analyze", "analyse",
    "summarise", "summarize", "pros and cons", "trade-off", "tradeoff",
    "multi", "several", "relationship between", "difference between",
    "evaluate", "critique", "design", "plan ", "strategy", "calculate",
)

_SIMPLE_CUES = ("who", "what", "when", "where", "which year", "how many")

_CLAUSE_SPLIT = re.compile(r"[;:]|\b(?: and | then | also | plus | after that )\b", re.IGNORECASE)


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of routing one query."""

    route: str  # LOCAL or CLOUD
    confidence: float  # 0..1, margin-based; suitable for later AUROC analysis
    reasons: list[str] = field(default_factory=list)
    rule_hits: list[str] = field(default_factory=list)
    estimated_cost: float = 1.0


def _complexity_score(query: str) -> tuple[float, list[str]]:
    q = query.strip()
    ql = q.lower()
    words = q.split()
    hits: list[str] = []
    score = 0.0

    for cue in _COMPLEX_CUES:
        if cue in ql:
            score += 1.5
            hits.append(f"complex-cue:{cue.strip()}")
    if len(words) > 25:
        score += 1.5
        hits.append("long-query")
    elif len(words) > 14:
        score += 0.75
        hits.append("medium-query")
    clauses = [c for c in _CLAUSE_SPLIT.split(q) if c.strip()]
    if len(clauses) > 2:
        score += 1.5
        hits.append("multi-clause")
    if q.count("?") > 1:
        score += 1.0
        hits.append("multi-question")
    if re.search(r"\b\d{4}\b.*\b\d{4}\b", q):
        score += 0.75
        hits.append("multi-date")
    if re.search(r"\b(first|then|finally|because|although|whereas)\b", ql):
        score += 0.5
        hits.append("reasoning-marker")

    for cue in _SIMPLE_CUES:
        if ql.startswith(cue) and len(words) <= 12 and len(clauses) == 1:
            score -= 1.0
            hits.append(f"simple-cue:{cue}")
            break
    return score, hits


def route(query: str, threshold: float = 1.5) -> RouteDecision:
    """Route *query* to ``LOCAL`` or ``CLOUD`` using hand-written rules."""
    score, hits = _complexity_score(query or "")
    is_cloud = score >= threshold
    dest = CLOUD if is_cloud else LOCAL
    margin = abs(score - threshold)
    confidence = min(0.5 + margin / 4.0, 0.99)
    reasons = [f"complexity-score={score:.2f} vs threshold={threshold:.2f}"]
    reasons += hits
    return RouteDecision(
        route=dest,
        confidence=round(confidence, 3),
        reasons=reasons,
        rule_hits=hits,
        estimated_cost=COST[dest],
    )
