"""v1 featurized router: LightGBM / logreg with a pre- vs post-retrieval ablation switch.

Retrieval feature-dict contract
-------------------------------
v1 accepts the retrieval feature dict **by name** from the retrieval slice and
must NOT import retrieval internals. The retrieval slice owns populating it;
this module only reads the keys below (all optional; missing keys fall back
to neutral defaults):

- ``top1_score``: float -- top retrieval hit score (higher = better match).
- ``score_margin``: float -- ``top1_score - top2_score`` (>= 0; large = clear winner).
- ``mean_topk_score``: float -- mean score over the top-k hits.
- ``num_hits``: int -- hits above the retrieval relevance threshold.
- ``topk``: int -- k that was requested (for normalising ``num_hits``).

Ablation switch (``mode``)
--------------------------
- ``"pre"``: query-text features only (routable *before* retrieval runs).
- ``"post"``: retrieval features only (route *after* retrieval).
- ``"both"``: query + retrieval features (default; expected best quality).

Backends: ``"builtin"`` (stdlib logistic regression, no dependencies),
``"sklearn"`` (LogisticRegression, lazy import) and ``"lightgbm"`` (lazy
import). Only ``"builtin"`` works without extra installs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .heuristic import CLOUD, COST, LOCAL, RouteDecision

MODES = ("pre", "post", "both")

#: Fixed feature order (also the column order for sklearn/lightgbm matrices).
PRE_FEATURES = (
    "qlen_tokens", "qlen_chars", "num_clauses", "qmark_count",
    "has_complex_cue", "has_simple_cue",
)
POST_FEATURES = (
    "top1_score", "score_margin", "mean_topk_score", "hit_rate",
)
FEATURE_ORDER: dict[str, tuple[str, ...]] = {
    "pre": PRE_FEATURES,
    "post": POST_FEATURES,
    "both": PRE_FEATURES + POST_FEATURES,
}

_COMPLEX_CUES = ("why", "how", "compare", "versus", " vs ", "explain",
                 "analy", "summar", "evaluat", "timeline", "difference")
_SIMPLE_CUES = ("who", "what", "when", "where", "which year", "how many")


def extract_features(
    query: str,
    retrieval: Mapping[str, Any] | None = None,
    mode: str = "both",
) -> dict[str, float]:
    """Build the numeric feature dict for *query* (+ optional *retrieval*).

    Missing retrieval keys fall back to neutral defaults so ``mode="both"``
    still works pre-retrieval (those features read as "no evidence").
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    q = query or ""
    ql = q.lower()
    words = q.split()
    feats: dict[str, float] = {}
    if mode in ("pre", "both"):
        feats = {
            "qlen_tokens": float(len(words)),
            "qlen_chars": float(len(q)),
            "num_clauses": float(q.count(";") + q.count(":") + q.count(" and ") + 1),
            "qmark_count": float(q.count("?")),
            "has_complex_cue": float(any(c in ql for c in _COMPLEX_CUES)),
            "has_simple_cue": float(any(ql.startswith(c) for c in _SIMPLE_CUES)),
        }
    if mode in ("post", "both"):
        r = retrieval or {}
        topk = max(int(r.get("topk", 5) or 5), 1)
        num_hits = max(int(r.get("num_hits", 0) or 0), 0)
        feats.update({
            "top1_score": float(r.get("top1_score", 0.0) or 0.0),
            "score_margin": max(float(r.get("score_margin", 0.0) or 0.0), 0.0),
            "mean_topk_score": float(r.get("mean_topk_score", 0.0) or 0.0),
            "hit_rate": min(num_hits / topk, 1.0),
        })
    order = FEATURE_ORDER[mode]
    return {k: feats[k] for k in order}


def to_vector(feats: Mapping[str, float], mode: str = "both") -> list[float]:
    """Order a feature dict into a model input vector."""
    return [float(feats[k]) for k in FEATURE_ORDER[mode]]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 500.0), -500.0)))


@dataclass
class FeaturizedRouter:
    """Binary route classifier: P(complex query) -> cloud, else local.

    ``label`` convention for :meth:`fit`: 1 = needs cloud, 0 = local suffices.
    """

    mode: str = "both"
    backend: str = "builtin"
    threshold: float = 0.5
    learning_rate: float = 0.1
    epochs: int = 200
    weights: list[float] = field(default_factory=list, repr=False)
    bias: float = 0.0
    n_features_: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._model: Any = None  # sklearn / lightgbm estimator when used

    # -- training ------------------------------------------------------
    def fit(
        self,
        queries: Sequence[str],
        labels: Sequence[int],
        retrievals: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> "FeaturizedRouter":
        """Train on parallel *queries* / *labels* (+ optional retrieval dicts)."""
        if len(queries) != len(labels):
            raise ValueError("queries and labels must be parallel")
        if retrievals is not None and len(retrievals) != len(queries):
            raise ValueError("retrievals must be parallel to queries")
        retrievals = retrievals or [None] * len(queries)
        X = [to_vector(extract_features(q, r, self.mode), self.mode)
             for q, r in zip(queries, retrievals)]
        if self.backend == "builtin":
            self._fit_builtin(X, list(labels))
        elif self.backend == "sklearn":
            from sklearn.linear_model import LogisticRegression  # lazy, optional dep

            self._model = LogisticRegression(max_iter=1000)
            self._model.fit(X, list(labels))
        elif self.backend == "lightgbm":
            import lightgbm as lgb  # lazy, optional dep

            data = lgb.Dataset(X, label=list(labels))
            self._model = lgb.train({"objective": "binary", "verbose": -1}, data)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        return self

    def _fit_builtin(self, X: list[list[float]], y: list[int]) -> None:
        n = len(X[0]) if X else 0
        self.n_features_ = n
        # Standardise with training mean/var for stable gradient descent.
        self._mean = [sum(col) / len(col) for col in zip(*X)] if X else [0.0] * n
        self._var = [
            sum((v - m) ** 2 for v in col) / len(col) + 1e-9
            for col, m in zip(zip(*X), self._mean)
        ] if X else [1.0] * n
        Xs = [[(v - m) / math.sqrt(vr) for v, m, vr in zip(row, self._mean, self._var)]
              for row in X]
        w = [0.0] * n
        b = 0.0
        for _ in range(max(self.epochs, 1)):
            for row, target in zip(Xs, y):
                p = _sigmoid(sum(wi * xi for wi, xi in zip(w, row)) + b)
                err = p - target
                w = [wi - self.learning_rate * err * xi for wi, xi in zip(w, row)]
                b -= self.learning_rate * err
        self.weights, self.bias = w, b

    # -- inference -----------------------------------------------------
    def predict_proba_complex(
        self, query: str, retrieval: Mapping[str, Any] | None = None
    ) -> float:
        """P(query needs cloud) in [0, 1]; usable directly for AUROC analysis."""
        feats = extract_features(query, retrieval, self.mode)
        vec = to_vector(feats, self.mode)
        if self.backend == "builtin":
            if not self.weights:
                raise RuntimeError("FeaturizedRouter is not fitted yet")
            xs = [(v - m) / math.sqrt(vr)
                  for v, m, vr in zip(vec, self._mean, self._var)]
            return _sigmoid(sum(w * x for w, x in zip(self.weights, xs)) + self.bias)
        if self._model is None:
            raise RuntimeError("FeaturizedRouter is not fitted yet")
        if self.backend == "sklearn":
            return float(self._model.predict_proba([vec])[0][1])
        return float(self._model.predict([vec])[0])

    def decide(
        self, query: str, retrieval: Mapping[str, Any] | None = None
    ) -> RouteDecision:
        """Route *query* (plus optional retrieval features) to local/cloud."""
        proba = self.predict_proba_complex(query, retrieval)
        dest = CLOUD if proba >= self.threshold else LOCAL
        return RouteDecision(
            route=dest,
            confidence=round(abs(proba - 0.5) * 2 * 0.49 + 0.5, 3),
            reasons=[f"v1-{self.backend}/{self.mode}: P(cloud)={proba:.3f}"],
            rule_hits=[],
            estimated_cost=COST[dest],
        )
