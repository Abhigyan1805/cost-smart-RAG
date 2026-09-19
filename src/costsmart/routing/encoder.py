"""v2 fine-tuned encoder router: interface stub (not implemented).

Intended design: fine-tune a small cross-encoder / sentence-transformer on
(query, route-label) pairs mined from v0/v1 disagreement + verifier outcomes,
so routing captures semantics that hand-written and featurized signals miss.

Training-data contract (for the future fine-tune job)::

    {"query": str, "label": 0 | 1, "source": "heuristic|featurized|verifier",
     "retrieval": {<same keys as the v1 retrieval feature-dict contract>}}

Until v2 lands, use :mod:`costsmart.routing.heuristic` (v0) or
:mod:`costsmart.routing.featurized` (v1). Every method here raises
:data:`NotImplementedError` with a pointer to the current alternative.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_V2_MSG = (
    "v2 encoder router is an interface stub: fine-tuning is not wired up yet. "
    "Use costsmart.routing.heuristic.route (v0) or "
    "costsmart.routing.featurized.FeaturizedRouter (v1) instead."
)


class EncoderRouter:
    """Fine-tuned encoder router interface (v2 stub)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def encode_query(self, query: str) -> Sequence[float]:
        """Embed *query* with the fine-tuned encoder."""
        raise NotImplementedError(_V2_MSG)

    def fit(
        self,
        queries: Sequence[str],
        labels: Sequence[int],
        retrievals: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> "EncoderRouter":
        """Fine-tune the encoder on (query, label) pairs."""
        raise NotImplementedError(_V2_MSG)

    def predict_proba_complex(self, query: str) -> float:
        """P(query needs cloud) in [0, 1]."""
        raise NotImplementedError(_V2_MSG)

    def decide(self, query: str) -> Any:
        """Route *query* to local/cloud."""
        raise NotImplementedError(_V2_MSG)
