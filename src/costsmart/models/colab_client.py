"""Colab client stub: handoff point for the captain-provided Colab session.

Local model tiers run in Google Colab, not on this machine. The captain
connects the Colab MCP on request; this module is the single place where that
live session attaches. Until then every method raises ``NotImplementedError``.

HANDOFF (for the captain / future slice):
  1. Start the Colab notebook that serves the local-tier model with an
     Ollama-compatible (or plain HTTP) endpoint.
  2. Expose the endpoint URL + credentials ONLY via environment:
       COSTSMART_COLAB_ENDPOINT  - base URL of the Colab-served model server
       COSTSMART_COLAB_TOKEN     - bearer token, if the session requires one
     Optionally mirror them into ``config/models.yaml`` local-tier entries.
  3. Implement ``ColabClient.generate()`` by delegating to ``OllamaClient``
     pointed at ``COSTSMART_COLAB_ENDPOINT`` (or a bespoke HTTP call), and
     delete this stub's ``NotImplementedError``.

Rules: no import-time connection, no hardcoded tokens/URLs, one injection
point (the two env vars above). See ``OllamaClient`` for the HTTP pattern.
"""

from __future__ import annotations

import os
from typing import Any

from costsmart.models.base import GenerateResult, LLMClient

COLAB_ENDPOINT_ENV = "COSTSMART_COLAB_ENDPOINT"
COLAB_TOKEN_ENV = "COSTSMART_COLAB_TOKEN"


def get_colab_endpoint() -> str:
    """Return the injected Colab endpoint URL (empty when not attached)."""
    return os.environ.get(COLAB_ENDPOINT_ENV, "")


class ColabClient(LLMClient):
    """Stub client for Colab-served local tiers. Not yet attached."""

    def __init__(self, name: str, model_id: str) -> None:
        super().__init__(name)
        self.model_id = model_id

    def generate(self, prompt: str, **kwargs: Any) -> GenerateResult:
        endpoint = get_colab_endpoint()
        if not endpoint:
            raise RuntimeError(
                "ColabClient has no live session: the captain connects the Colab "
                f"session on request; set {COLAB_ENDPOINT_ENV} (+ optional "
                f"{COLAB_TOKEN_ENV}) to the Colab model endpoint first. "
                "See docs/colab-handoff.md. "
                "Local execution must go through this interface - "
                "never assume a local Ollama daemon."
            )
        # Same Ollama-compatible /api/generate contract as OllamaClient,
        # plus the optional bearer token (OllamaClient takes no headers).
        import time

        import httpx

        headers: dict[str, str] = {}
        token = os.environ.get(COLAB_TOKEN_ENV, "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout_s = float(kwargs.pop("timeout_s", 120.0))
        started = time.monotonic()
        resp = httpx.post(
            f"{endpoint.rstrip('/')}/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False, **kwargs},
            headers=headers,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        latency = time.monotonic() - started
        return GenerateResult(
            text=payload.get("response", ""),
            tokens=int(payload.get("eval_count", 0) or 0),
            latency_s=latency,
            raw=payload,
        )
