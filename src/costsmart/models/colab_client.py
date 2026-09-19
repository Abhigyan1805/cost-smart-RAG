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
        raise NotImplementedError(
            "ColabClient is a stub: attach the captain-provided Colab session first. "
            f"Set {COLAB_ENDPOINT_ENV} (+ optional {COLAB_TOKEN_ENV}) to the live "
            "Colab model endpoint, then implement generate() here. "
            "Local execution must go through this interface - "
            "never assume a local Ollama daemon."
        )
