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

        headers: dict[str, str] = {}
        token = os.environ.get(COLAB_TOKEN_ENV, "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout_s = float(kwargs.pop("timeout_s", 120.0))
        url = f"{endpoint.rstrip('/')}/api/generate"
        body = {"model": self.model_id, "prompt": prompt, "stream": False, **kwargs}
        started = time.monotonic()
        payload = _post_json(url, body, headers, timeout_s)
        latency = time.monotonic() - started
        return GenerateResult(
            text=payload.get("response", ""),
            tokens=int(payload.get("eval_count", 0) or 0),
            latency_s=latency,
            raw=payload,
        )


def _post_json(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
    """POST JSON and return the decoded object (httpx preferred, stdlib fallback).

    The worker env is stdlib-only (no pip); the live call must work there.
    """
    import json

    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]
    if httpx is not None:
        resp = httpx.post(url, json=body, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    import urllib.error
    import urllib.request

    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=raw,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            text = resp.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Colab endpoint HTTP {exc.code}: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Colab endpoint unreachable: {exc}") from exc
    if status >= 400:
        raise RuntimeError(f"Colab endpoint HTTP {status}")
    return json.loads(text)
