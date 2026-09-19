"""Thin Ollama-compatible client interface.

Local model tiers run in Google Colab, not on this machine. This client only
talks to an Ollama-compatible HTTP endpoint whose base URL is injected via
config/env - it never assumes a local Ollama daemon.

Injection point: ``base_url`` constructor arg, or ``OLLAMA_BASE_URL`` env var.
No connection is made at import time.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from costsmart.models.base import GenerateResult, LLMClient


class OllamaClient(LLMClient):
    """Minimal Ollama-compatible ``/api/generate`` client."""

    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(name)
        self.model_id = model_id
        # Single documented injection point: explicit arg wins, else env.
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, **kwargs: Any) -> GenerateResult:
        if not self.base_url:
            raise RuntimeError(
                "OllamaClient has no endpoint: pass base_url or set OLLAMA_BASE_URL. "
                "No local daemon is assumed."
            )
        started = time.monotonic()
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False, **kwargs},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        latency = time.monotonic() - started
        text = payload.get("response", "")
        # Ollama reports prompt/eval counts; fall back to 0 when absent.
        tokens = int(payload.get("eval_count", 0) or 0)
        return GenerateResult(text=text, tokens=tokens, latency_s=latency, raw=payload)
