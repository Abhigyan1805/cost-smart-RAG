"""Model registry: build LLMClients from config/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from costsmart.models.base import LLMClient

DEFAULT_MODELS_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"


def load_model_configs(path: str | Path = DEFAULT_MODELS_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("models", []))


def create_client(entry: dict[str, Any]) -> LLMClient:
    """Instantiate the right client type from one models.yaml entry."""
    provider = entry.get("provider", "cloud")
    name = entry["name"]
    model_id = entry["model_id"]
    if provider == "colab":
        from costsmart.models.colab_client import ColabClient

        return ColabClient(name=name, model_id=model_id)
    # Lazy imports: keep registry importable in stdlib-only envs (httpx may
    # be absent until a live call actually needs it).
    from costsmart.models.ollama_client import OllamaClient

    if provider == "ollama":
        return OllamaClient(name=name, model_id=model_id)
    # Cloud tiers: concrete cloud client lands in a later slice; reuse the
    # thin Ollama-compatible shim until then so routing code has a client.
    return OllamaClient(name=name, model_id=model_id)


def get_client(name: str, path: str | Path = DEFAULT_MODELS_PATH) -> LLMClient:
    for entry in load_model_configs(path):
        if entry.get("name") == name:
            return create_client(entry)
    raise KeyError(f"Unknown model {name!r} in {path}")
