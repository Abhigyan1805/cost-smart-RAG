"""Dual-mode cost accounting.

Two columns are populated on every attempt row:

* ``cloud_spend_usd`` - metered cloud spend from per-token prices
  (tokens_in/out x model price). Zero for local/Ollama routes.
* ``amortized_usd`` - Colab GPU-seconds share: the fraction of a rented GPU
  hour attributable to this attempt, divided by the effective concurrency
  (the ``OLLAMA_NUM_PARALLEL``-equivalent ``concurrency`` field). This is
  the "what does local inference really cost" mode.

Pilot rule: the oracle sweep only *estimates* spend (stub executor, no API
calls). Real cloud calls arrive in a later slice; the accounting functions
below are already real.
"""

from __future__ import annotations

import os

# Per-1M-token prices in USD: model_version -> (input, output).
# Local quantized routes cost no metered cloud spend.
MODEL_PRICES_PER_1M = {
    "ollama-qwen2.5-3b": (0.0, 0.0),
    "ollama-llama3.1-8b": (0.0, 0.0),
    "colab-t4-local": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

# Colab-style GPU rental prices in USD/hour. Override via COLAB_GPU_PRICES
# ("T4=0.35,L4=0.60") or COLAB_GPU_USD_PER_HOUR (single default).
GPU_USD_PER_HOUR = {
    "T4": 0.35,
    "L4": 0.60,
    "A100-40GB": 1.60,
    "A100-80GB": 2.20,
}
DEFAULT_GPU_TYPE = "T4"

CONCURRENCY_ENV_VAR = "OLLAMA_NUM_PARALLEL"


def read_concurrency(default: int = 1) -> int:
    """OLLAMA_NUM_PARALLEL-equivalent effective concurrency (int >= 1)."""
    raw = os.environ.get(CONCURRENCY_ENV_VAR, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def gpu_hourly_rate(gpu_type: str = DEFAULT_GPU_TYPE) -> float:
    """USD/hour for a GPU type, honoring env overrides."""
    flat = os.environ.get("COLAB_GPU_USD_PER_HOUR")
    if flat:
        try:
            return max(0.0, float(flat))
        except ValueError:
            pass
    table = dict(GPU_USD_PER_HOUR)
    for chunk in os.environ.get("COLAB_GPU_PRICES", "").split(","):
        if "=" in chunk:
            name, price = chunk.split("=", 1)
            try:
                table[name.strip()] = max(0.0, float(price))
            except ValueError:
                continue
    return table.get(gpu_type, table[DEFAULT_GPU_TYPE])


def cloud_spend_usd(
    tokens_in: int,
    tokens_out: int,
    model_version: str,
    prices: dict | None = None,
) -> float:
    """Metered cloud spend for one attempt. Local routes price at 0."""
    table = prices if prices is not None else MODEL_PRICES_PER_1M
    price_in, price_out = table.get(model_version, (0.0, 0.0))
    return max(0, tokens_in) * price_in / 1_000_000 + max(0, tokens_out) * price_out / 1_000_000


def amortized_usd(
    gpu_seconds: float,
    gpu_type: str = DEFAULT_GPU_TYPE,
    concurrency: int | None = None,
) -> float:
    """Colab GPU-seconds share: hourly/3600 * seconds / concurrency."""
    if concurrency is None:
        concurrency = read_concurrency()
    concurrency = max(1, int(concurrency))
    return max(0.0, gpu_seconds) * gpu_hourly_rate(gpu_type) / 3600.0 / concurrency


def cost_record(
    tokens_in: int,
    tokens_out: int,
    model_version: str,
    gpu_seconds: float = 0.0,
    gpu_type: str = DEFAULT_GPU_TYPE,
    concurrency: int | None = None,
) -> dict:
    """Both accounting modes for one attempt (always populated)."""
    if concurrency is None:
        concurrency = read_concurrency()
    return {
        "tokens_in": max(0, tokens_in),
        "tokens_out": max(0, tokens_out),
        "gpu_seconds": max(0.0, gpu_seconds),
        "concurrency": max(1, int(concurrency)),
        "cloud_spend_usd": cloud_spend_usd(tokens_in, tokens_out, model_version),
        "amortized_usd": amortized_usd(gpu_seconds, gpu_type, concurrency),
    }
