"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateResult:
    """Result of a single generation call."""

    text: str
    tokens: int = 0
    logprobs: list[float] | None = None
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract LLM client. All model tiers implement this interface."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerateResult:
        """Generate a completion. Must return text/tokens/logprobs/latency."""
        raise NotImplementedError
