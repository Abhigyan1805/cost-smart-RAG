"""Agent actions: the loop's discrete action set."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    """One step the agent loop can take."""

    ACCEPT = "ACCEPT"  #: answer verified; return it.
    RETRIEVE = "RETRIEVE"  #: fetch more evidence, then re-answer.
    ESCALATE = "ESCALATE"  #: re-answer with the strong (cloud) model.
    CLARIFY = "CLARIFY"  #: ask the user a clarifying question.


@dataclass(frozen=True)
class ActionDecision:
    """A chosen action with its rationale (logged to the trace)."""

    action: Action
    rationale: str
    confidence: float = 0.0  # verifier score that motivated this step
    iteration: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
