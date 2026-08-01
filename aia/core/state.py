"""The assistant's state machine, and the stopwatch that keeps it honest.

Latency is the requirement most likely to rot silently — a model swap or an
extra round trip costs 300 ms and nothing visibly breaks. So every stage
transition is timestamped and each turn logs a breakdown against the budget.
The verification step in docs/PLAN.md reads exactly these lines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class State(Enum):
    IDLE = "idle"            # listening for the wake word only
    LISTENING = "listening"  # wake fired, capturing the utterance
    THINKING = "thinking"    # transcribing / routing
    ACTING = "acting"        # a plugin command is running
    SPEAKING = "speaking"    # audio going out
    ERROR = "error"


@dataclass
class Turn:
    """Timings for one wake-to-reply cycle, in milliseconds from the wake word."""

    started: float = field(default_factory=time.monotonic)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks[name] = (time.monotonic() - self.started) * 1000

    @property
    def total_ms(self) -> float:
        return (time.monotonic() - self.started) * 1000

    def report(self, budget_ms: int) -> str:
        parts = " ".join(f"{k}={v:.0f}" for k, v in self.marks.items())
        total = self.total_ms
        verdict = "OK" if total <= budget_ms else f"OVER by {total - budget_ms:.0f}ms"
        return f"turn {total:.0f}ms [{verdict}] {parts}"


class Machine:
    """Tracks the current state and announces transitions.

    Kept free of any I/O so the UI overlay can subscribe to it later without
    the state machine knowing a display exists.
    """

    def __init__(self, budget_ms: int):
        self.budget_ms = budget_ms
        self._state = State.IDLE
        self._listeners: list = []
        self.turn: Turn | None = None

    @property
    def state(self) -> State:
        return self._state

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def to(self, state: State) -> None:
        if state is self._state:
            return
        previous, self._state = self._state, state
        log.debug("state %s -> %s", previous.value, state.value)
        for fn in self._listeners:
            try:
                fn(previous, state)
            except Exception:
                log.exception("state listener failed")

    def begin_turn(self) -> Turn:
        self.turn = Turn()
        return self.turn

    def end_turn(self) -> None:
        if self.turn is not None:
            report = self.turn.report(self.budget_ms)
            if self.turn.total_ms > self.budget_ms:
                log.warning(report)
            else:
                log.info(report)
            self.turn = None
        self.to(State.IDLE)
