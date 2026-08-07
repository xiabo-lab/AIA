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


# The mark that decides whether a turn was fast enough. Set by `main.py` the
# moment audio starts, not when it finishes.
VERDICT_MARK = "audio_out"


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

    def judged_ms(self, total: float | None = None) -> float:
        """The elapsed time the budget is actually about.

        Not the total. A turn does not end when the assistant stops being
        slow, it ends when it stops talking, and reading a reply aloud is not
        latency — it is the answer arriving. `main.py` marks `audio_out`
        before `speaker.wait()` for exactly this reason, with a comment
        saying so, and then this class spent every spoken turn comparing the
        total against the budget anyway. A 545 ms reply to a track title that
        takes three seconds to say was logged `OVER by 4938ms`.

        What the user experiences as slowness is the wait before hearing
        anything, so that is what is judged. Turns that never reach
        `audio_out` — an empty transcript, a turn that died — have nothing
        better to offer than the total.
        """
        if total is None:
            total = self.total_ms
        return self.marks.get(VERDICT_MARK, total)

    def report(self, budget_ms: int) -> tuple[str, bool]:
        """The log line, and whether it broke the budget.

        Both together because they are one judgement made from one reading of
        the clock. Asking separately meant `report()` and the caller each
        called `total_ms`, so the line printed one number and the decision to
        warn about it used another.
        """
        parts = " ".join(f"{k}={v:.0f}" for k, v in self.marks.items())
        total = self.total_ms
        judged = self.judged_ms(total)
        over = judged > budget_ms
        verdict = f"OVER by {judged - budget_ms:.0f}ms" if over else "OK"
        # The total is still shown. It is the wrong thing to judge and the
        # right thing to be able to see — a turn that answered in 400 ms and
        # then talked for nine seconds is worth noticing, just not as a
        # latency failure.
        return f"turn {judged:.0f}ms to audio [{verdict}] of {total:.0f}ms total · {parts}", over


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
            report, over = self.turn.report(self.budget_ms)
            if over:
                log.warning(report)
            else:
                log.info(report)
            self.turn = None
        self.to(State.IDLE)
