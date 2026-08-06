"""Nothing outlives a day.

One background thread applies `RetentionConfig` to both things AIA accumulates:
the conversation database and the saved utterances. It runs once at startup —
so a restart is also a cleanup, and a machine that was off for a week comes back
with an empty transcript rather than a week-old one — and then on an interval.

Off the voice loop entirely. The sweep runs on its own thread, the database
delete is handed to the history writer's queue, and unlinking a hundred small
files takes single-digit milliseconds. Nothing here is on the path between the
wake word and a reply.

## The one dangerous thing in this module

Deleting recordings is irreversible and the directory it points at lives under
`.bench/`, which also holds measurement corpora that cannot be recreated —
`wake-trials-pre-phasefix/` is 27 captures taken through a decimator bug that
no longer exists, and it is the only "before" audio in the project. A prune
that walked `.bench/` would destroy it silently.

So the rules here are narrow on purpose:

* the directory is named explicitly in config, never derived or widened;
* the glob is `*.wav` and **never recursive**, so a subdirectory is invisible
  to this code no matter what it contains;
* the `keep` newest files are exempt from the clock entirely.

The last one is not caution for its own sake. A test once ran the count-prune
against the live directory with keep=3 and took 101 real captures down to 3;
the numbers derived from them survive only because they had been written into
commit messages.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from aia.core.config import RetentionConfig
from aia.ui.history import ConversationLog

log = logging.getLogger(__name__)


def expired_recordings(directory: Path, cutoff: float, keep: int,
                       now: float | None = None) -> list[Path]:
    """Which recordings may be deleted: older than `cutoff`, beyond the newest `keep`.

    Both conditions must hold. Age alone would empty the directory after a quiet
    day, and the newest captures are the ones a misrecognition is diagnosed from;
    a count alone is what let four days of everything said in the room sit on
    the card.

    Age is read from the file's mtime rather than parsed out of its name. The
    names are timestamps and sorting them is sorting by age, but mtime is the
    fact and the name is a convention — and a test can set mtime, which is what
    makes the 24-hour boundary something this project can actually check.

    Not recursive, by design. See the module docstring.
    """
    if not directory.is_dir():
        return []

    files: list[tuple[float, str, Path]] = []
    for path in directory.glob("*.wav"):
        try:
            files.append((path.stat().st_mtime, path.name, path))
        except OSError as exc:
            # Vanished between the glob and the stat, or unreadable. Either way
            # it is not this sweep's problem.
            log.debug("could not stat %s: %s", path, exc)

    # Newest first, so the protected set is a prefix. The name breaks ties,
    # which matters only for files written inside the same clock tick but makes
    # the result deterministic — and therefore testable.
    files.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [path for mtime, _name, path in files[max(0, keep):] if mtime < cutoff]


def prune_recordings(directory: Path, cutoff: float, keep: int) -> int:
    """Delete the expired recordings. Returns how many went."""
    removed = 0
    for path in expired_recordings(directory, cutoff, keep):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:  # being read, or a read-only card
            log.debug("could not delete %s: %s", path, exc)
    if removed:
        log.info("expired %d recording(s) from %s", removed, directory)
    return removed


class Retention:
    """The sweep, on its own thread."""

    def __init__(self, cfg: RetentionConfig, history: ConversationLog | None = None):
        self.cfg = cfg
        self.history = history
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sweep(self, now: float | None = None) -> int:
        """One pass. Returns the number of recordings deleted.

        The database side is queued rather than counted here — it happens on the
        history writer's thread, which is the only thread that writes, and it
        logs its own total.
        """
        now = time.time() if now is None else now
        cutoff = now - self.cfg.seconds
        if self.history is not None:
            self.history.purge(cutoff)
        return prune_recordings(self.cfg.recordings, cutoff, self.cfg.keep_recordings)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception:
                # A failed sweep must not end the sweeping. The realistic causes
                # — a card gone read-only, a directory removed underneath us —
                # are all things that may fix themselves.
                log.exception("retention sweep failed")
            self._stop.wait(self.cfg.sweep_interval_s)

    def start(self) -> None:
        log.info("retention: %g h, keeping the newest %d recording(s) in %s",
                 self.cfg.hours, self.cfg.keep_recordings, self.cfg.recordings)
        self._thread = threading.Thread(
            target=self._run, name="aia-retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
