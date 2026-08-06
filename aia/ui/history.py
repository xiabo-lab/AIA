"""The conversation, written down.

Everything the user said and everything AIA answered, kept for
`RetentionConfig.hours` and then deleted. This is what the web UI reads; the
GTK overlay keeps its own three-line view and knows nothing about any of it.

Two properties matter more than anything else here, and they are why this is
not simply `sqlite3.connect()` at the call site:

**The voice loop must never wait on a disk.** `record()` enqueues and returns.
It does not open a file, take a lock, or touch the database, and a full queue
drops the message rather than blocking — the SD card this runs on can stall for
tens of milliseconds, and a turn has a 2.5 s budget with an audible cost for
overrunning it. A lost line of transcript is invisible; a stalled turn is not.

**One writer, always.** Every insert and every purge happens on the same thread
through the same connection, so there is no write contention to reason about
and no "database is locked" to handle on the path that matters. Readers — the
HTTP thread — open their own short-lived connection and only ever SELECT.

Timestamps are wall clock (`time.time`), not monotonic. Retention is a promise
about the time of day, and it has to survive a restart; monotonic clocks do
neither.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Roles that may be written. `status` lines ("Listening…") are deliberately not
# among them: they are the overlay saying it is awake, not something anybody
# said, and storing them would put a line of furniture between every real turn.
USER = "user"
AIA = "aia"

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL    NOT NULL,
    role     TEXT    NOT NULL,
    text     TEXT    NOT NULL,
    language TEXT
);
CREATE INDEX IF NOT EXISTS messages_ts ON messages (ts);
"""

# Deep enough to ride out a slow write, shallow enough that a wedged writer
# cannot accumulate a session's worth of transcript in memory. A turn produces
# two rows, so this is ~128 turns of backlog.
QUEUE_DEPTH = 256

_STOP = object()


def open_database(path: Path) -> sqlite3.Connection:
    """A connection with the schema in place.

    WAL because there is one writer and one reader and they must not block each
    other — the reader is an HTTP request being served while a turn is in
    flight, and the default rollback journal makes that a lock contest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # Durability is not worth a disk sync per turn here. The worst case for
    # NORMAL is losing the last few messages to a power cut, against an fsync
    # on the SD card on the path this module exists to keep off.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def purge_expired(conn: sqlite3.Connection, cutoff: float) -> int:
    """Delete messages older than `cutoff` (a unix timestamp). Returns how many.

    Strictly `<` so a message exactly at the boundary survives. The boundary
    case is what the retention tests pin down, and "older than 24 hours" reads
    on the side of keeping.
    """
    cur = conn.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


def read_since(conn: sqlite3.Connection, since_id: int, limit: int) -> list[dict]:
    """Messages after `since_id`, oldest first — the order they are displayed in.

    `limit` bounds the *newest* end rather than the oldest, so a client that has
    been away for an hour receives the most recent page and not the start of the
    backlog it would then have to scroll past.
    """
    rows = conn.execute(
        "SELECT id, ts, role, text, language FROM messages "
        "WHERE id > ? ORDER BY id DESC LIMIT ?",
        (since_id, limit),
    ).fetchall()
    return [dict(row) for row in reversed(rows)]


class ConversationLog:
    """Records the conversation off the voice loop's thread.

    Degrades to a no-op if the database cannot be opened — a read-only card, a
    full disk. Losing the transcript is a shame; refusing to listen because of
    it would be a fault.
    """

    def __init__(self, path: Path):
        self.path = path
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self._dropped = 0
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, name="aia-history", daemon=True)
        self._thread.start()

    # ── the voice loop's side: never blocks ──────────────────────────

    def record(self, role: str, text: str, language: str | None = None) -> None:
        text = text.strip()
        if not text or self._failed:
            return
        try:
            self._q.put_nowait(("insert", (time.time(), role, text, language)))
        except queue.Full:
            # Say it once per hundred, not once per message: the reason the
            # queue is full is that writing is already slow, and a log line per
            # drop would add work to exactly the thread that is behind.
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.warning("conversation history is behind; %d message(s) dropped",
                            self._dropped)

    def purge(self, cutoff: float) -> None:
        """Ask the writer to delete everything older than `cutoff`.

        Queued rather than done here so it cannot collide with an insert, and
        so the retention thread never waits on the database either.
        """
        try:
            self._q.put_nowait(("purge", cutoff))
        except queue.Full:
            log.debug("history queue full; purge deferred to the next sweep")

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until everything queued so far has been written.

        For shutdown and for tests. Nothing on the voice path calls this.
        """
        done = threading.Event()
        try:
            self._q.put(("flush", done), timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    # ── the reader's side: its own connection, SELECT only ───────────

    def recent(self, since_id: int = 0, limit: int = 200) -> list[dict]:
        """Read the transcript. Safe to call from another thread."""
        if self._failed:
            return []
        try:
            conn = sqlite3.connect(str(self.path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            try:
                return read_since(conn, since_id, limit)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            log.warning("could not read the conversation history: %s", exc)
            return []

    # ── the writer thread ────────────────────────────────────────────

    def _run(self) -> None:
        try:
            conn = open_database(self.path)
        except sqlite3.Error:
            log.exception("could not open %s; the conversation will not be recorded",
                          self.path)
            self._failed = True
            return

        log.info("conversation history at %s", self.path)
        try:
            while True:
                item = self._q.get()
                if item is _STOP:
                    return
                kind, payload = item
                try:
                    if kind == "insert":
                        conn.execute(
                            "INSERT INTO messages (ts, role, text, language) "
                            "VALUES (?, ?, ?, ?)", payload)
                        conn.commit()
                    elif kind == "purge":
                        removed = purge_expired(conn, payload)
                        if removed:
                            log.info("expired %d conversation message(s)", removed)
                    elif kind == "flush":
                        payload.set()
                except sqlite3.Error:
                    # One bad write must not end the recording for the session.
                    log.exception("history write failed")
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                log.debug("closing the history database failed", exc_info=True)

    def close(self) -> None:
        self.flush(timeout=2.0)
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
