"""The conversation database: what it stores, and what it stops storing.

Against a temporary database only. Nothing here reads `CONFIG.retention`, so
these can never point at the real transcript.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from aia.core.config import RetentionConfig
from aia.ui.history import (AIA, USER, ConversationLog, open_database,
                            purge_expired, read_since)
from aia.ui.retention import Retention

HOUR = 3600.0
DAY = 24 * HOUR


class Expiry(unittest.TestCase):
    """The 24-hour rule on stored messages, at the boundary.

    Driven against a plain connection rather than through the writer thread, so
    a failure here is about the SQL and not about timing.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = open_database(Path(self._tmp.name) / "history.db")
        self.now = time.time()

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def add(self, age_hours: float, role: str = USER, text: str = "播放音乐") -> None:
        self.conn.execute(
            "INSERT INTO messages (ts, role, text, language) VALUES (?,?,?,?)",
            (self.now - age_hours * HOUR, role, text, "zh"))
        self.conn.commit()

    def remaining(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def test_older_than_24h_is_deleted(self):
        self.add(age_hours=25)
        self.add(age_hours=30)
        self.assertEqual(purge_expired(self.conn, self.now - DAY), 2)
        self.assertEqual(self.remaining(), 0)

    def test_just_under_24h_survives(self):
        """Criterion 6: cleanup must not remove data younger than 24 hours."""
        self.add(age_hours=23.98)
        self.add(age_hours=12)
        self.add(age_hours=0)
        self.assertEqual(purge_expired(self.conn, self.now - DAY), 0)
        self.assertEqual(self.remaining(), 3)

    def test_only_the_expired_half_goes(self):
        for age in (48, 36, 25, 23, 6, 0.1):
            self.add(age_hours=age)
        purge_expired(self.conn, self.now - DAY)
        self.assertEqual(self.remaining(), 3)

    def test_there_is_no_floor_on_conversation_text(self):
        """Unlike recordings, text has no exemption — an idle day empties it.

        The floor on recordings exists because audio cannot be recaptured. That
        argument does not apply here, and the requirement is unconditional.
        """
        for i in range(200):
            self.add(age_hours=25 + i * 0.01)
        purge_expired(self.conn, self.now - DAY)
        self.assertEqual(self.remaining(), 0)


class Reading(unittest.TestCase):
    """What the web UI asks for: everything after the id it already has."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = open_database(Path(self._tmp.name) / "history.db")
        now = time.time()
        for i in range(10):
            self.conn.execute(
                "INSERT INTO messages (ts, role, text, language) VALUES (?,?,?,?)",
                (now - (10 - i) * 60, USER if i % 2 == 0 else AIA, f"line {i}", "en"))
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_returns_oldest_first(self):
        rows = read_since(self.conn, 0, 100)
        self.assertEqual([r["text"] for r in rows], [f"line {i}" for i in range(10)])

    def test_since_id_returns_only_what_is_new(self):
        rows = read_since(self.conn, 7, 100)
        self.assertEqual([r["text"] for r in rows], ["line 7", "line 8", "line 9"])

    def test_limit_takes_the_newest_end(self):
        """A client that has been away gets the latest page, not the oldest."""
        rows = read_since(self.conn, 0, 3)
        self.assertEqual([r["text"] for r in rows], ["line 7", "line 8", "line 9"])

    def test_nothing_new_is_an_empty_list(self):
        self.assertEqual(read_since(self.conn, 10, 100), [])


class Recording(unittest.TestCase):
    """The threaded writer, end to end."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "history.db"
        self.log = ConversationLog(self.path)

    def tearDown(self) -> None:
        self.log.close()
        self._tmp.cleanup()

    def test_a_turn_round_trips(self):
        self.log.record(USER, "播放五月天", "zh")
        self.log.record(AIA, "正在播放五月天。", "zh")
        self.assertTrue(self.log.flush())

        rows = self.log.recent()
        self.assertEqual([r["role"] for r in rows], [USER, AIA])
        # Mandarin has to survive the round trip intact — it is most of what
        # this assistant hears.
        self.assertEqual(rows[0]["text"], "播放五月天")
        self.assertEqual(rows[1]["language"], "zh")

    def test_blank_messages_are_not_recorded(self):
        self.log.record(USER, "   ")
        self.log.record(AIA, "")
        self.assertTrue(self.log.flush())
        self.assertEqual(self.log.recent(), [])

    def test_text_is_stripped(self):
        self.log.record(USER, "  pause the music \n")
        self.assertTrue(self.log.flush())
        self.assertEqual(self.log.recent()[0]["text"], "pause the music")

    def test_purge_runs_on_the_writer_thread(self):
        conn = sqlite3.connect(str(self.path))
        try:
            self.log.record(USER, "recent")
            self.assertTrue(self.log.flush())
            conn.execute("UPDATE messages SET ts = ts - ?", (2 * DAY,))
            conn.commit()
        finally:
            conn.close()

        self.log.purge(time.time() - DAY)
        self.assertTrue(self.log.flush())
        self.assertEqual(self.log.recent(), [])


class Sweep(unittest.TestCase):
    """`Retention.sweep` doing both jobs in one pass."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.recordings = root / "utterances"
        self.recordings.mkdir()
        self.cfg = RetentionConfig(
            hours=24.0,
            recordings=self.recordings,
            keep_recordings=2,
            database=root / "history.db",
        )
        self.log = ConversationLog(self.cfg.database)

    def tearDown(self) -> None:
        self.log.close()
        self._tmp.cleanup()

    def test_one_pass_expires_both_kinds_of_data(self):
        import os

        now = time.time()
        for i, age in enumerate((48, 36, 2, 1)):
            path = self.recordings / f"r{i}.wav"
            path.write_bytes(b"RIFF")
            when = now - age * HOUR
            os.utime(path, (when, when))

        self.log.record(USER, "old")
        self.log.record(AIA, "old reply")
        self.assertTrue(self.log.flush())
        conn = sqlite3.connect(str(self.cfg.database))
        try:
            conn.execute("UPDATE messages SET ts = ts - ?", (2 * DAY,))
            conn.commit()
        finally:
            conn.close()

        removed = Retention(self.cfg, self.log).sweep(now=now)
        self.assertTrue(self.log.flush())

        # Two recordings were expired; the two newest were kept by the floor.
        self.assertEqual(removed, 2)
        self.assertEqual(len(list(self.recordings.glob("*.wav"))), 2)
        self.assertEqual(self.log.recent(), [])

    def test_sweep_on_empty_state_does_nothing(self):
        self.assertEqual(Retention(self.cfg, self.log).sweep(), 0)


class ConfigInvariant(unittest.TestCase):
    """The two recording rules must not contradict each other."""

    def test_floor_above_the_count_cap_is_refused(self):
        from aia.core.config import AudioConfig, Config

        with self.assertRaises(ValueError):
            Config(audio=AudioConfig(keep_utterances=50),
                   retention=RetentionConfig(keep_recordings=100))

    def test_the_shipped_configuration_is_consistent(self):
        from aia.core.config import CONFIG

        self.assertLessEqual(CONFIG.retention.keep_recordings,
                             CONFIG.audio.keep_utterances)


if __name__ == "__main__":
    unittest.main()
