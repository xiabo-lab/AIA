"""The 24-hour rule, checked at its edges.

Everything here runs against `tempfile` directories and synthetic files. It must
stay that way: the real recording directory sits under `.bench/`, next to
measurement corpora that cannot be recreated, and a test that pruned the live
directory once destroyed 101 captures. No test in this file may take a path
from `CONFIG`.

Runs anywhere — no microphone, no ALSA, no numpy. That is deliberate. This
project can otherwise only be tested on the Pi, and expiry logic is the one
part that is pure enough to check on the machine it is written on:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from aia.ui.retention import expired_recordings, prune_recordings

HOUR = 3600.0
DAY = 24 * HOUR


class RecordingsBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.now = time.time()
        self.cutoff = self.now - DAY

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, age_hours: float) -> Path:
        """A .wav of a given age. Age is set on the file, not implied by name."""
        path = self.dir / name
        path.write_bytes(b"RIFF....WAVEfmt ")
        when = self.now - age_hours * HOUR
        os.utime(path, (when, when))
        return path

    def names(self, paths) -> set[str]:
        return {p.name for p in paths}


class ExpiryBoundary(RecordingsBase):
    """Exactly what "older than 24 hours" means, on both sides of the line."""

    def setUp(self) -> None:
        super().setUp()
        # Well past the floor, so the age rule is what decides every case here.
        self.keep = 2

    def test_older_than_24h_expires(self):
        self.write("old.wav", age_hours=24.5)
        self.write("a.wav", age_hours=0.1)
        self.write("b.wav", age_hours=0.2)
        expired = expired_recordings(self.dir, self.cutoff, self.keep)
        self.assertEqual(self.names(expired), {"old.wav"})

    def test_just_under_24h_survives(self):
        """23 h 59 m is not 24 hours old. This is criterion 6."""
        self.write("nearly.wav", age_hours=23.98)
        self.write("a.wav", age_hours=0.1)
        self.write("b.wav", age_hours=0.2)
        self.assertEqual(expired_recordings(self.dir, self.cutoff, self.keep), [])

    def test_exactly_at_the_boundary_survives(self):
        """The comparison is strict, so a file exactly on the line is kept."""
        path = self.write("edge.wav", age_hours=0)
        os.utime(path, (self.cutoff, self.cutoff))
        self.write("a.wav", age_hours=0.1)
        self.write("b.wav", age_hours=0.2)
        self.assertEqual(expired_recordings(self.dir, self.cutoff, self.keep), [])

    def test_fresh_recordings_are_never_touched(self):
        for i in range(20):
            self.write(f"f{i:02d}.wav", age_hours=i * 0.5)  # 0 to 9.5 hours
        self.assertEqual(expired_recordings(self.dir, self.cutoff, self.keep), [])


class KeepFloor(RecordingsBase):
    """The newest N survive the clock. Recordings cannot be remade."""

    def test_newest_are_kept_however_old(self):
        for i in range(10):
            # All ancient; only their order distinguishes them.
            self.write(f"r{i:02d}.wav", age_hours=100 + i)
        expired = expired_recordings(self.dir, self.cutoff, keep=4)
        # r00 is the newest of the ten (age 100 h), r09 the oldest (109 h).
        self.assertEqual(self.names(expired),
                         {"r04.wav", "r05.wav", "r06.wav", "r07.wav", "r08.wav", "r09.wav"})

    def test_fewer_files_than_the_floor_means_nothing_expires(self):
        for i in range(5):
            self.write(f"r{i}.wav", age_hours=200)
        self.assertEqual(expired_recordings(self.dir, self.cutoff, keep=100), [])

    def test_floor_and_clock_must_both_be_satisfied(self):
        """Beyond the floor is not enough on its own — the file must also be old."""
        for i in range(10):
            self.write(f"new{i}.wav", age_hours=1 + i * 0.1)   # all recent
        expired = expired_recordings(self.dir, self.cutoff, keep=3)
        self.assertEqual(expired, [], "recent files were deleted for being beyond the floor")

    def test_the_hundred_newest_survive_a_full_directory(self):
        """The configured floor, against a directory that is entirely expired."""
        for i in range(150):
            self.write(f"r{i:03d}.wav", age_hours=48 + i * 0.01)
        expired = expired_recordings(self.dir, self.cutoff, keep=100)
        self.assertEqual(len(expired), 50)
        survivors = {p.name for p in self.dir.glob("*.wav")} - self.names(expired)
        self.assertEqual(len(survivors), 100)
        # And the survivors are the newest, not an arbitrary hundred.
        self.assertIn("r000.wav", survivors)
        self.assertNotIn("r149.wav", survivors)


class Scope(RecordingsBase):
    """What the sweep is allowed to see. The narrow part is the important part."""

    def test_subdirectories_are_invisible(self):
        """A corpus in a subdirectory must survive, whatever its age.

        `.bench/` holds `wake-trials-pre-phasefix/` — 27 recordings captured
        through a decimator bug that no longer exists, and the only surviving
        "before" audio in the project. A recursive glob here would take them.
        """
        corpus = self.dir / "wake-trials-pre-phasefix"
        corpus.mkdir()
        for i in range(5):
            path = corpus / f"trial{i}.wav"
            path.write_bytes(b"RIFF")
            old = self.now - 90 * DAY
            os.utime(path, (old, old))
        for i in range(200):
            self.write(f"r{i:03d}.wav", age_hours=48)

        prune_recordings(self.dir, self.cutoff, keep=100)
        self.assertEqual(len(list(corpus.glob("*.wav"))), 5)

    def test_only_wav_files(self):
        note = self.dir / "notes.md"
        note.write_text("measurements worth keeping")
        old = self.now - 90 * DAY
        os.utime(note, (old, old))
        for i in range(150):
            self.write(f"r{i:03d}.wav", age_hours=48)

        prune_recordings(self.dir, self.cutoff, keep=100)
        self.assertTrue(note.exists())

    def test_missing_directory_is_not_an_error(self):
        self.assertEqual(expired_recordings(self.dir / "gone", self.cutoff, 10), [])
        self.assertEqual(prune_recordings(self.dir / "gone", self.cutoff, 10), 0)

    def test_empty_directory_is_not_an_error(self):
        self.assertEqual(prune_recordings(self.dir, self.cutoff, 10), 0)


class Pruning(RecordingsBase):
    """The deletion itself, not just the selection."""

    def test_expired_files_are_removed_and_counted(self):
        for i in range(120):
            self.write(f"old{i:03d}.wav", age_hours=48 + i * 0.01)
        removed = prune_recordings(self.dir, self.cutoff, keep=100)
        self.assertEqual(removed, 20)
        self.assertEqual(len(list(self.dir.glob("*.wav"))), 100)

    def test_pruning_twice_is_stable(self):
        for i in range(120):
            self.write(f"old{i:03d}.wav", age_hours=48 + i * 0.01)
        prune_recordings(self.dir, self.cutoff, keep=100)
        self.assertEqual(prune_recordings(self.dir, self.cutoff, keep=100), 0)


if __name__ == "__main__":
    unittest.main()
