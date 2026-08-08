"""The endpointer, against the turns it got wrong.

Cases here are real failures from the journal of 2026-08-07, reduced to the
numbers the decision actually uses.

No microphone and no webrtcvad verdicts: `_is_speech` is replaced by a scripted
pattern, so this drives the real `collect` loop over known audio content. The
frames themselves are silence and are never inspected.

Unlike the rest of `tests/`, this needs numpy — so on a machine without it the
module skips rather than fails, and the suite still runs anywhere:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    from aia.audio.vad import Endpointer
    from aia.core.config import CONFIG

    DEPS = True
except ImportError:  # no numpy/webrtcvad — the development machine
    DEPS = False


@unittest.skipUnless(DEPS, "needs numpy and webrtcvad; run this one on the Pi")
class RejectedAudioIsKept(unittest.TestCase):
    """A failed turn must stop destroying its own evidence.

    Every rejection used to return None and drop what it had heard, so the
    captures the user complains about were the only ones that could never be
    listened to — the journal said "390 ms unbroken" and there was no way to
    find out what that 390 ms was.
    """

    def _frames(self, n: int):
        return (np.zeros(CONFIG.audio.frame_samples, dtype=np.int16)
                for _ in range(n))

    def _endpointer(self, pattern):
        ep = Endpointer(CONFIG.audio, CONFIG.vad)
        it = iter(pattern)
        ep._is_speech = lambda frame: next(it, False)
        return ep

    def test_stalled_capture_keeps_what_it_heard(self):
        """23:53:19 — 210 ms of speech, then nothing. What was the 210 ms?"""
        pattern = ([True] * 7) + ([False] * 200)
        ep = self._endpointer(pattern)

        self.assertIsNone(ep.collect(self._frames(len(pattern) + 2000)))
        self.assertEqual(ep.reject_reason, "stalled")
        self.assertIsNotNone(ep.rejected)
        # Everything heard, not just what was collected after speech onset.
        self.assertGreater(len(ep.rejected), CONFIG.audio.frame_samples * 7)

    def test_silence_after_the_wake_word_is_kept_too(self):
        """The capture that never started is the one worth listening to.

        This is the path where `collected` is empty, so saving that instead of
        everything heard would have written a silent file and proved nothing.
        """
        ep = Endpointer(CONFIG.audio, CONFIG.vad)
        ep._is_speech = lambda frame: False
        self.assertIsNone(ep.collect(self._frames(2000)))
        self.assertEqual(ep.reject_reason, "nospeech")
        self.assertIsNotNone(ep.rejected)
        self.assertGreater(len(ep.rejected), 0)

    def test_a_successful_capture_leaves_no_rejection_behind(self):
        """Stale state here would save the wrong audio under the wrong name."""
        pattern = ([True] * 60) + ([False] * 40)
        ep = self._endpointer(pattern)

        self.assertIsNotNone(ep.collect(self._frames(len(pattern) + 2000)))
        self.assertIsNone(ep.rejected)
        self.assertIsNone(ep.reject_reason)

    def test_a_rejection_does_not_outlive_the_next_capture(self):
        """The attribute is per-call state, not a growing log."""
        ep = self._endpointer(([True] * 7) + ([False] * 200))
        self.assertIsNone(ep.collect(self._frames(400)))
        self.assertIsNotNone(ep.rejected)

        pattern = ([True] * 60) + ([False] * 40)
        it = iter(pattern)
        ep._is_speech = lambda frame: next(it, False)
        self.assertIsNotNone(ep.collect(self._frames(len(pattern) + 2000)))
        self.assertIsNone(ep.rejected)


if __name__ == "__main__":
    unittest.main()
