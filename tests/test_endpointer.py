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

import queue
import types
import unittest

try:
    import numpy as np

    from aia.audio.capture import Microphone
    from aia.audio.vad import Endpointer
    from aia.core.config import CONFIG

    DEPS = True
except ImportError:  # no numpy/webrtcvad — the development machine
    DEPS = False


@unittest.skipUnless(DEPS, "needs numpy and webrtcvad; run this one on the Pi")
class EndpointerDecision(unittest.TestCase):
    """What counts as something a person said on purpose.

    The point is not that the bars now admit more — it is that they admit the
    four captures below, which were real commands, and still reject the two
    after them, which were noise.
    """

    # 390 ms is the longest run the 30 ms frame grid can produce below a
    # 400 ms bar, and it is the number all three failures reported.
    NEAR_MISS_RUN_MS = 390

    @property
    def frame_ms(self) -> int:
        return CONFIG.audio.frame_ms

    def _frames(self, n: int):
        return (np.zeros(CONFIG.audio.frame_samples, dtype=np.int16)
                for _ in range(n))

    def _run(self, pattern):
        """Returns (audio, reason). Trailing silence lets every path exit."""
        ep = Endpointer(CONFIG.audio, CONFIG.vad)
        it = iter(pattern)
        ep._is_speech = lambda frame: next(it, False)
        audio = ep.collect(self._frames(len(pattern) + 2000))
        return audio, ep.reject_reason

    def _speech(self, ms: int):
        return [True] * (ms // self.frame_ms)

    def _silence(self, ms: int):
        return [False] * (ms // self.frame_ms)

    def _fragmented(self, total_ms: int, run_ms: int):
        """Speech totalling `total_ms`, never unbroken for longer than `run_ms`.

        This is what a command spoken over ducked music looks like to
        webrtcvad: plenty of voiced audio, chopped up by the song behind it.
        """
        pattern: list[bool] = []
        remaining = total_ms
        while remaining > 0:
            chunk = min(run_ms, remaining)
            if pattern:
                pattern += self._silence(self.frame_ms)
            pattern += self._speech(chunk)
            remaining -= chunk
        return pattern

    def test_run_bar_is_reachable_on_the_frame_grid(self):
        """A bar of 400 ms demanded 420 — runs only come in frame steps."""
        ep = Endpointer(CONFIG.audio, CONFIG.vad)
        self.assertEqual(ep._run_bar_ms % self.frame_ms, 0)
        self.assertLessEqual(ep._run_bar_ms, CONFIG.vad.min_run_ms)
        self.assertGreaterEqual(self.NEAR_MISS_RUN_MS, ep._run_bar_ms)

    def test_long_fragmented_capture_is_kept(self):
        """19:35:34 and 21:16:54 — seconds of speech, thrown away."""
        for total_ms in (3780, 3240):
            with self.subTest(total_ms=total_ms):
                audio, reason = self._run(
                    self._fragmented(total_ms, self.NEAR_MISS_RUN_MS)
                    + self._silence(600))
                self.assertIsNotNone(
                    audio, f"{total_ms} ms of speech was discarded ({reason})")

    def test_short_command_at_the_grid_boundary_is_kept(self):
        """23:54:14 — 720 ms of speech, 390 ms unbroken, stalled and lost."""
        audio, reason = self._run(
            self._fragmented(720, self.NEAR_MISS_RUN_MS) + self._silence(600))
        self.assertIsNotNone(audio, f"a 720 ms command was rejected ({reason})")

    def test_speech_bar_is_reachable_on_the_frame_grid(self):
        """500 ms demanded 510, for the same reason 400 demanded 420."""
        ep = Endpointer(CONFIG.audio, CONFIG.vad)
        self.assertEqual(ep._speech_bar_ms % self.frame_ms, 0)
        self.assertLessEqual(ep._speech_bar_ms, CONFIG.vad.min_speech_ms)

    def test_a_short_two_syllable_command_is_kept(self):
        """10:49:15 — 450 ms of voiced audio that transcribed to 关机.

        The command being reported as broken, thrown away by the bar meant to
        reject coughs. It is one unbroken run, which is what tells it from one.
        """
        audio, reason = self._run(self._speech(450) + self._silence(600))
        self.assertIsNotNone(audio, f"'关机' was rejected ({reason})")

    def test_capture_holding_almost_nothing_is_still_rejected(self):
        """23:53:19 — 210 ms of speech. This one really was nothing."""
        audio, reason = self._run(self._speech(210) + self._silence(5000))
        self.assertIsNone(audio)
        self.assertEqual(reason, "stalled")

    def test_a_cough_is_still_rejected(self):
        """What `min_run_ms` is for: a scatter of noise, not a command.

        330 ms total with a 150 ms longest run is the recording that motivated
        the unbroken-run bar. Waiving that bar for long captures must not
        waive it for this one.
        """
        audio, _ = self._run(self._fragmented(330, 150) + self._silence(5000))
        self.assertIsNone(audio)

    def test_an_ordinary_command_still_needs_an_unbroken_run(self):
        """The waiver is for long captures only — see ample_speech_ms."""
        self.assertGreater(CONFIG.vad.ample_speech_ms, CONFIG.vad.min_speech_ms)
        # 600 ms total, longest run 150 ms: above min_speech, below ample.
        audio, _ = self._run(self._fragmented(600, 150) + self._silence(5000))
        self.assertIsNone(audio)


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


@unittest.skipUnless(DEPS, "needs numpy and webrtcvad; run this one on the Pi")
class BoundedDrain(unittest.TestCase):
    """`Microphone.drain(keep_ms=...)` — what survives the wake word.

    Driven against a stub rather than a real `Microphone`, because
    constructing one opens the capture device and the assistant is normally
    holding it. Only `_q` and `cfg` are touched.
    """

    def _stub(self, n_blocks: int):
        s = types.SimpleNamespace()
        s.cfg = CONFIG.audio
        s._q = queue.Queue(maxsize=1000)
        s._dropped_samples = 0
        s._drain_baseline = 0
        for i in range(n_blocks):
            s._q.put_nowait(np.full(self.block, i, dtype=np.int16))
        return s

    @property
    def block(self) -> int:
        return CONFIG.audio.capture_rate // 100  # 10 ms

    def _remaining(self, s):
        out = []
        while not s._q.empty():
            out.append(s._q.get_nowait())
        return out

    def test_keeps_the_newest_audio_in_order(self):
        """The tail is the user's opening syllable; the head is the music."""
        s = self._stub(100)  # 1000 ms buffered
        Microphone.drain(s, keep_ms=700)

        left = self._remaining(s)
        kept_ms = sum(len(b) for b in left) * 1000 / CONFIG.audio.capture_rate
        ids = [int(b[0]) for b in left]
        self.assertGreaterEqual(kept_ms, 700)
        self.assertLess(kept_ms, 700 + self.block * 1000 / CONFIG.audio.capture_rate + 1)
        self.assertEqual(ids[-1], 99, "the newest block must survive")
        self.assertEqual(ids, sorted(ids), "spliced audio would corrupt timing")

    def test_default_still_takes_everything(self):
        """After speaking, all of it is the assistant's own voice."""
        s = self._stub(100)
        Microphone.drain(s)
        self.assertTrue(s._q.empty())

    def test_a_buffer_shorter_than_keep_ms_is_left_alone(self):
        """The steady-state case: the queue is near-empty and nothing is lost."""
        s = self._stub(3)  # 30 ms buffered
        Microphone.drain(s, keep_ms=700)
        self.assertEqual(s._q.qsize(), 3)


if __name__ == "__main__":
    unittest.main()
