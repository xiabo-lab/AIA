"""The speech output path, without Piper and without a sound card.

The interesting question here is not how a sentence is synthesised — that is
Piper's problem and it is measured on the Pi. It is whether AIA can tell that
nobody can hear it. It could not: `warm()` synthesised into a void for a day
while the only HDMI sink was held by another process, and every boot logged two
successful voice timings on the way.

So `Speaker` is built without ever starting a subprocess, and the only thing
faked is `sounddevice`.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from aia.tts.piper import Speaker


class FakeVoice:
    """Enough of a `Voice` for the probe: it only asks for a sample rate."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate


def speaker_with(voices: dict, play_raises: Exception | None = None) -> Speaker:
    """A `Speaker` that never spawned Piper, with `sd.play` under our control."""
    spk = object.__new__(Speaker)
    spk._voices = voices
    spk._played: list[tuple[np.ndarray, int]] = []

    def fake_play(samples, rate, blocking):
        if play_raises is not None:
            # Same shape as the real thing: three attempts, then give up.
            logging.getLogger("aia.tts.piper").error("fake failure")
            return False
        spk._played.append((samples, rate))
        return True

    spk._play = fake_play
    return spk


class TestTestTone(unittest.TestCase):
    def test_is_quiet_and_short(self):
        tone = Speaker._test_tone(22050)
        self.assertEqual(tone.dtype, np.int16)
        self.assertEqual(len(tone), int(22050 * 120 / 1000))
        # Well under full scale. A boot chime at 0 dBFS is a fright, and this
        # fires on every start.
        self.assertLess(np.abs(tone).max(), 0.1 * 32767)

    def test_starts_and_ends_at_silence(self):
        """No click. A tone cut off mid-cycle is louder than the tone."""
        tone = Speaker._test_tone(22050)
        self.assertEqual(tone[0], 0)
        self.assertLess(abs(int(tone[-1])), 32)

    def test_is_not_silence(self):
        tone = Speaker._test_tone(22050)
        self.assertGreater(np.abs(tone).max(), 0.01 * 32767)


class TestProbeOutput(unittest.TestCase):
    def test_reports_success_and_actually_plays(self):
        spk = speaker_with({"en": FakeVoice(22050)})
        self.assertTrue(spk.probe_output())
        self.assertEqual(len(spk._played), 1)
        samples, rate = spk._played[0]
        self.assertEqual(rate, 22050)
        self.assertGreater(len(samples), 0)

    def test_uses_the_voice_sample_rate(self):
        """A probe at the wrong rate is a pitch-shifted beep that still proves
        the device opens — but it would mask a rate the device cannot take."""
        spk = speaker_with({"zh": FakeVoice(16000)})
        spk.probe_output()
        self.assertEqual(spk._played[0][1], 16000)

    def test_failure_is_loud_and_does_not_raise(self):
        """The whole point. A dead output must be visible at ERROR, and must
        not stop an assistant that can still hear and act."""
        spk = speaker_with({"en": FakeVoice()}, play_raises=RuntimeError("no device"))
        with self.assertLogs("aia.tts.piper", level="ERROR") as captured:
            self.assertFalse(spk.probe_output())
        self.assertTrue(any("AUDIO OUTPUT IS DEAD" in line for line in captured.output))

    def test_survives_having_no_voices(self):
        """Every voice file missing is already logged per voice; the probe must
        not turn that into a crash on top."""
        spk = speaker_with({})
        spk.cfg = type("Cfg", (), {"sample_rate": 22050})()
        self.assertTrue(spk.probe_output())


if __name__ == "__main__":
    unittest.main()
