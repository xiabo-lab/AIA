"""Utterance endpointing.

The wake word tells us speech is starting. This tells us it has stopped, which
is the harder half: end too early and the user is cut off mid-sentence, end too
late and every command carries the delay.

`silence_ms` is charged directly to the fast-path latency budget, so it is a
deliberate trade rather than a tuning afterthought.
"""

from __future__ import annotations

import logging

import numpy as np
import webrtcvad

from aia.core.config import AudioConfig, VadConfig

log = logging.getLogger(__name__)


class Endpointer:
    """Collects frames until the user stops speaking.

    Note webrtcvad is a *voice activity* detector, not a speech recogniser: it
    will happily call a slammed door "speech". That is acceptable here because
    it only runs in the window after a wake word, and a spurious endpoint just
    produces a transcript the intent router will fail to match.
    """

    def __init__(self, audio: AudioConfig, cfg: VadConfig):
        self.audio = audio
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self._frame_ms = audio.frame_ms

    def _is_speech(self, frame: np.ndarray) -> bool:
        # webrtcvad wants exactly 10/20/30 ms of 16-bit mono PCM. A short
        # trailing frame from the queue would raise, so treat it as silence.
        if len(frame) != self.audio.frame_samples:
            return False
        return self.vad.is_speech(frame.tobytes(), self.audio.target_rate)

    def collect(self, frames, on_frame=None) -> np.ndarray | None:
        """Consume frames until the utterance ends. Returns 16 kHz int16 audio.

        `on_frame(frame, speech, started)` is called for every frame consumed,
        so a caller can show what is being heard as it arrives. It exists for
        scripts/wake_test.py, which displays a live transcript — the point of
        the hook is that the tool watches the *real* endpointer rather than
        reimplementing this loop and then disagreeing with it.

        Returns None if the user never actually said anything — the wake word
        fired on a false positive, or on the assistant's own output.

        Two guards here exist because of how people actually talk to an
        assistant. Both were added after watching this fail on real speech:

        **Onset debounce.** The wake word fires the instant the phrase is
        recognised, which is typically while the user is still finishing the
        last syllable of it. A single VAD frame of that tail used to be enough
        to declare the utterance started, so the natural pause that follows
        ("小艾同学" ... "play some music") was read as end-of-speech and the
        turn ended having captured 0.5 s of nothing. Speech must now persist
        for `onset_ms` before it counts as the start.

        **Minimum speech.** Even after a real start, an utterance is not
        allowed to end until it contains at least `min_speech_ms` of actual
        voiced audio. Otherwise a cough, a door, or the tail of the wake word
        can still end the turn before the command arrives.
        """
        collected: list[np.ndarray] = []
        preroll: list[np.ndarray] = []
        silence_ms = 0
        waited_ms = 0
        onset_ms = 0
        speech_ms = 0
        started = False

        preroll_frames = max(1, self.cfg.preroll_ms // self._frame_ms)

        for frame in frames:
            speech = self._is_speech(frame)
            if on_frame is not None:
                on_frame(frame, speech, started)

            if not started:
                waited_ms += self._frame_ms
                onset_ms = onset_ms + self._frame_ms if speech else 0

                # Hold a little pre-roll so the first phoneme is not clipped
                # when speech begins part-way through a frame.
                preroll.append(frame)
                if len(preroll) > preroll_frames:
                    preroll.pop(0)

                if onset_ms >= self.cfg.onset_ms:
                    started = True
                    collected = list(preroll)
                    speech_ms = onset_ms
                elif waited_ms >= self.cfg.max_wait_ms:
                    log.info("no speech after wake word (%d ms)", waited_ms)
                    return None
                continue

            collected.append(frame)
            if speech:
                speech_ms += self._frame_ms
                silence_ms = 0
            else:
                silence_ms += self._frame_ms

            if silence_ms >= self.cfg.silence_ms and speech_ms >= self.cfg.min_speech_ms:
                break
            if len(collected) * self._frame_ms >= self.cfg.max_utterance_ms:
                log.warning("utterance hit the %d ms cap", self.cfg.max_utterance_ms)
                break
            # Started on something that turned out not to be speech: give up
            # waiting rather than sitting here until the hard cap.
            if silence_ms >= self.cfg.max_wait_ms:
                log.info("utterance stalled with only %d ms of speech", speech_ms)
                return None

        if not collected or speech_ms < self.cfg.min_speech_ms:
            log.info("discarding utterance with %d ms of speech", speech_ms)
            return None
        audio = np.concatenate(collected)
        log.info("utterance: %.2f s (%d ms voiced)",
                 len(audio) / self.audio.target_rate, speech_ms)
        return audio
