"""Utterance endpointing.

The wake word tells us speech is starting. This tells us it has stopped, which
is the harder half: end too early and the user is cut off mid-sentence, end too
late and every command carries the delay.

`silence_ms` is charged directly to the fast-path latency budget, so it is a
deliberate trade rather than a tuning afterthought.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import webrtcvad

from aia.core.config import AudioConfig, VadConfig

log = logging.getLogger(__name__)

# Peak level above which a capture counts as too hot to endpoint: -6 dBFS of
# a 16-bit sample. Not a guess — it is where the two populations actually
# separate. Every capture that ran to the cap peaked between -3.6 and
# 0.0 dBFS; every one that endpointed normally peaked between -10.8 and
# -24.3. See `Endpointer._report_cap`.
_HOT_PEAK = 16422


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
        # Everything heard during a capture that was then rejected, and why.
        # See `collect` — the caller saves these so a failure can be listened
        # to instead of guessed at.
        self.rejected: np.ndarray | None = None
        self.reject_reason: str | None = None

    @property
    def _run_bar_ms(self) -> int:
        """`min_run_ms`, lowered to something a run can actually reach.

        Runs are counted a frame at a time, so their length is always a
        multiple of `frame_ms` — at 30 ms frames a run is 360, 390 or 420 ms
        and never 400. Comparing against a bar of 400 therefore demands 420,
        and 390 is the closest a capture can come without passing.

        That is not a rounding curiosity. Three separate failures on
        2026-08-07 reported *exactly* 390 ms unbroken, which is the value the
        grid produces when speech ran right up to the intended bar:

          19:35:34  discarding utterance ... (390 ms unbroken)
          21:16:54  discarding utterance ... (390 ms unbroken)
          23:54:14  utterance stalled ...    (390 ms unbroken)

        So the bar is floored onto the grid rather than the config number being
        edited: 400 keeps meaning "about four hundred", and the frame size —
        which is an audio-format constraint, not a tuning knob — stops silently
        adding 5% to every threshold expressed in milliseconds.
        """
        return self.cfg.min_run_ms - (self.cfg.min_run_ms % self._frame_ms)

    @property
    def _speech_bar_ms(self) -> int:
        """`min_speech_ms`, on the frame grid, for the same reason as above.

        Voiced time is accumulated a frame at a time exactly as runs are, so a
        bar of 500 ms is really 510 — it can only be met at 480 or 510. That
        cost a real 关机 the turn it was said in. Aligned here rather than left
        to whoever next edits the number.
        """
        return self.cfg.min_speech_ms - (self.cfg.min_speech_ms % self._frame_ms)

    def _is_utterance(self, speech_ms: int, longest_run_ms: int) -> bool:
        """Does this look like something a person said on purpose?

        Two ways to qualify. Enough voiced audio *and* enough of it unbroken is
        the ordinary one. The second exists because the unbroken bar was
        calibrated against captures holding a few hundred milliseconds of
        speech and is simply not evidence about a capture holding seconds of
        it — see `VadConfig.ample_speech_ms`.
        """
        if speech_ms < self._speech_bar_ms:
            return False
        return (longest_run_ms >= self._run_bar_ms
                or speech_ms >= self.cfg.ample_speech_ms)

    def _fresh_vad(self) -> None:
        """Start each utterance from the same state as every other one.

        `webrtcvad.Vad` adapts to what it has heard, and the assistant holds two
        of these — one for commands and one for confirmations. The confirmation
        one runs rarely, so its adaptation was always minutes stale by the time
        it mattered, which is the single moment that must not misfire: it
        decides whether an irreversible action goes ahead.

        Constructing one is trivial, and starting fresh also makes the decision
        reproducible, so scripts/wake_test.py replaying a recording reaches the
        same verdict the live loop did.
        """
        self.vad = webrtcvad.Vad(self.cfg.aggressiveness)

    def _is_speech(self, frame: np.ndarray) -> bool:
        # webrtcvad wants exactly 10/20/30 ms of 16-bit mono PCM. A short
        # trailing frame from the queue would raise, so treat it as silence.
        if len(frame) != self.audio.frame_samples:
            return False
        return self.vad.is_speech(frame.tobytes(), self.audio.target_rate)

    def _report_cap(self, peak: int, speech_ms: int, captured_ms: int) -> None:
        """Say why an utterance could only end by running out of room.

        Hitting `max_utterance_ms` always means the endpointer never saw
        `silence_ms` of quiet, so the two numbers worth having are how much of
        the capture read as speech and how loud it was. This exists because
        working that out the first time took amplifying WAVs offline, and the
        answer should be one line in the journal instead.

        A microphone near full scale makes webrtcvad call *every* frame
        speech. Measured: a capture that is 33% voiced at its own level is
        100% voiced once amplified to peak 0.0 dBFS, on every file tried. The
        bar is proximity to the rail rather than clipping — at +24 dB only
        0.8% of samples actually clipped and the verdict was already 100%
        voiced, and two real capped captures had *no* clipped samples at all.

        Logged, not acted on. Refusing to trust the detector on loud frames is
        a tuning change that needs measuring against real speech first: a
        shout is loud and is also a real command.
        """
        peak_dbfs = -99.0 if peak <= 0 else 20 * math.log10(peak / 32768)
        voiced_pct = 100.0 * speech_ms / max(captured_ms, 1)
        if peak >= _HOT_PEAK:
            log.warning(
                "that capture peaked at %.1f dBFS and read %.0f%% speech — too "
                "hot to endpoint. Near full scale every frame looks like "
                "speech, so the utterance can only end at the cap. Lower the "
                "capture gain (amixer -c <n> sset Mic <lower>) and turn Auto "
                "Gain Control off.",
                peak_dbfs, voiced_pct,
            )
        else:
            log.warning(
                "capture ran to the cap at %.1f dBFS peak, %.0f%% speech — the "
                "endpointer never saw %d ms of quiet.",
                peak_dbfs, voiced_pct, self.cfg.silence_ms,
            )

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
        voiced audio, of which `min_run_ms` is unbroken. Otherwise a cough, a
        door, or the tail of the wake word can still end the turn before the
        command arrives.

        The unbroken part is not decoration. Total voiced time is a sum over
        the whole capture, so a scatter of unrelated noises adds up to a
        command; one missed recording held 330 ms of speech whose longest
        continuous stretch was 150 ms. Real phrases here run 1260 ms unbroken.

        It is also not evidence about a capture holding *seconds* of speech,
        which is why `ample_speech_ms` waives it — see `_is_utterance`. And the
        bar itself is frame-aligned before use, because a threshold in
        milliseconds that the frame grid cannot land on is 5% higher than it
        reads; see `_run_bar_ms`.

        Falling short of either bar does not end the turn — the loop keeps
        listening, up to `max_wait_ms` of silence. That is the whole point: the
        failure being fixed was a breath ending the capture a second before the
        command was spoken, and the cure is to still be listening when it is.

        **A rejection keeps its audio.** Every path that returns None leaves
        what it heard on `self.rejected`, with `self.reject_reason` naming the
        path. Until this existed a failed turn destroyed its own evidence: the
        journal said "390 ms unbroken" and there was no way to find out whether
        that was the user's voice being chopped up or the room's music being
        counted, because the only recordings kept were the ones that worked.
        """
        collected: list[np.ndarray] = []
        preroll: list[np.ndarray] = []
        # Every frame this call consumed, kept only so a rejection can be
        # saved and listened to. `collected` starts at speech onset and so is
        # empty on the one failure that matters most — the capture that never
        # started — which is exactly the audio needed to tell "the user said
        # nothing" from "the user spoke and we did not hear it". Bounded by the
        # loop's own exits: max_wait_ms then max_utterance_ms, ~14 s worst case.
        heard: list[np.ndarray] = []
        silence_ms = 0
        waited_ms = 0
        onset_ms = 0
        speech_ms = 0
        run_ms = 0
        longest_run_ms = 0
        started = False
        peak_sample = 0

        self._fresh_vad()
        self.rejected = None
        self.reject_reason = None
        preroll_frames = max(1, self.cfg.preroll_ms // self._frame_ms)

        def reject(reason: str) -> None:
            """Hand the caller the audio behind a rejection, then drop it."""
            self.reject_reason = reason
            self.rejected = np.concatenate(heard) if heard else None

        for frame in frames:
            speech = self._is_speech(frame)
            heard.append(frame)
            if on_frame is not None:
                on_frame(frame, speech, started)

            # Track how close the input ran to the rail. See `_report_cap`.
            if len(frame):
                peak_sample = max(peak_sample, int(np.max(np.abs(frame))))

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
                    speech_ms = run_ms = longest_run_ms = onset_ms
                elif waited_ms >= self.cfg.max_wait_ms:
                    log.info("no speech after wake word (%d ms)", waited_ms)
                    reject("nospeech")
                    return None
                continue

            collected.append(frame)
            if speech:
                speech_ms += self._frame_ms
                run_ms += self._frame_ms
                longest_run_ms = max(longest_run_ms, run_ms)
                silence_ms = 0
            else:
                silence_ms += self._frame_ms
                run_ms = 0

            # Not ending on a pause unless what came before it looks like
            # something a person said on purpose. Falling short here does not
            # end the turn — the loop simply keeps listening, which is what
            # gives the real command, arriving a second after somebody cleared
            # their throat, somewhere to land.
            if (silence_ms >= self.cfg.silence_ms
                    and self._is_utterance(speech_ms, longest_run_ms)):
                break
            if len(collected) * self._frame_ms >= self.cfg.max_utterance_ms:
                log.warning("utterance hit the %d ms cap", self.cfg.max_utterance_ms)
                self._report_cap(peak_sample, speech_ms,
                                 len(collected) * self._frame_ms)
                break
            # Started on something that turned out not to be speech: give up
            # waiting rather than sitting here until the hard cap.
            if silence_ms >= self.cfg.max_wait_ms:
                log.info("utterance stalled with only %d ms of speech (%d ms unbroken)",
                         speech_ms, longest_run_ms)
                reject("stalled")
                return None

        if not collected or not self._is_utterance(speech_ms, longest_run_ms):
            log.info("discarding utterance with %d ms of speech (%d ms unbroken, "
                     "bar %d/%d ms)", speech_ms, longest_run_ms,
                     self._speech_bar_ms, self._run_bar_ms)
            reject("discarded")
            return None
        audio = np.concatenate(collected)
        log.info("utterance: %.2f s (%d ms voiced, %d ms unbroken)",
                 len(audio) / self.audio.target_rate, speech_ms, longest_run_ms)
        return audio
