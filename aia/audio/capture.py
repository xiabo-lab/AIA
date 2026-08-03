"""Microphone capture, resampled to what the models expect.

One process owns the microphone for the lifetime of the assistant. Everything
downstream — wake word, VAD, STT — reads 16 kHz mono int16 frames from the
queue this fills, so nothing else ever opens the device. Two consumers fighting
over an ALSA capture handle is the failure mode this design exists to prevent.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Iterator

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

from aia.core.config import AudioConfig

log = logging.getLogger(__name__)

# No audio for this long means the stream has died rather than the room being
# quiet — silence still arrives as frames. Generous enough never to fire on a
# busy turn.
STALL_TIMEOUT_S = 5.0

# Blocks are dropped whenever nothing is reading the queue, which is normal for
# the duration of a turn — that audio is stale and would be drained anyway. It
# is *not* normal while frames are being consumed steadily: that means the wake
# word cannot keep up with real time, and the audio it does see is spliced
# across the gap. The two cases are told apart by how long the consumer took to
# come back for the next block; anything under this was consuming steadily.
CONSUMING_GAP_S = 0.5

# How often that may be said out loud. Overruns arrive in bursts and per-burst
# logging turns the report into part of the fault: writing a line costs time in
# the very loop that is already failing to keep up, which drops more audio,
# which logs another line. Under scripts/wake_test.py — whose live transcript
# makes the consumer far heavier than the assistant's — that ran at nearly one
# line per frame and, over ssh, blocked on the terminal until the tool stopped
# dead. So the count is accumulated and reported on an interval.
OVERRUN_REPORT_S = 5.0

# How much audio the queue is allowed to hold. Sized against the consumer's
# worst case rather than its average: measured live on this Pi the wake word
# takes 0.30 ms on a median frame but 60 ms at p99 and 286 ms at worst, because
# Vosk's decoder finalises in bursts. Anything the queue cannot ride out is
# audio deleted from the middle of the stream, which is a missed wake word with
# nothing in the journal to explain it. A second is ~3.5x the worst spike seen.
#
# Depth is safe to spend only because the queue evicts its oldest audio rather
# than refusing new audio when it is full — see `_callback`. What it holds is
# therefore always the most recent second, so a consumer that has been away
# comes back to the live edge instead of to history. Without that, depth buys
# stall tolerance at the price of staleness, and this number could not go up.
QUEUE_SECONDS = 1.0


def find_input_device(match: str) -> int:
    """Index of the first input device whose name contains `match`.

    Matching on name rather than a fixed index because the ALSA card number
    moves when other USB audio is present — the mic came up as hw:2,0 here,
    but that is not stable across reboots or re-plugging.
    """
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and match in dev["name"]:
            log.info("microphone: [%d] %s", idx, dev["name"].strip())
            return idx
    raise RuntimeError(
        f"no input device matching {match!r}; "
        f"available: {[d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]}"
    )


class Microphone:
    """Continuous 16 kHz frame source.

    The device is opened at its own rate (48 kHz — it rejects 16 kHz outright)
    and decimated by exactly 3.

    Two things here are deliberate and were both learned the hard way:

    **The callback does no signal processing.** It copies the block into a
    queue and returns. An earlier version ran a scipy resampler inside the
    callback — the callback runs on a realtime audio thread with a hard
    deadline, and that is not something to do there. Resampling happens on the
    consumer side instead, where being slow costs latency rather than samples.

    **The block size is PortAudio's choice, not ours.** Asking this device for
    a fixed blocksize silently *loses audio* — measured over 5 s runs, the
    fraction of real time actually delivered was:

        blocksize 1440 (30 ms)   0.82        blocksize 4800   0.82
        blocksize 1440, low      0.78        blocksize 9600   0.71
        blocksize 0 (auto)       0.995   <-- and zero overflows

    Roughly a fifth of all audio was being dropped, with nothing failing
    loudly. So the stream is opened with `blocksize=0` and hands back blocks of
    whatever size it likes, which is why `frames()` below re-frames them into
    the fixed-size frames the VAD and wake word require.

    **What it likes is not one size.** Measured over 6 s this device delivered
    `{417: 423, 15: 91, 16: 57, 14: 37}` — mostly 8.7 ms blocks, but nearly a
    third of them a third of a millisecond. That is fine for re-framing and
    ruinous for anything that counts blocks and believes the number means
    something. The queue did: thirty-two blocks is 278 ms of audio when they
    are large and 9 ms when they are small, against a consumer that stalls for
    up to 286 ms while Vosk finalises. So the callback coalesces up to one
    frame's worth before queueing, which costs no latency — a frame is complete
    only when its last sample arrives, either way — and makes the queue depth
    mean a duration.

    **The anti-alias filter keeps its state between blocks.** Decimating
    without a low-pass folds everything above 8 kHz back down into the speech
    band; filtering each block independently instead introduces a discontinuity
    at every boundary, 33 times a second. Neither is visible in casual testing
    and both quietly degrade recognition, so the filter state (`_zi`) is
    carried across calls.

    **So does the decimation phase, for the same reason and at greater cost.**
    Taking every third sample of each block independently — `filtered[::3]` —
    is only correct when every block length divides by 3. This device's do not.
    Measured over 6 s it delivers `{417: 423, 15: 91, 16: 57, 14: 37}`, and
    while 417 is 3x139, the 14s and 16s are 15.5% of all blocks and shift the
    output sample grid by a sample or two every time. The shift is never
    corrected, so the stream picks up a phase discontinuity about sixteen times
    a second.

    That is not a subtle degradation. Signal-to-spurious ratio of a tone
    through this path, before and after carrying the phase:

                  per-block phase    carried phase
        300 Hz         29.0 dB          47.2 dB
        1 kHz          18.5 dB          90.7 dB
        3 kHz           7.2 dB         103.1 dB
        5 kHz           4.5 dB          94.0 dB

    At 3 kHz — where the consonants are — the artefacts were nearly as loud as
    the speech. It also ran the clock 0.21% fast, 7.6 s per hour. Every
    recognition threshold in this project predates the fix, so anything tuned
    against captured audio (the wake phrase variants above all) is worth
    re-measuring rather than trusting.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.device = find_input_device(cfg.device_match)

        # Holds capture-rate audio; the consumer downsamples. Bounding this by a
        # count of *blocks* bounds it by nothing useful, because the device does
        # not deliver blocks of one size — measured over 6 s it produced
        # {417: 423, 15: 91, 16: 57, 14: 37}. Thirty-two of those is 278 ms of
        # audio if they all happen to be large and 9 ms if they all happen to be
        # small, so the safety margin against a stalled consumer was a lottery,
        # and the small blocks kept winning it. The callback coalesces to one
        # frame's worth first, and the depth is then a duration.
        self._chunk = cfg.capture_block
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._q: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(4, round(QUEUE_SECONDS * cfg.capture_rate / self._chunk)))
        # Counted in samples, not blocks: a "block" is no longer a fixed amount
        # of anything, and milliseconds of lost audio is the number that means
        # something to whoever reads the log line.
        self._dropped_samples = 0
        self._stream: sd.InputStream | None = None
        self._decim = cfg.capture_rate // cfg.target_rate
        if cfg.capture_rate % cfg.target_rate:
            raise ValueError(
                f"capture rate {cfg.capture_rate} is not an integer multiple of "
                f"{cfg.target_rate}; pick a device rate that is"
            )

        # Corner at 7.2 kHz: comfortably below the 8 kHz Nyquist of the 16 kHz
        # output, and above the ~7 kHz that carries any useful speech energy.
        self._sos = butter(8, 7200, btype="low", fs=cfg.capture_rate, output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0
        # Index within the next block that continues the output sample grid.
        # See the class docstring: this is as load-bearing as `_zi`.
        self._phase = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Overflow means a consumer is too slow. Log rather than raise:
            # dropping audio is far better than killing the assistant's only
            # microphone.
            log.warning("audio status: %s", status)

        # copy() because PortAudio reuses this buffer after we return. Held
        # until there is a frame's worth, then queued as one piece — still only
        # copying, which is all this thread is allowed to do, but it means the
        # queue holds a predictable amount of time rather than a lottery.
        # Chunks come out between `_chunk` and `_chunk` plus one device block,
        # so the depth calculation is tight enough to trust.
        #
        # This costs no latency: a frame is only complete once its last sample
        # has arrived either way. Coalescing moves where the assembly happens,
        # not when it finishes.
        self._pending.append(indata[:, 0].copy())
        self._pending_samples += frames
        if self._pending_samples < self._chunk:
            return

        chunk = (self._pending[0] if len(self._pending) == 1
                 else np.concatenate(self._pending))
        self._pending.clear()
        self._pending_samples = 0
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            # Full means a consumer has been away longer than the queue is
            # deep, and something has to go. It must be the *oldest* audio, not
            # this chunk. Discarding the newest keeps a second of history and
            # throws away what is being said right now, which is backwards for
            # a system whose whole job is to notice that it is being spoken to
            # — and it is not a theoretical preference: the endpointer, handed
            # a stale second, finds speech in it, ends the utterance on it, and
            # returns having captured the wrong moment entirely.
            #
            # The queue fills on every turn, because nothing reads audio while
            # the STT request is in flight, so this path is ordinary rather
            # than exceptional. Nothing is logged from here: this is a realtime
            # thread and at the rate overruns arrive the logging cost more than
            # the fault. The consumer summarises instead.
            try:
                self._dropped_samples += len(self._q.get_nowait())
            except queue.Empty:
                pass  # a consumer beat us to it; there is room now either way
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                # Lost the race for the slot just freed. One chunk, next time.
                self._dropped_samples += len(chunk)

    def _downsample(self, block: np.ndarray) -> np.ndarray:
        filtered, self._zi = sosfilt(self._sos, block.astype(np.float32), zi=self._zi)
        # Keep one output grid across the whole stream rather than restarting it
        # per block. `_phase` is where this block's first kept sample sits; after
        # consuming `len(filtered)` samples the next one moves back by that much,
        # modulo the decimation factor.
        out = filtered[self._phase :: self._decim]
        self._phase = (self._phase - len(filtered)) % self._decim
        return out.astype(np.int16)

    def _reset_resampler(self) -> None:
        """Forget filter and phase state, for when the stream is discontinuous.

        Carrying either across a reopened device would apply state from before
        the gap to audio after it, which is the thing this whole mechanism
        exists to avoid within a stream.
        """
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._phase = 0

    def __enter__(self) -> "Microphone":
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.cfg.capture_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            blocksize=0,  # see the class docstring — anything else drops audio
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "capturing at %d Hz, decimating /%d to %d Hz",
            self.cfg.capture_rate, self._decim, self.cfg.target_rate,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _restart(self) -> bool:
        """Reopen the stream after it has stopped delivering."""
        log.warning("microphone stopped delivering audio; reopening")
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            log.debug("closing the dead stream failed", exc_info=True)
        self._stream = None
        try:
            self.device = find_input_device(self.cfg.device_match)
            self.__enter__()
            self._reset_resampler()
            return True
        except Exception as exc:
            log.error("could not reopen the microphone: %s", exc)
            return False

    def frames(self) -> Iterator[np.ndarray]:
        """Yield fixed-size 16 kHz int16 frames forever, oldest first.

        The stream hands back blocks of arbitrary length (see the class
        docstring), but webrtcvad accepts only exact 10/20/30 ms frames and
        openWakeWord wants a consistent chunk. So blocks are downsampled, then
        cut to `frame_samples` with the remainder carried into the next block —
        never padded or truncated, which would corrupt frame timing.
        """
        n = self.cfg.frame_samples
        residual = np.empty(0, dtype=np.int16)
        seen_dropped = self._dropped_samples
        last_get = time.monotonic()
        overrun_samples = 0
        last_report = last_get
        while True:
            try:
                # A bare get() blocks forever, which is how a dead input
                # stream presented as "the wake word works once and then
                # never again" — silent, with no error anywhere. Time out
                # and check instead, so a stall is recoverable and, more
                # importantly, visible in the journal.
                block = self._q.get(timeout=STALL_TIMEOUT_S)
            except queue.Empty:
                if not self._restart():
                    time.sleep(1.0)
                seen_dropped = self._dropped_samples
                last_get = last_report = time.monotonic()
                overrun_samples = 0
                continue

            now = time.monotonic()
            if self._dropped_samples > seen_dropped and now - last_get < CONSUMING_GAP_S:
                # Losing audio while somebody is actively reading it. See
                # CONSUMING_GAP_S — this is the overrun that shows up as the
                # wake word mysteriously missing, so it is worth a warning and
                # not a counter nobody reads. Counted here, said below.
                overrun_samples += self._dropped_samples - seen_dropped
            seen_dropped = self._dropped_samples
            last_get = now

            if overrun_samples and now - last_report >= OVERRUN_REPORT_S:
                log.warning("audio overrun: %.0f ms of audio lost while consuming, last %.0f s",
                            overrun_samples * 1000 / self.cfg.capture_rate,
                            now - last_report)
                overrun_samples = 0
                last_report = now

            residual = np.concatenate([residual, self._downsample(block)])
            while len(residual) >= n:
                yield residual[:n]
                residual = residual[n:]

    def drain(self) -> None:
        """Discard buffered audio and reset the drop counter.

        Called after speaking, so the assistant does not immediately transcribe
        its own reply leaking back through the microphone — and so the next
        turn starts from live audio rather than working through a backlog of
        whatever accumulated while it was busy.
        """
        discarded = 0
        while True:
            try:
                discarded += len(self._q.get_nowait())
            except queue.Empty:
                break
        if discarded or self._dropped_samples:
            log.debug("drained %.0f ms of buffered audio, %.0f ms dropped while busy",
                      discarded * 1000 / self.cfg.capture_rate,
                      self._dropped_samples * 1000 / self.cfg.capture_rate)
        self._dropped_samples = 0
