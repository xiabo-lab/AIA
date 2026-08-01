"""Microphone capture, resampled to what the models expect.

One process owns the microphone for the lifetime of the assistant. Everything
downstream — wake word, VAD, STT — reads 16 kHz mono int16 frames from the
queue this fills, so nothing else ever opens the device. Two consumers fighting
over an ALSA capture handle is the failure mode this design exists to prevent.
"""

from __future__ import annotations

import logging
import queue
from typing import Iterator

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

from aia.core.config import AudioConfig

log = logging.getLogger(__name__)


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

    **The anti-alias filter keeps its state between blocks.** Decimating
    without a low-pass folds everything above 8 kHz back down into the speech
    band; filtering each block independently instead introduces a discontinuity
    at every boundary, 33 times a second. Neither is visible in casual testing
    and both quietly degrade recognition, so the filter state (`_zi`) is
    carried across calls.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.device = find_input_device(cfg.device_match)
        # Holds raw capture-rate blocks; the consumer downsamples. Deliberately
        # shallow: a wake-word system always wants near-live audio, and a deep
        # queue just means that after a busy turn the next one is spent
        # chewing through stale backlog instead of listening.
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._dropped = 0
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

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Overflow means a consumer is too slow. Log rather than raise:
            # dropping a block is far better than killing the assistant's only
            # microphone.
            log.warning("audio status: %s", status)
        try:
            # copy() because PortAudio reuses this buffer after we return.
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # The queue fills on every turn — nothing consumes audio while the
            # STT request is in flight — so this is normal, not exceptional.
            # It is summarised on drain rather than logged per block: at ~47
            # blocks a second it produced thousands of lines per turn, which
            # buried the actual failure in the journal and cost real I/O in
            # the middle of the latency budget.
            self._dropped += 1

    def _downsample(self, block: np.ndarray) -> np.ndarray:
        filtered, self._zi = sosfilt(self._sos, block.astype(np.float32), zi=self._zi)
        return filtered[:: self._decim].astype(np.int16)

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
        while True:
            residual = np.concatenate([residual, self._downsample(self._q.get())])
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
                self._q.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        if discarded or self._dropped:
            log.debug("drained %d buffered blocks, %d dropped while busy",
                      discarded, self._dropped)
        self._dropped = 0
