"""Microphone capture, resampled to what the models expect.

One process owns the microphone for the lifetime of the assistant. Everything
downstream — wake word, VAD, STT — reads 16 kHz mono int16 frames from the
queue this fills, so nothing else ever opens the device. Two consumers fighting
over an ALSA capture handle is the failure mode this design exists to prevent.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

from aia.core.config import AudioConfig, MicProfile

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

# How long to wait before reopening a microphone that has stopped delivering,
# and the ceiling that wait grows to. A device that has been unplugged cannot be
# retried back into existence, so the interval doubles rather than spinning at a
# fixed rate for as long as the assistant runs — but the ceiling stays low,
# because this is an appliance and it has to recover on its own when the cable
# goes back in.
RESTART_BACKOFF_S = 1.0
MAX_RESTART_BACKOFF_S = 30.0


_CARD_LINE = re.compile(r"^\s*(\d+)\s*\[([^\]]*)\]\s*:\s*(.*)$", re.M)
_HW_RE = re.compile(r"\(hw:(\d+),\d+\)")


def _live_cards() -> dict[int, str]:
    """ALSA card index -> description, read fresh from /proc/asound/cards.

    Deliberately never cached. This is the only view of the sound cards that is
    guaranteed to reflect a re-plug, which is what makes it the thing to check
    PortAudio's own list against. Empty anywhere without /proc/asound, and
    every caller treats "empty" as "cannot tell" rather than "no cards".
    """
    try:
        text = Path("/proc/asound/cards").read_text()
    except OSError:
        return {}
    return {int(m.group(1)): f"{m.group(2).strip()} — {m.group(3).strip()}"
            for m in _CARD_LINE.finditer(text)}


def _hw_card(name: str) -> int | None:
    """The ALSA card number PortAudio baked into a device name, if it has one.

    Names look like "USB PnP Sound Device: Audio (hw:3,0)". That number is a
    snapshot from enumeration time, not a live fact — comparing it against
    `_live_cards` is exactly how staleness is detected.
    """
    m = _HW_RE.search(name)
    return int(m.group(1)) if m else None


def _matching_inputs(match: str) -> list[tuple[int, dict]]:
    return [(idx, dev) for idx, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0 and match in dev["name"]]


def _candidates(profiles: Sequence[MicProfile]) -> list[tuple[int, dict, MicProfile]]:
    """Every plugged-in device matching any known profile, best first.

    Profiles are walked in order and their matches appended in that order, so
    the head of this list is the most preferred microphone that is actually
    present. That is the whole of the auto-detection: swap one known capsule
    for another and the next open finds it, with its own settings.
    """
    out: list[tuple[int, dict, MicProfile]] = []
    for prof in profiles:
        out.extend((idx, dev, prof) for idx, dev in _matching_inputs(prof.match))
    return out


def _stale(candidates: list[tuple[int, dict, MicProfile]]) -> bool:
    """Does PortAudio's device list disagree with the kernel's?

    Two ways it can, both seen on this Pi: a match naming a card that no longer
    exists (the mic was re-plugged and renumbered), or no match at all because
    the list predates the microphone being plugged in. Both leave the assistant
    retrying forever against a list that cannot come true.
    """
    live = _live_cards()
    if not live:
        return False
    if not candidates:
        return True
    return any(_hw_card(dev["name"]) not in live for _, dev, _p in candidates)


def _amixer_read(card: int, control: str) -> str:
    """Raw `amixer sget` output for one control, or "" if it cannot be read."""
    try:
        r = subprocess.run(["amixer", "-c", str(card), "sget", control],
                           capture_output=True, timeout=5, text=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


def _mixer_state(card: int) -> str:
    """What the `Mic` control actually reads back, for the log line.

    Applying a setting and reporting the setting you *asked for* is how a
    microphone ends up documented at a gain it is not at. Read it back.
    """
    m = re.search(r"Capture (\d+) \[\d+%\] \[(-?[\d.]+dB)\]", _amixer_read(card, "Mic"))
    return f"{m.group(1)} ({m.group(2)})" if m else "unreadable"


def _agc_state(card: int) -> str:
    """Whether Auto Gain Control is actually on, read back the same way.

    Worth surfacing next to the gain rather than reporting what the profile
    asked for. AGC winds gain into the rail on its own schedule, and a
    microphone near full scale cannot be endpointed — every utterance runs to
    `max_utterance_ms` because webrtcvad calls every frame speech. That is the
    single worst failure this subsystem has had, and "is AGC off?" is the first
    question to ask about it. A device that does not expose the control at all
    reads as unavailable, which is different from off.
    """
    out = _amixer_read(card, "Auto Gain Control")
    if not out:
        return "not available"
    m = re.search(r"\[(on|off)\]", out)
    return m.group(1) if m else "unreadable"


def _apply_profile(prof: MicProfile, card: int | None) -> None:
    """Put the microphone's mixer into the state it was measured in.

    Applied on every open, re-plugs included, because that is the point of a
    profile: ALSA's stored state is keyed per card and holds only what
    `alsactl store` last captured, so a capsule the system has never seen has
    no entry at all. This one arrived at 30/30 — 33.00 dB, deep into the range
    where proximity to full scale drives voiced% to 100% and the endpointer can
    never terminate.

    Best-effort throughout. A microphone that does not expose these controls,
    or a host without amixer, is not a reason to refuse to listen — it is a
    reason to say so once and carry on with whatever the mixer already had.
    """
    if card is None:
        return
    wanted = [("Mic", None if prof.gain is None else str(prof.gain))]
    if prof.agc is not None:
        wanted.append(("Auto Gain Control", "on" if prof.agc else "off"))

    for control, value in wanted:
        if value is None:
            continue
        try:
            r = subprocess.run(
                ["amixer", "-c", str(card), "-q", "sset", control, value],
                capture_output=True, timeout=5, text=True)
            if r.returncode != 0:
                log.warning("could not set %r to %s on card %d: %s", control,
                            value, card,
                            r.stderr.strip() or f"amixer exited {r.returncode}")
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not set %r on card %d: %s", control, card, exc)


def _refresh_portaudio() -> bool:
    """Rebuild PortAudio's cached device list. Says whether it worked.

    **Only safe where this process holds no open stream.** `_terminate()` tears
    down every stream PortAudio owns, input and output alike. `tts/piper.py`
    documents the reverse mistake in detail: refreshing to recover *output*
    destroyed the microphone, so the wake word fired exactly once per run.

    Called from one place, `_restart()`, after the dead input stream has been
    closed. A reply being spoken at that instant is the cost, and it is the
    same trade piper.py already reasoned through, pointing the other way: a
    lost sentence against an assistant that cannot hear again until someone
    restarts it.
    """
    try:
        sd._terminate()
        sd._initialize()
        return True
    except Exception:
        log.warning("could not rebuild PortAudio's device list", exc_info=True)
        return False


@dataclass(frozen=True)
class Selection:
    """The microphone that was actually opened.

    Kept as a value rather than left inside the log line it used to be
    formatted into. The settings UI has to report the microphone AIA is using,
    and re-deriving that later would re-run the whole preference walk and could
    easily answer with a *different* device than the one the open stream is
    reading from — which is the exact confusion this project already spent a
    long hunt on. Whatever `find_input_device` chose is recorded here and read
    from there afterwards.
    """

    index: int
    profile: MicProfile
    name: str
    card: int | None


def find_input_device(profiles: Sequence[MicProfile], allow_refresh: bool = False,
                      quiet: bool = False) -> Selection:
    """The most preferred microphone that is plugged in, as a `Selection`.

    Matching on name rather than a fixed index because the ALSA card number
    moves when other USB audio is present — the mic came up as hw:2,0 here,
    but that is not stable across reboots or re-plugging.

    `allow_refresh` permits rebuilding PortAudio's device list when this one is
    provably stale. It is only safe where no input stream is open; see
    `_refresh_portaudio`. `quiet` drops the staleness report to DEBUG, for the
    retry loop where it would otherwise repeat for as long as the microphone
    stays unplugged.
    """
    found = _candidates(profiles)

    # PortAudio enumerates once per process, so an index outlives the card
    # layout that produced it. After a re-plug the same index can point at a
    # different capsule — or at nothing. Measured here: the list still said
    # "USB PnP Sound Device: Audio (hw:2,0)" long after that card became
    # hw:3,0, so every reopen attempt asked ALSA for a card index 2 that no
    # longer existed ("Cannot get card index for 2") and the assistant stayed
    # deaf in a retry loop while a working microphone sat unused at hw:3,0.
    if allow_refresh and _stale(found):
        log.log(
            logging.DEBUG if quiet else logging.WARNING,
            "PortAudio's device list disagrees with the kernel — it offers %s "
            "for %s, live cards are %s; rebuilding it",
            [d["name"].strip() for _, d, _p in found] or "nothing",
            [p.match for p in profiles], sorted(_live_cards()),
        )
        if _refresh_portaudio():
            found = _candidates(profiles)

    if found:
        idx, dev, prof = found[0]
        # Say so when there was a choice. Two USB microphones enumerate with
        # names that differ only by a card number that moves on reboot and
        # re-plug. Preference order is as good a rule as any, but applying it
        # silently is what turned swapping a microphone into a long hunt: the
        # capture came from a different capsule with ~10 dB more gain, every
        # log line looked normal, and the only symptom was every utterance
        # running to the endpointer's cap.
        #
        # Still a warning rather than fatal. A wrong pick used to be
        # undetectable downstream, which is what argued for refusing to start;
        # now the resolved ALSA card is in the line below and can be checked
        # against /proc/asound. An appliance that has to recover unattended is
        # better off deaf-and-guessing than deaf-and-refusing.
        if len(found) > 1:
            log.warning(
                "%d microphones are plugged in; using %s. The others are %s — "
                "reorder AudioConfig.microphones if that is the wrong one.",
                len(found), dev["name"].strip(),
                [d["name"].strip() for _, d, _p in found[1:]],
            )
        card = _hw_card(dev["name"])
        _apply_profile(prof, card)
        log.info("microphone: [%d] %s (ALSA card %s: %s) — %s, gain now %s",
                 idx, dev["name"].strip(),
                 "?" if card is None else card,
                 _live_cards().get(card, "not in /proc/asound — list is stale"),
                 prof.note or prof.match,
                 "?" if card is None else _mixer_state(card))
        return Selection(idx, prof, dev["name"].strip(), card)

    # Not found is two different faults wearing the same face, and by far the
    # likelier one is that the device is fine and somebody else has it: ALSA
    # drops a card that is already open out of enumeration entirely, so
    # PortAudio reports no inputs at all rather than a busy one. Saying "no
    # input device" then sends the reader after the cable, the driver and the
    # card number, when the answer is `systemctl --user stop aia`. The card
    # stays listed in /proc/asound whether or not it is open, which is what
    # tells the two apart.
    busy = [p.match for p in profiles if _card_present(p.match)]
    if busy:
        raise RuntimeError(
            f"the microphone matching {busy[0]!r} exists but could not be opened — "
            "it is almost certainly already in use. The device allows one "
            "reader, and AIA is usually it: `systemctl --user stop aia`."
        )
    raise RuntimeError(
        f"no input device matching any of {[p.match for p in profiles]}; "
        f"available: {[d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]}"
    )


def _card_present(match: str) -> bool:
    """Is a sound card whose description contains `match` known to the kernel?

    Linux-only and best-effort — anywhere without /proc/asound this simply says
    no and the caller falls back to the generic message.
    """
    return any(match.lower() in desc.lower() for desc in _live_cards().values())


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
        self.selection = find_input_device(cfg.microphones)
        # Re-read on every reopen, so a re-plug that lands on a different
        # capsule is reflected wherever this is reported.
        self._described: tuple[float, dict] | None = None
        self._describe_lock = threading.Lock()

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
        self._drain_baseline = 0
        # Samples the anti-alias filter pushed past full scale. Worth surfacing
        # rather than silently flattening: it means the microphone gain is too
        # high or the speaker is too close, which is a thing a person can fix.
        self._clipped_samples = 0
        # Set by the audio thread, read and cleared by the consumer. See
        # `_callback` — logging from a realtime callback is not allowed here.
        self._status = None
        self._restart_failures = 0
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

    @property
    def device(self) -> int:
        return self.selection.index

    @property
    def profile(self) -> MicProfile:
        return self.selection.profile

    def describe(self) -> dict:
        """The microphone actually in use, for the settings page.

        Everything here is read from the open stream's own `Selection` and from
        the mixer itself — not from `AudioConfig`. A configured device name is
        not an answer to "which microphone is AIA using": two USB capsules are
        known to this project, either may be plugged in, the ALSA card number
        moves on every re-plug, and `_restart` can swap the selection out from
        under a running session.

        Cached briefly because two of these fields cost an `amixer` subprocess
        each, and a settings page that is left open should not spawn processes
        on a timer.
        """
        with self._describe_lock:
            now = time.monotonic()
            if self._described is not None and now - self._described[0] < 5.0:
                return self._described[1]

            sel = self.selection
            card = sel.card
            info = {
                "name": sel.name,
                "profile": sel.profile.note or sel.profile.match,
                "device_index": sel.index,
                "alsa_card": card,
                "card_description": (
                    _live_cards().get(card) if card is not None else None),
                "capture_rate": self.cfg.capture_rate,
                "sample_rate": self.cfg.target_rate,
                "channels": self.cfg.channels,
                "sample_format": self.cfg.dtype,
                "gain": "?" if card is None else _mixer_state(card),
                "agc": "?" if card is None else _agc_state(card),
                "configured_gain": sel.profile.gain,
                "configured_agc": sel.profile.agc,
                "streaming": self._stream is not None,
                # A running total, not a rate. Non-zero means the gain is too
                # high or the source too close, and it is the first thing to
                # look at when utterances stop endpointing.
                "clipped_samples": self._clipped_samples,
            }
            self._described = (now, info)
            return info

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Overflow means a consumer is too slow. Recorded rather than
            # logged: this is a realtime thread with a hard deadline, and
            # logging takes locks, formats strings and writes to the journal —
            # any of which can block on something another thread holds. It is
            # also self-reinforcing, because PortAudio raises these flags
            # precisely when the machine is already struggling, so the report
            # arrives as a burst and lengthens the stall it is reporting. The
            # consumer says it out loud instead.
            self._status = status

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
        # Clip, do not wrap. A low-pass overshoots on transients — measured,
        # this one reaches 136% of full scale on a full-scale square wave and
        # 100.4% on a plain full-scale tone — and `astype` on an out-of-range
        # float wraps, turning a sample near +32767 into one near -32768. That
        # is a full-amplitude sign inversion, i.e. a loud click, landing on
        # exactly the loudest phonemes. Not theoretical on this hardware: of 73
        # real captures the median peaks at 6776 but one already sits at 32767,
        # because the USB mic has no headroom management and a plosive from
        # close range reaches the rail.
        np.clip(out, -32768, 32767, out=out)
        clipped = int(np.count_nonzero(np.abs(out) >= 32767))
        if clipped:
            self._clipped_samples += clipped
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
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            # close() even if stop() throws. Skipping it leaks the ALSA handle
            # for the life of the process, and the device allows one reader —
            # so the next thing to want the microphone cannot have it.
            stream.stop()
        finally:
            stream.close()

    def _restart(self) -> bool:
        """Reopen the stream after it has stopped delivering.

        Failures are logged once, not once per attempt. A microphone that has
        been unplugged is not coming back on its own, and the retry loop below
        will keep trying for as long as the assistant runs — at one line per
        attempt that is thousands of identical entries in the journal, which
        buries whatever else went wrong that night.
        """
        first = self._restart_failures == 0
        log.log(logging.WARNING if first else logging.DEBUG,
                "microphone stopped delivering audio; reopening")
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            log.debug("closing the dead stream failed", exc_info=True)
        self._stream = None
        try:
            # Refresh is allowed here and nowhere else: the dead input stream
            # was just closed above, so there is none for `_terminate()` to
            # destroy. This is the path that could not recover a re-plug.
            #
            # Not on every attempt, though. The list has to be rebuilt for a
            # device that comes back under a new card number to be visible at
            # all, but rebuilding it once per retry churns PortAudio for as
            # long as the microphone stays unplugged — and piper.py explains
            # what tearing PortAudio down does to anything still playing. A
            # microphone that is gone is usually gone for minutes, so first
            # attempt and then occasionally is enough to catch it returning.
            refresh = first or self._restart_failures % 8 == 0
            # Re-selects rather than reopening the same index, so unplugging one
            # known microphone and plugging in another is recovered from here,
            # with the new capsule's own mixer settings applied.
            self.selection = find_input_device(
                self.cfg.microphones, allow_refresh=refresh, quiet=not first)
            # Whatever the settings page last reported describes the previous
            # capsule; it may not be this one.
            with self._describe_lock:
                self._described = None
            self.__enter__()
            self._reset_resampler()
            if self._restart_failures:
                log.warning("microphone recovered after %d failed attempts",
                            self._restart_failures)
            self._restart_failures = 0
            return True
        except Exception as exc:
            self._restart_failures += 1
            log.log(logging.ERROR if first else logging.DEBUG,
                    "could not reopen the microphone: %s", exc)
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
        seen_clipped = self._clipped_samples
        last_get = time.monotonic()
        overrun_samples = 0
        last_report = last_get
        backoff = RESTART_BACKOFF_S
        while True:
            try:
                # A bare get() blocks forever, which is how a dead input
                # stream presented as "the wake word works once and then
                # never again" — silent, with no error anywhere. Time out
                # and check instead, so a stall is recoverable and, more
                # importantly, visible in the journal.
                block = self._q.get(timeout=STALL_TIMEOUT_S)
            except queue.Empty:
                if self._restart():
                    backoff = RESTART_BACKOFF_S
                else:
                    # Back off rather than hammering. A missing microphone
                    # cannot be retried into existence, and at a fixed one
                    # second this spun for as long as the assistant ran.
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_RESTART_BACKOFF_S)
                # The new stream shares nothing with the old one, so neither
                # does the partial frame left over from it.
                residual = np.empty(0, dtype=np.int16)
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

            # Everything the audio thread is not allowed to say for itself gets
            # said here, on one interval, off the realtime path.
            if now - last_report >= OVERRUN_REPORT_S:
                if overrun_samples:
                    log.warning("audio overrun: %.0f ms of audio lost while consuming, last %.0f s",
                                overrun_samples * 1000 / self.cfg.capture_rate,
                                now - last_report)
                if self._status is not None:
                    status, self._status = self._status, None
                    log.warning("audio status: %s", status)
                if self._clipped_samples > seen_clipped:
                    log.warning("%d samples clipped at full scale in the last %.0f s — "
                                "the microphone gain is too high, or the source is too close",
                                self._clipped_samples - seen_clipped, now - last_report)
                seen_clipped = self._clipped_samples
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
        # The counter only ever climbs, and readers keep their own baseline.
        # Zeroing it from here was a read-modify-write racing the audio thread's
        # `+=`, which is not atomic — the occasional increment simply vanished.
        # It only ever cost an inaccurate diagnostic, but a counter that lies is
        # worse than no counter when it is the thing you are debugging with.
        dropped = self._dropped_samples - self._drain_baseline
        self._drain_baseline = self._dropped_samples
        if discarded or dropped:
            log.debug("drained %.0f ms of buffered audio, %.0f ms dropped while busy",
                      discarded * 1000 / self.cfg.capture_rate,
                      dropped * 1000 / self.cfg.capture_rate)
