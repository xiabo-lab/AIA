"""Wake word detection.

This is the only component that runs continuously, so its cost is the
assistant's idle cost. Everything else in the pipeline is gated behind it.

No engine ships a pretrained Chinese wake word. openWakeWord bundles six
English phrases and nothing else; training a custom model needs a synthetic
corpus and a GPU. Porcupine does Mandarin properly but requires a Picovoice
access key, and 小艾同学 is not one of its four built-in Chinese keywords, so it
also needs a custom .ppn from their Console.

The default backend therefore takes a different route: run a small Vosk
recogniser continuously and look for the phrase in what it produces. The
honest trade, measured on this Pi (one core of four):

                       idle    during speech
    Vosk small-cn      6.1%       ~49%        any phrase, free, no account
    openWakeWord       5.6%        5.6%       six English phrases only
    Porcupine          0.6%        0.6%       purpose-built, needs a key

Idle cost is fine — a wake word's usual state is silence, and 6.1% of one core
matches what openWakeWord costs anyway. The asymmetry is the point: a real
wake-word model runs the same cheap network whatever it hears, while a
recogniser does full decoding work the moment anyone speaks. In a room with a
television on, this sits near 49% of a core indefinitely.

(Polling `PartialResult` less often does not help — measured across intervals
from 30 ms to 1 s, CPU was flat within noise. The cost is `AcceptWaveform`
decoding speech, not the partial hypothesis readout. So partials are read every
frame, which is what keeps wake latency down.)

The second cost is accuracy: a general recogniser transcribing everything will
accept background Mandarin far more readily than a model trained to fire on
exactly one phrase. The spec asks for background conversations to be ignored
and false activations minimised; that is the part an access key would buy.

What can be done without one is to stop treating similarity as the whole
answer. Scanning every window of the running partial gives ordinary speech a
great many chances to score well *somewhere*, and the phrase's own tail is the
worst offender — 同学 is two of the three syllables of the 爱同学 variant and
scored 0.875 against it, which is to say the assistant woke up whenever anyone
in the room said "classmate". A match must therefore also account for the
phrase's first syllable; `_covers_onset` is that rule, and it is what makes the
window scan safe rather than merely fast.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from aia.core.config import AudioConfig, WakeConfig
from aia.router.fast import normalise

log = logging.getLogger(__name__)

# openWakeWord's feature extractor is built around 80 ms of 16 kHz audio.
# Feeding it our 30 ms capture frames directly works but wastes work and
# blurs the detection boundary, so frames are buffered to this size.
CHUNK_SAMPLES = 1280

# How much of the running partial to scan for the wake phrase. Long enough to
# hold the phrase plus a few syllables either side, short enough that the
# window scan stays cheap when it runs on every frame.
SCAN_CHARS = 24

# Windows recur constantly as a partial grows a character at a time, so their
# pinyin is memoised. The key space is unbounded over hours of speech, hence a
# cap: the cache is emptied rather than allowed to grow.
PINYIN_CACHE_MAX = 4096

# A latched openWakeWord detection re-arms when scores fall clear of the
# threshold. If they never do — a score parked in the band just under it — the
# latch would hold forever and the wake word would be silently dead, so it also
# expires on time.
LATCH_TIMEOUT_S = 10.0

# RMS of an int16 frame below which AlwaysOn treats the room as quiet. Speech
# sits well above this; a silent room sits an order of magnitude below.
ALWAYS_ON_RMS = 200.0


class WakeWord(ABC):
    """A wake-word engine. Fed 16 kHz int16 frames, says when it fired."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool:
        """Consume one frame; True on the rising edge of a detection."""

    def reset(self) -> None:
        """Forget accumulated state, so one utterance cannot re-trigger."""

    def close(self) -> None:
        """Release anything the engine holds. Must tolerate being called twice."""


class AlwaysOn(WakeWord):
    """No gating: the next sound is treated as a command.

    For bringing up the rest of the pipeline on a machine where no wake model
    is usable. Enable with AIA_NO_WAKE=1.

    It still waits for *sound*. Returning True on every frame, silence
    included, put the main loop into a permanent cycle: open a turn, hear
    nothing for `max_wait_ms`, close it, open another. That ducked and
    un-ducked whatever was playing every four seconds forever, spawning
    playerctl processes the whole time — which made no-wake mode unusable for
    the one thing it exists to test.
    """

    def __init__(self) -> None:
        self._latched = False

    def detect(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if level < ALWAYS_ON_RMS:
            # Same hysteresis as the real backends: re-arm only once the room
            # is properly quiet, so one utterance cannot open two turns.
            if level < ALWAYS_ON_RMS * 0.5:
                self._latched = False
            return False
        if self._latched:
            return False
        self._latched = True
        log.debug("no-wake mode: level %.0f, treating this as a command", level)
        return True

    def reset(self) -> None:
        self._latched = False


def _bundled_model_path(name: str) -> str:
    """Resolve a wake phrase to the ONNX file shipped inside openwakeword.

    The package bundles its pretrained models rather than downloading them, and
    takes explicit paths — not names — so this does the lookup. Filenames carry
    a version suffix (`hey_jarvis_v0.1.onnx`), hence the prefix match.

    Versions are ordered numerically, not as text. Sorting the names as strings
    puts `_v0.10` before `_v0.2`, so the day a model gets a tenth revision the
    lookup would quietly pick an older file than the one asked for.
    """
    import openwakeword.model

    directory = Path(openwakeword.model.__file__).parent / "resources" / "models"
    exact = directory / f"{name}.onnx"
    if exact.exists():
        return str(exact)

    def version(path: Path) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in path.stem.rsplit("_v", 1)[-1].split("."))
        except ValueError:
            return ()

    candidates = sorted(directory.glob(f"{name}_v*.onnx"), key=version)
    if candidates:
        return str(candidates[-1])

    available = sorted(
        p.stem.rsplit("_v", 1)[0]
        for p in directory.glob("*.onnx")
        if p.stem not in ("embedding_model", "melspectrogram", "silero_vad")
    )
    raise RuntimeError(
        f"no bundled wake model {name!r}; available: {available}. "
        "There is no pretrained 'Hey AIA' — use Porcupine with an access key "
        "for that, or train a custom openWakeWord model."
    )


class OpenWakeWord(WakeWord):
    def __init__(self, cfg: WakeConfig):
        from openwakeword.model import Model

        self.cfg = cfg
        self._buf = np.empty(0, dtype=np.int16)
        # Firing again while the user is still saying the phrase would open a
        # second capture on top of the first. Latch until scores fall back.
        self._latched = False
        self._latched_at = 0.0

        path = _bundled_model_path(cfg.model)
        self.model = Model(wakeword_model_paths=[path])
        log.info("wake word %r via openWakeWord (%s), threshold %.2f",
                 cfg.model, Path(path).name, cfg.threshold)

    def detect(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame])
        fired = False
        while len(self._buf) >= CHUNK_SAMPLES:
            chunk, self._buf = self._buf[:CHUNK_SAMPLES], self._buf[CHUNK_SAMPLES:]
            scores = self.model.predict(chunk)
            top = max(scores.values()) if scores else 0.0
            # Checked on every chunk, not just on a high score: the case this
            # guards against is a score parked *between* the threshold and the
            # re-arm level, which reaches neither branch below and would
            # otherwise hold the latch — and the wake word — shut for good.
            if self._latched and time.monotonic() - self._latched_at >= LATCH_TIMEOUT_S:
                log.warning("wake latch held %.0f s without re-arming; clearing",
                            LATCH_TIMEOUT_S)
                self._latched = False
            if top >= self.cfg.threshold:
                if not self._latched:
                    self._latched = True
                    self._latched_at = time.monotonic()
                    fired = True
                    log.info("wake word fired (score %.2f)", top)
            elif top < self.cfg.threshold * 0.5:
                # Hysteresis: only re-arm once well clear of the threshold, so
                # a score hovering at the boundary cannot chatter.
                self._latched = False
        return fired

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.int16)
        self._latched = False
        # Zero the model's internal feature buffers, otherwise audio from
        # before the last command can still contribute to the next score.
        try:
            self.model.reset()
        except AttributeError:
            pass


@dataclass
class _Target:
    """One spelling of the wake phrase, reduced to what matching needs.

    `pinyin` is the sound compared against. `onset` is how many characters of
    it belong to the first syllable — the part that has to be genuinely heard
    for a match to mean anything (see `_covers_onset`). `chars` is the length
    of the original spelling, because the scan windows are cut from Han text
    and measured in characters, not in pinyin.
    """

    spelling: str
    pinyin: str
    onset: int
    chars: int
    # Pre-loaded with `pinyin` as the second sequence. SequenceMatcher indexes
    # whichever sequence is set as seq2 and caches that index, so holding one
    # matcher per target and only swapping seq1 does the indexing once at
    # startup instead of once per window per frame.
    matcher: SequenceMatcher = field(repr=False, compare=False)


def _make_target(spelling: str) -> _Target:
    pinyin = normalise(spelling)
    # One Han character is exactly one syllable, so the pinyin of the first
    # character is the onset. For a non-Han phrase this degrades to a single
    # character, which leaves the guard inert rather than wrong.
    onset = len(normalise(spelling[:1])) if spelling else 0
    return _Target(
        spelling=spelling,
        pinyin=pinyin,
        onset=min(onset, len(pinyin)),
        chars=len(spelling),
        matcher=SequenceMatcher(None, "", pinyin, autojunk=False),
    )


def _covers_onset(matcher: SequenceMatcher, onset: int) -> bool:
    """Did the alignment actually match the target's *first* syllable?

    This is the guard that separates the wake phrase from its own tail, and it
    is not optional. Similarity alone accepts far too much, because a window
    that is a clean substring of the target scores almost as well as the target
    itself:

        heard      against      ratio
        同学        爱同学        0.875   <- fires, and 同学 is everywhere
        老同学      爱同学        0.842
        位同学      爱同学        0.842
        同学们      爱同学        0.737

    All of them clear a 0.72 threshold. Raising the threshold does not fix it:
    the genuine reduced form 爱同学 has only three syllables, so anything loose
    enough to tolerate one dropped syllable necessarily accepts 同学 — which is
    two of those three. The distinguishing sound is the 爱 at the front, so the
    fix is to require that it was heard rather than to keep haggling over a
    number.

    `matcher` must have the target's pinyin as its second sequence, so the `b`
    offsets of the matching blocks index the target.
    """
    if onset <= 0:
        return True
    covered = 0
    for _, start, size in matcher.get_matching_blocks():
        covered += max(0, min(start + size, onset) - start)
        if covered >= onset:
            return True
    return False


class VoskWakeWord(WakeWord):
    """Continuous Mandarin recognition, matched against the wake phrase.

    Detection runs off *partial* results rather than final ones. Vosk only
    finalises an utterance once it hears trailing silence, so waiting for
    `Result()` would add the whole endpointing delay to the wake word — the one
    stage of the pipeline that is supposed to be instant. Partials update every
    frame, so the phrase is caught mid-sentence, which also means "小艾同学，
    播放音乐" spoken as one breath still triggers.
    """

    def __init__(self, cfg: WakeConfig, rate: int = 16000):
        import json

        from vosk import KaldiRecognizer, Model, SetLogLevel

        self._json = json
        self.cfg = cfg
        if not cfg.vosk_model.exists():
            raise RuntimeError(
                f"missing Vosk model at {cfg.vosk_model}. "
                "Fetch it with scripts/get_wake_model.sh"
            )

        # Vosk logs every grammar and lattice detail at default verbosity.
        SetLogLevel(-1)
        t0 = time.monotonic()
        self._model = Model(str(cfg.vosk_model))
        self._rec = KaldiRecognizer(self._model, rate)
        self._rate = rate
        self._KaldiRecognizer = KaldiRecognizer
        # True once audio has been fed since the last reset. Rebuilding the
        # recogniser is the only reliable way to clear it, but it is not free
        # and the main loop resets on several paths per turn, so a reset with
        # nothing to clear is skipped.
        self._dirty = False
        # Memoised window pinyin, and the last scanned tail with its verdict.
        # Both exist to keep the scan below off the critical path; see
        # `_matches`.
        self._pinyin: dict[str, str] = {}
        self._last_scan: str | None = None
        self._last_result: tuple[bool, float, str] = (False, 0.0, "")

        # The phrase plus any spellings previously observed from this speaker.
        # All are compared by sound, so the list only needs to cover genuinely
        # different *pronunciations*, not different characters — and so the
        # deduplication is by sound too: 小艾同学 and 小爱同学 are one target,
        # not two, and scanning both did the identical work twice on every
        # frame. Length is part of the key because the scan window widths come
        # from the spelling, not from the pinyin.
        targets: dict[tuple[int, str], _Target] = {}
        for spelling in (cfg.phrase, *cfg.variants):
            target = _make_target(spelling)
            targets.setdefault((target.chars, target.pinyin), target)
        self._targets = tuple(targets.values())

        log.info(
            "wake phrase %r via Vosk, matching by sound at >= %.2f "
            "(targets: %s; onset must be heard), model loaded in %.0f ms",
            cfg.phrase, cfg.similarity,
            "/".join(t.spelling for t in self._targets),
            (time.monotonic() - t0) * 1000,
        )

    def _heard(self) -> str:
        # Vosk inserts spaces between tokens; the phrase is matched without
        # them so tokenisation differences ("小爱 同学" vs "小爱同学") do not
        # cause a miss.
        partial = self._json.loads(self._rec.PartialResult()).get("partial", "")
        return partial.replace(" ", "")

    def _pinyin_of(self, window: str) -> str:
        """`normalise`, memoised.

        A partial grows a character at a time, so the same windows are
        converted again and again — the cache turns almost every lookup after
        the first frame of an utterance into a dict hit.
        """
        sound = self._pinyin.get(window)
        if sound is None:
            sound = normalise(window)
            if len(self._pinyin) >= PINYIN_CACHE_MAX:
                self._pinyin.clear()
            self._pinyin[window] = sound
        return sound

    def _matches(self, text: str) -> tuple[bool, float, str]:
        """Does the tail of `text` sound like the wake phrase?

        Exact string matching was the first implementation and it missed most
        of the time in real use: a recogniser returns whatever characters fit
        the sounds, and 小艾同学 comes back as 小爱同学, 小爱同雪, 消爱同学 and
        others depending on how it was said. Listing every spelling is a losing
        game.

        Comparing *pinyin* collapses the whole family — every one of those is
        `xiaoaitongxue` — which is the same reason the intent router matches by
        sound rather than by character.

        Every window is scanned, not just the tail. Tail-only looks sufficient
        because partials usually grow a character at a time, but Vosk can jump
        several at once, and then "小爱同学播放音乐" arrives whole and scores
        0.48 against its own tail. The phrase has to be findable wherever it
        sits.

        Scanning that many windows is what makes similarity alone unsafe: every
        extra window is another chance for ordinary speech to score well
        somewhere. So a window has to clear the threshold *and* account for the
        phrase's first syllable — `_covers_onset` has the measurements, and 同学
        is the word it exists to reject.

        The returned score is the best *accepted* one. A window that scored
        high but failed the onset guard is not reported as a near miss, because
        it is not one: no threshold change would ever accept it, and telling
        the tuning tool otherwise is how the threshold got talked downwards in
        the first place. The window returned when nothing is accepted is the
        closest thing heard, for diagnosis only.
        """
        if not text:
            return False, 0.0, ""
        text = text[-SCAN_CHARS:]
        if text == self._last_scan:
            # Partials update every frame but usually say the same thing. At
            # 33 Hz this skips most of the work below outright.
            return self._last_result

        best = 0.0
        best_window = ""
        near = 0.0
        near_window = ""
        threshold = self.cfg.similarity

        for target in self._targets:
            matcher = target.matcher
            for width in (target.chars - 1, target.chars, target.chars + 1):
                if width <= 0:
                    continue
                if width > len(text):
                    # Text shorter than any window: compare it whole rather
                    # than skipping. Without this a short result was reported
                    # as "nothing like the wake word" when it is in fact most
                    # of it. The onset guard still applies.
                    windows = (text,)
                else:
                    windows = tuple(text[s:s + width]
                                    for s in range(len(text) - width + 1))
                for window in windows:
                    sound = self._pinyin_of(window)
                    if not sound:
                        continue
                    matcher.set_seq1(sound)
                    score = matcher.ratio()
                    if score > near:
                        near, near_window = score, window
                    # get_matching_blocks() is only worth its cost on a window
                    # that would otherwise fire, which is a handful per hour.
                    if score < threshold or score <= best:
                        continue
                    if not _covers_onset(matcher, target.onset):
                        log.debug("ignoring %r (%.2f vs %s): onset not heard",
                                  window, score, target.spelling)
                        continue
                    best, best_window = score, window
                    if best >= 1.0:
                        result = (True, best, best_window)
                        self._last_scan, self._last_result = text, result
                        return result

        result = (best > 0.0 and best >= threshold, best, best_window or near_window)
        self._last_scan, self._last_result = text, result
        return result

    def detect(self, frame: np.ndarray) -> bool:
        self._dirty = True
        if self._rec.AcceptWaveform(frame.tobytes()):
            # Utterance finalised without the phrase appearing — drop it so the
            # recogniser starts clean rather than accumulating context.
            text = self._json.loads(self._rec.Result()).get("text", "").replace(" ", "")
        else:
            text = self._heard()

        hit, score, window = self._matches(text)
        if hit:
            log.info("wake phrase detected: heard %r (%.2f) in %r", window, score, text)
            self.reset()
            return True
        return False

    def reset(self) -> None:
        # The cached verdict has to go regardless: it is keyed on the scanned
        # text, and after a reset the same text is a fresh utterance.
        self._last_scan = None
        self._last_result = (False, 0.0, "")
        if not self._dirty:
            # Nothing has been fed since the last reset, so there is nothing to
            # clear. The main loop resets on several paths per turn — and
            # `detect` resets itself on a hit — so this was rebuilding the
            # recogniser two or three times per turn, landing in the gap
            # between the reply finishing and the assistant listening again.
            return
        self._dirty = False
        self._pinyin.clear()
        # A fresh recogniser is the reliable way to clear both the partial
        # hypothesis and the decoder state; Reset() leaves the phrase in the
        # partial on some builds, which re-fires immediately on the next frame.
        self._rec = self._KaldiRecognizer(self._model, self._rate)


class Porcupine(WakeWord):
    """Picovoice Porcupine — a purpose-built engine. Needs an access key.

    For Mandarin this needs two extra files beyond the key: the `zh` parameter
    file, and a keyword `.ppn` for the phrase. Only 你好, 咖啡, 水饺 and 豪猪 are
    built in, so anything else — 小艾同学 included — is generated in the
    Picovoice Console. `scripts/get_porcupine_zh.sh` fetches the parameter file.
    """

    def __init__(self, cfg: WakeConfig):
        import pvporcupine

        if not cfg.access_key:
            raise RuntimeError(
                "porcupine backend needs PICOVOICE_ACCESS_KEY; "
                "get one free at https://console.picovoice.ai"
            )
        missing = [p for p in (cfg.porcupine_params, cfg.porcupine_keyword) if not p.exists()]
        if missing:
            raise RuntimeError(
                f"porcupine is missing {[str(p) for p in missing]}. "
                "Run scripts/get_porcupine_zh.sh, and generate the .ppn for "
                f"{cfg.phrase!r} (Chinese, Raspberry Pi) at https://console.picovoice.ai"
            )

        self._p = pvporcupine.create(
            access_key=cfg.access_key,
            model_path=str(cfg.porcupine_params),
            keyword_paths=[str(cfg.porcupine_keyword)],
        )
        self._buf = np.empty(0, dtype=np.int16)
        log.info("wake phrase %r via Porcupine (zh)", cfg.phrase)

    def detect(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame])
        n = self._p.frame_length
        fired = False
        while len(self._buf) >= n:
            chunk, self._buf = self._buf[:n], self._buf[n:]
            if self._p.process(chunk) >= 0:
                fired = True
        return fired

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.int16)

    def close(self) -> None:
        # pvporcupine.create() allocates native resources that the garbage
        # collector knows nothing about.
        engine, self._p = self._p, None
        if engine is not None:
            engine.delete()


def build(wake: WakeConfig, audio: AudioConfig) -> WakeWord:
    """Construct the configured backend.

    `audio` is taken rather than assumed. VoskWakeWord has to be told the
    sample rate of the frames it will be fed, and a recogniser told the wrong
    rate does not fail — it decodes garbage, so the wake word simply stops
    working with nothing in the journal to say why. Reading the rate from the
    same config the microphone uses makes that unrepresentable.
    """
    if wake.disabled:
        log.warning("wake word DISABLED (AIA_NO_WAKE=1) — any speech is a command")
        return AlwaysOn()
    if wake.backend == "porcupine":
        return Porcupine(wake)
    if wake.backend == "openwakeword":
        return OpenWakeWord(wake)
    return VoskWakeWord(wake, audio.target_rate)
