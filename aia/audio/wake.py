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
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from aia.core.config import WakeConfig
from aia.router.fast import normalise, similarity

log = logging.getLogger(__name__)

# openWakeWord's feature extractor is built around 80 ms of 16 kHz audio.
# Feeding it our 30 ms capture frames directly works but wastes work and
# blurs the detection boundary, so frames are buffered to this size.
CHUNK_SAMPLES = 1280

# How much of the running partial to scan for the wake phrase. Long enough to
# hold the phrase plus a few syllables either side, short enough that the
# window scan stays cheap when it runs on every frame.
SCAN_CHARS = 24


class WakeWord(ABC):
    """A wake-word engine. Fed 16 kHz int16 frames, says when it fired."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool:
        """Consume one frame; True on the rising edge of a detection."""

    def reset(self) -> None:
        """Forget accumulated state, so one utterance cannot re-trigger."""


class AlwaysOn(WakeWord):
    """No gating: the next speech is treated as a command.

    For bringing up the rest of the pipeline on a machine where no wake model
    is usable. Enable with AIA_NO_WAKE=1.
    """

    def detect(self, frame: np.ndarray) -> bool:
        return True


def _bundled_model_path(name: str) -> str:
    """Resolve a wake phrase to the ONNX file shipped inside openwakeword.

    The package bundles its pretrained models rather than downloading them, and
    takes explicit paths — not names — so this does the lookup. Filenames carry
    a version suffix (`hey_jarvis_v0.1.onnx`), hence the prefix match.
    """
    import openwakeword.model

    directory = Path(openwakeword.model.__file__).parent / "resources" / "models"
    exact = directory / f"{name}.onnx"
    if exact.exists():
        return str(exact)
    for candidate in sorted(directory.glob(f"{name}_v*.onnx")):
        return str(candidate)

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
            if top >= self.cfg.threshold:
                if not self._latched:
                    self._latched = True
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
        # The phrase plus any spellings previously observed from this speaker.
        # All are compared by sound, so the list only needs to cover genuinely
        # different *pronunciations*, not different characters.
        self._targets = tuple(dict.fromkeys((cfg.phrase, *cfg.variants)))
        log.info(
            "wake phrase %r via Vosk, matching by sound at >= %.2f (targets: %s), "
            "model loaded in %.0f ms",
            cfg.phrase, cfg.similarity, "/".join(self._targets),
            (time.monotonic() - t0) * 1000,
        )

    def _heard(self) -> str:
        # Vosk inserts spaces between tokens; the phrase is matched without
        # them so tokenisation differences ("小爱 同学" vs "小爱同学") do not
        # cause a miss.
        partial = self._json.loads(self._rec.PartialResult()).get("partial", "")
        return partial.replace(" ", "")

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
        sits. Only the recent text is scanned, so this stays cheap at 33 Hz.
        """
        if not text:
            return False, 0.0, ""
        text = text[-SCAN_CHARS:]
        best = 0.0
        best_window = ""
        for target in self._targets:
            span = len(target)
            for width in (span - 1, span, span + 1):
                if width <= 0:
                    continue
                if width > len(text):
                    # Text shorter than any window: compare it whole rather
                    # than skipping. Without this a two-character result like
                    # 同学 scored a flat 0.00 — reported as "nothing like the
                    # wake word" when it is in fact most of it.
                    score = similarity(normalise(text), normalise(target))
                    if score > best:
                        best, best_window = score, text
                    continue
                for start in range(len(text) - width + 1):
                    window = text[start:start + width]
                    score = similarity(normalise(window), normalise(target))
                    if score > best:
                        best, best_window = score, window
                        if best >= 1.0:
                            return True, best, best_window
        return best >= self.cfg.similarity, best, best_window

    def detect(self, frame: np.ndarray) -> bool:
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


def build(cfg: WakeConfig) -> WakeWord:
    if cfg.disabled:
        log.warning("wake word DISABLED (AIA_NO_WAKE=1) — any speech is a command")
        return AlwaysOn()
    if cfg.backend == "porcupine":
        return Porcupine(cfg)
    if cfg.backend == "openwakeword":
        return OpenWakeWord(cfg)
    return VoskWakeWord(cfg)
