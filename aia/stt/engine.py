"""Speech to text, against a resident whisper.cpp server.

Why a server rather than shelling out to whisper-cli per utterance: the model
has to stay in memory. Loading `ggml-base` costs ~96 ms and Piper's voice up to
1190 ms — paying either on the critical path blows the budget. The server also
isolates a crash in native code from the assistant process.

## Language handling, and a design that was tried and rejected

Whisper's automatic language detection costs a whole extra encoder pass.
Measured on this Pi 5, same clip, `-ac 512`:

    json + explicit language     499 ms (en)   624 ms (zh)
    json + auto                  877 ms (en)  1020 ms (zh)
    verbose_json + auto         1268 ms (en)  1407 ms (zh)

Auto-detect costs ~+390 ms, and `verbose_json` costs a further ~+390 ms because
it runs DTW alignment to produce word-level timestamps.

The obvious optimisation is to skip detection: transcribe in whatever language
the user spoke last, and only re-run when that turns out to be wrong. That was
built, measured, and removed, because nothing detects "wrong" cheaply enough:

  * **Script does not work.** Whisper emits text in the script of the language
    you asked for. Mandarin audio forced through English came back as fluent
    English prose — it quietly *translated* rather than failing.
  * **Confidence does not work reliably.** Mean word probability did separate
    the bad case (0.17 vs 0.69) — but only with temperature fallback enabled,
    which is also what made that pass take **10.4 s**. Disabling fallback to
    cap the latency made Whisper settle on a confident mistranslation instead
    (0.77, still 6.5 s), destroying the signal. And reading confidence at all
    requires `verbose_json`, whose +390 ms is the entire saving.

So the saving was ~390 ms, the worst case was a 6-10 s pass returning a fluent
translation of something the user never said, and the detector for it cost as
much as the thing it was avoiding. Auto-detect is correct, predictable, and
still leaves ~700 ms of headroom in the fast-path budget. Do not re-litigate
this without new measurements.
"""

from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np
import requests

from aia.core.config import SttConfig

log = logging.getLogger(__name__)


def _wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    """In-memory 16-bit mono WAV. The server wants a file upload, not raw PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


class Transcript:
    __slots__ = ("text", "language", "ms", "confidence")

    def __init__(self, text: str, language: str, ms: float, confidence: float | None = None):
        self.text = text
        self.language = language
        self.ms = ms
        # Mean per-word probability, or None when running in the fast `json`
        # mode that does not compute word timings. The spec asks STT to return
        # a confidence score; set SttConfig.verbose to get one, and read the
        # module docstring for what it costs.
        self.confidence = confidence

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def __repr__(self) -> str:
        conf = "-" if self.confidence is None else f"{self.confidence:.2f}"
        return f"<Transcript {self.language} {self.ms:.0f}ms conf={conf} {self.text!r}>"


# Whisper reports languages by English name in verbose_json ("chinese"), but
# everything else here uses ISO-ish short codes.
_LANG_NAMES = {"english": "en", "chinese": "zh"}


class SpeechToText:
    def __init__(self, cfg: SttConfig, rate: int = 16000):
        self.cfg = cfg
        self.rate = rate
        self._session = requests.Session()

    def listen(self, audio: np.ndarray) -> Transcript:
        """Transcribe an utterance, detecting the language automatically."""
        wav = _wav_bytes(audio, self.rate)
        fmt = "verbose_json" if self.cfg.verbose else "json"
        language = "auto" if self.cfg.auto_detect else self.cfg.default_language

        t0 = time.monotonic()
        resp = self._session.post(
            self.cfg.url,
            files={"file": ("utterance.wav", wav, "audio/wav")},
            data={"response_format": fmt, "language": language},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        ms = (time.monotonic() - t0) * 1000

        text = (payload.get("text") or "").strip()

        # Plain `json` carries no language field, so fall back to the script of
        # the text. That is reliable here precisely *because* the language was
        # auto-detected: Whisper has already decoded in the right language, so
        # the script it produced is the answer rather than a guess.
        reported = _LANG_NAMES.get(str(payload.get("language", "")).lower())
        lang = reported or detect_script(text) or self.cfg.default_language

        confidence = None
        if self.cfg.verbose:
            probs = [
                w["probability"]
                for seg in payload.get("segments", [])
                for w in seg.get("words", [])
                if "probability" in w
            ]
            confidence = sum(probs) / len(probs) if probs else None

        result = Transcript(text, lang, ms, confidence)
        log.info("stt %s", result)
        return result

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Block until the server answers, so startup order does not matter."""
        deadline = time.monotonic() + timeout
        silent = np.zeros(self.rate // 10, dtype=np.int16)
        while time.monotonic() < deadline:
            try:
                self._session.post(
                    self.cfg.url,
                    files={"file": ("probe.wav", _wav_bytes(silent, self.rate), "audio/wav")},
                    data={"response_format": "json", "language": self.cfg.default_language},
                    timeout=10,
                ).raise_for_status()
                return True
            except Exception:
                time.sleep(0.5)
        return False


def detect_script(text: str) -> str | None:
    """Crude English-vs-Chinese split on character ranges.

    Only has to separate Latin from Han, which is a far easier call than
    Cantonese from Mandarin — Whisper reports both of those as `zh`, which is
    part of why Cantonese is out of scope for this stage.

    Note that a code-switched command like "Play 周杰伦" resolves to `en`:
    the carrier sentence is English and only the proper noun is not, so an
    English reply is the right one.
    """
    if not text:
        return None
    han = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if han and han >= latin:
        return "zh"
    if latin:
        return "en"
    return None
