"""Speech to text, against a resident whisper.cpp server.

Why a server rather than shelling out to whisper-cli per utterance: the model
has to stay in memory. Loading `ggml-base` costs ~96 ms and Piper's voice up to
1190 ms — paying either on the critical path blows the budget. The server also
isolates a crash in native code from the assistant process.

## Language handling

Measured on this Pi 5, same clip, `-ac 512`:

    json + explicit language     499 ms (en)   624 ms (zh)
    json + auto                  877 ms (en)  1020 ms (zh)
    verbose_json + auto         1268 ms (en)  1407 ms (zh)

Naming the language is ~390 ms cheaper than `auto`, and `verbose_json` costs a
further ~390 ms because it runs DTW alignment for word-level timestamps.

A conversation therefore detects its language **once** and then names it. The
first version of this file tried a cleverer thing — transcribe in whatever was
spoken last, and re-run when that turned out to be wrong — and it was removed,
because nothing detects "wrong" cheaply:

  * **Script does not work as a wrongness test.** Whisper emits text in the
    script of the language you asked for. Mandarin audio forced through English
    came back as fluent English prose — it quietly *translated* rather than
    failing.
  * **Confidence does not work reliably.** Mean word probability did separate
    the bad case (0.17 vs 0.69), but only with temperature fallback enabled,
    which is also what made that pass take **10.4 s**. Disabling fallback made
    Whisper settle on a confident mistranslation instead (0.77, still 6.5 s).
    And reading confidence at all requires `verbose_json`, whose +390 ms is the
    entire saving.

What replaced it is not that scheme. There is no per-utterance guessing: the
language is fixed for the conversation the moment the first utterance is
understood, and only reconsidered after the conversation goes idle. That is
both what a conversation is, and — the reason it was actually needed —
containment. Automatic detection ranges over all 99 languages Whisper knows,
and on this device it returned Korean (총치) and Japanese (よいしょ, じゃあ) for
Mandarin speech. Naming the language removes the choice.

The one residual guess is `detect_script(...) == "other"`, which catches a
first utterance that was decoded into an unsupported language and redoes it.
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
    """Transcription, with the conversation's language held steady.

    The first utterance is detected automatically; everything after it is
    transcribed in that same language until the conversation goes idle. Two
    reasons, and the second is the one that matters:

    **It is what a conversation is.** People do not change language between
    one sentence and the next, and an assistant that re-decides every time
    will eventually decide wrong mid-exchange.

    **Automatic detection ranges over all 99 languages Whisper knows.** On
    this device it picked Korean and Japanese for Mandarin speech often
    enough to matter — 총치, よいしょ, じゃあ — none of which can route, and
    none of which the assistant supports. Naming the language removes the
    choice, and is also ~390 ms faster per utterance than `auto`.
    """

    def __init__(self, cfg: SttConfig, rate: int = 16000):
        self.cfg = cfg
        self.rate = rate
        self._session = requests.Session()
        # The conversation's language, once known.
        self.locked: str | None = None
        self._last_used = 0.0

    def _current_language(self) -> str:
        """The language to ask for, expiring the lock if idle."""
        if self.locked is None:
            return "auto" if self.cfg.auto_detect else self.cfg.default_language
        if (self.cfg.lock_timeout_s
                and time.monotonic() - self._last_used > self.cfg.lock_timeout_s):
            log.info("language lock on %r expired after %.0fs idle",
                     self.locked, time.monotonic() - self._last_used)
            self.locked = None
            return "auto" if self.cfg.auto_detect else self.cfg.default_language
        return self.locked

    def reset_language(self) -> None:
        """Forget the conversation's language, so the next turn re-detects."""
        if self.locked is not None:
            log.info("language lock on %r cleared", self.locked)
        self.locked = None

    def listen(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """Transcribe an utterance in the conversation's language.

        `language` names it outright for this one utterance, for the case where
        the caller genuinely knows better than the lock does. Answering a
        question is that case: the assistant has just asked something out loud
        in a particular language and is holding the floor for the reply, so the
        reply is in that language and there is nothing to detect.

        It matters because the alternative failed in the field. A one-word
        confirmation is very little audio to identify a language from, and with
        the lock idle, automatic detection rendered a spoken 确定 as 'Trading'
        and as 'seting.' — English, twice, for the answer to a Chinese
        question. Neither is a yes, so the shutdown was silently cancelled.
        Naming the language is also ~390 ms cheaper, per the measurements above.
        """
        wav = _wav_bytes(audio, self.rate)
        result = self._transcribe(wav, language or self._current_language())

        # Decoded in a language this assistant does not support. Redo it in a
        # supported one rather than handing the router text it can never
        # match — the CJK neighbours are what Whisper reaches for on Mandarin,
        # so Chinese is the right second guess when nothing else is known.
        if detect_script(result.text) == "other":
            retry_in = self.locked or "zh"
            log.info("transcript %r is not a supported language; retrying as %r",
                     result.text[:20], retry_in)
            result = self._transcribe(wav, retry_in)

        script = detect_script(result.text)
        if script in self.cfg.supported_languages:
            result.language = script
            if self.locked is None:
                self.locked = script
                log.info("conversation language locked to %r", script)
        self._last_used = time.monotonic()
        return result

    def _transcribe(self, wav: bytes, language: str) -> Transcript:
        fmt = "verbose_json" if self.cfg.verbose else "json"

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

        # Plain `json` carries no language field. `language` is what was
        # *asked for*, which for a locked conversation is already the answer;
        # when it was "auto", the script of the text says what Whisper chose.
        # `listen()` corrects this afterwards if the script disagrees.
        reported = _LANG_NAMES.get(str(payload.get("language", "")).lower())
        lang = (reported if reported in self.cfg.supported_languages else None)             or (language if language in self.cfg.supported_languages else None)             or detect_script(text) or self.cfg.default_language

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


# Scripts that mean the language was detected wrongly. Whisper's automatic
# detection ranges over all 99 languages it knows, and on Mandarin speech it
# reaches for its CJK neighbours often enough to matter — real examples from
# this device: 총치 (Korean), よいしょ and じゃあ (Japanese). None of that can
# route, and none of it is a language this assistant supports.
_FOREIGN_RANGES = (
    ("぀", "ヿ"),  # hiragana + katakana
    ("가", "힯"),  # hangul syllables
    ("ᄀ", "ᇿ"),  # hangul jamo
    ("Ѐ", "ӿ"),  # cyrillic
    ("฀", "๿"),  # thai
    ("؀", "ۿ"),  # arabic
    ("ऀ", "ॿ"),  # devanagari
)


def detect_script(text: str) -> str | None:
    """Classify a transcript as `en`, `zh`, or `other`.

    `other` means Whisper decoded in a language this assistant does not
    support — the caller re-runs the pass with an explicit language rather
    than handing the router text it can never match.

    Note a code-switched command like "Play 周杰伦" resolves to `en`: the
    carrier sentence is English and only the proper noun is not, so an English
    reply is the right one.

    Japanese written purely in kanji is indistinguishable from Chinese here,
    and deliberately so — it is Han either way, and the router matches by
    pinyin, so treating it as Chinese loses nothing.
    """
    if not text:
        return None
    if any(low <= ch <= high for ch in text for low, high in _FOREIGN_RANGES):
        return "other"
    han = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if han and han >= latin:
        return "zh"
    if latin:
        return "en"
    return None
