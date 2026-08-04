"""Speech to text, against a resident whisper.cpp server.

Why a server rather than shelling out to whisper-cli per utterance: the model
has to stay in memory. Loading `ggml-base` costs ~96 ms and Piper's voice up to
1190 ms — paying either on the critical path blows the budget. The server also
isolates a crash in native code from the assistant process.

## Language handling

**Every utterance is detected on its own.** Say "play some music", then
"播放音乐", then "play some music" again, and each is transcribed in the
language it was spoken in. Nothing has to be selected, and there is no mode
to be in.

It did not used to work that way. A conversation detected its language once
and then named it for everything after, on the reasoning that people do not
change language mid-conversation and that naming it is faster. Both halves
were wrong for this assistant. People here *do* switch — that is the whole
point of a bilingual household — and the lock turned the second language into
a failure that lasted five minutes.

The cost of being wrong is not a slightly worse transcript. Whisper does not
fail when handed audio in a language it was not asked for; it quietly
translates. Measured on 35 real captures, all Mandarin, forced through
`language=en`:

    下一首   -> 'Next one.'        关机 -> 'Guanji.'
    前一首的 -> 'Money, brother.'  播放… -> '(Song)'
                                        -> '(speaking in foreign language)'

Fluent, confident, and unroutable. That is what a user hits on their first
Mandarin command after an English one, and it is the reported symptom.

## Why the lock was safe to remove

It was containment for a real problem: automatic detection ranges over all 99
languages Whisper knows, and on this device it used to return Korean (총치)
and Japanese (よいしょ, じゃあ) for Mandarin speech. That was measured on audio
captured through the broken decimator. Re-measured on 35 captures taken after
the phase fix: **0 decoded into an unsupported language.** The failure the
lock existed to contain was a symptom of the capture bug, and it went with it.
`detect_script(...) == "other"` stays as the net, and now catches nothing.

## What it costs, and what was tried instead

Measured over those same 35 captures, `ggml-base`, `-ac 512`, scored by
whether the transcript routes to a command — the only accuracy metric that
means anything here:

    base   auto                  1323 ms median   27-28/35 routed
    base   language named         725 ms median
    base   q5_1 quantised        1663 ms median   28/35 routed
    small  q5_1 auto             6627 ms median   32/35 routed
    small  q5_1 language named   3403 ms median   32/35 routed
    base   auto + primed vocab  16862 ms median   27/35 routed

So detecting every utterance costs ~600 ms against naming one. Three attempts
to avoid paying it, all rejected on measurement:

  * **A bigger model.** `small` really is better — 32/35 against 28/35, which
    is the Mandarin accuracy everyone wants — and it is 6.6 s per utterance
    against a 2.5 s budget for the whole turn. Not close, even with the
    language named.
  * **Quantisation to buy that back.** `q5_1` is *slower* than `f16` on this
    ARM core, at identical accuracy. There is no trade here to make.
  * **Detecting on a short prefix, then naming the language.** `-ac 512` fixes
    the encoder at 512 positions no matter how short the clip is, so detecting
    on 1.5 s costs 1230 ms against 1322 ms for the whole utterance. Two passes
    would be 2013 ms — worse than simply asking for `auto` once.
  * **Priming the decoder with the command vocabulary.** 781 characters of
    known phrases took 16.9 s and routed one *fewer*.

`verbose_json` is a separate ~390 ms on top, for DTW word alignment. It is off
unless someone asks for confidence.
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
    """Transcription, detecting the spoken language on every utterance.

    Nothing is remembered between turns. A command in English followed by one
    in Mandarin followed by one in English is three independent detections,
    which is what lets the user switch without saying so. The module docstring
    has what that costs and the three cheaper schemes that were measured and
    rejected.

    The exception is a caller that genuinely knows the language, which is
    `listen(..., language=...)` — see there.
    """

    def __init__(self, cfg: SttConfig, rate: int = 16000):
        self.cfg = cfg
        self.rate = rate
        self._session = requests.Session()

    def _detect_with(self) -> str:
        """What to ask for when the caller has not named a language."""
        return "auto" if self.cfg.auto_detect else self.cfg.default_language

    def listen(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """Transcribe an utterance, detecting its language unless told.

        `language` names it outright, for a caller that knows better than any
        detector could. Answering a question is that case: the assistant has
        just asked something out loud in a particular language and is holding
        the floor for the reply, so the reply is in that language and there is
        nothing to detect.

        That path is not an optimisation, it is a correctness fix, and it must
        stay. A one-word confirmation is very little audio to identify a
        language from, and automatic detection rendered a spoken 确定 as
        'Trading' and as 'seting.' — English, twice, answering a Chinese
        question. Neither is a yes, so a shutdown the user had authorised was
        silently cancelled. Naming it is also ~600 ms cheaper.
        """
        wav = _wav_bytes(audio, self.rate)
        result = self._transcribe(wav, language or self._detect_with())

        # Decoded into a language this assistant does not support. Redo it in
        # a supported one rather than handing the router text it can never
        # match — the CJK neighbours are what Whisper reaches for on Mandarin,
        # so Chinese is the right second guess. This fired on audio captured
        # through the broken decimator and has not fired since; it is kept as
        # a net, not as a working part of the path.
        if detect_script(result.text) == "other":
            log.info("transcript %r is not a supported language; retrying as zh",
                     result.text[:20])
            result = self._transcribe(wav, "zh")

        # The script of what came back is a better answer than what was asked
        # for: `auto` reports nothing in the fast `json` mode, and a
        # code-switched "帮我 search the weather" should be replied to in
        # whichever language carries the sentence. See `detect_script`.
        script = detect_script(result.text)
        if script in self.cfg.supported_languages:
            result.language = script
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
