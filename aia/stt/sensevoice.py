"""Speech to text with SenseVoiceSmall, in-process via sherpa-onnx.

The default backend. It replaces whisper.cpp for three reasons, in the order
they matter to the person standing in front of the device:

**It knows Cantonese.** Whisper does not, in any usable sense — ~49.5% CER, and
it fails by producing confident Mandarin-ish text rather than by refusing. That
is not a tuning problem, and no amount of latency work fixes it. SenseVoice is
trained on `yue` as one of its five languages and tags each utterance with the
one it used, so Cantonese is recognised *as Cantonese*.

**It is not autoregressive.** Whisper decodes one token at a time and pays for
its 30 s padded encoder window whether or not there is 30 s of audio — the
reason `audio_ctx=512` is the most load-bearing number in `config.py`. Sense-
Voice runs the encoder once and reads the whole transcript off in a single
non-autoregressive pass, so cost tracks the audio's real length instead of a
fixed window, and a two-syllable 暂停 costs about what two syllables should.

**It runs in this process.** No server, no port, no unit to be out of sync
with, and no startup ordering — which removes the failure mode that cost
hours: `aia-whisper.service` in a restart loop with the assistant sitting in
"waiting for whisper-server", from nothing worse than a stripped +x bit.

## What it gives up

Whisper's automatic detection ranges over 99 languages; SenseVoice knows five
(zh, en, ja, ko, yue). For this assistant that is a straight improvement — the
93 it does not know were all wrong answers here — but it is a smaller net.

There are no per-word probabilities, so `Transcript.confidence` is always
None on this backend. Whisper only produced one in `verbose_json` at a ~390 ms
cost and it was off by default, so nothing on the voice path loses anything;
`SttConfig.verbose` simply has no effect here.

## Language, and why the reply language is a different question

`language: auto` is the default and should stay. The whole point of this
household is that a Mandarin command follows an English one with nothing to
select, and naming a language outright is what broke that under Whisper — see
that module's docstring for what forcing the wrong one actually produces.

The recogniser can return `yue`. AIA cannot *answer* in Cantonese, because
Piper ships no `yue` voice and adding one is explicitly not part of this work.
So the two questions are kept apart: `Transcript.detected` carries what was
spoken, `Transcript.language` carries the voice to answer in, and `_REPLY_IN`
below is the one table that maps the first onto the second. Cantonese is
answered in Mandarin, deliberately and visibly, rather than by a silent
fallback that nobody would be able to find later.

Nothing is translated on the way through. 搜索 Taylor Swift 的歌词 reaches the
router exactly as it was said.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from aia.core.config import SenseVoiceConfig, SttConfig
from aia.stt.base import (
    SttBackend,
    SttUnavailable,
    Transcript,
    as_float32,
    detect_script,
    parse_lang_tag,
    strip_meta,
)

log = logging.getLogger(__name__)

# What language to answer in, for each language SenseVoice can report.
#
# `yue` -> `zh` is the decision this table exists to make visible: there is no
# Cantonese Piper voice, so a Cantonese command is answered in Mandarin. When a
# `yue` voice lands, this row and a file in `TtsConfig.voices` are the whole
# change — which is exactly what aia/tts/language.py's docstring asks for.
#
# `ja` and `ko` are here because SenseVoice can return them and AIA supports
# neither. They map to Mandarin for the same reason `detect_script` retries in
# Chinese: on this device, in this room, a `ja` or `ko` tag on a real utterance
# means Mandarin was misheard, not that somebody switched to Japanese.
_REPLY_IN = {
    "zh": "zh",
    "yue": "zh",
    "en": "en",
    "ja": "zh",
    "ko": "zh",
}

# Below this, there is nothing to transcribe and the feature extractor is being
# asked to build frames out of less than a phoneme. The endpointer already
# refuses to hand over anything under `min_speech_ms` (500 ms), so on the live
# path this never fires — it is here for `replay.py`, the test harness, and any
# future caller that has not got that guard in front of it.
_MIN_AUDIO_MS = 100


class SenseVoiceSTT(SttBackend):
    """SenseVoiceSmall INT8, loaded once and reused for the life of the process.

    The model is loaded on `wait_ready()` at startup, or lazily on the first
    `listen()` for a caller that skipped it. It is never loaded per utterance:
    that is the same architectural requirement that made whisper a server, and
    it is the reason this class holds a recogniser rather than a path.
    """

    name = "sherpa-onnx (SenseVoiceSmall)"

    def __init__(self, cfg: SttConfig, rate: int):
        # `rate` is required and is the truth about the samples, not a wish.
        # The whisper backend learned this the expensive way: it wrote this
        # number into a WAV header, and a header that disagreed with the
        # samples did not fail — it transcribed a pitch-shifted signal and
        # returned confident wrong text. Here it goes to `accept_waveform`,
        # which resamples on it, so the same lie would produce the same class
        # of silent nonsense.
        self.cfg = cfg
        self.sv: SenseVoiceConfig = cfg.sensevoice
        self.rate = rate
        self._recognizer = None
        self._load_ms: float | None = None
        # The voice loop is single-threaded, but the settings page is not on
        # it: `describe()` runs on the HTTP thread and can arrive during a
        # turn. A lock around the recogniser costs nothing uncontended and
        # removes the question entirely.
        self._lock = threading.Lock()

    # ── loading ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Build the recogniser. Raises SttUnavailable with a fixable message."""
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SttUnavailable(
                "sherpa-onnx is not installed. "
                "pip install -r requirements.txt (or: pip install sherpa-onnx)"
            ) from exc

        for label, path in (("model", self.sv.model), ("tokens", self.sv.tokens)):
            if not path.is_file():
                raise SttUnavailable(
                    f"SenseVoice {label} is missing at {path} — "
                    "fetch it with ./scripts/get_sensevoice.sh"
                )

        t0 = time.monotonic()
        # `language=""` is sherpa-onnx's spelling of automatic detection. The
        # config says "auto" because that is what a person writing a config
        # file would write, and this is the one place that has to know the two
        # are the same thing.
        language = "" if self.sv.language in ("auto", "", None) else self.sv.language
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.sv.model),
            tokens=str(self.sv.tokens),
            num_threads=self.sv.num_threads,
            use_itn=self.sv.use_itn,
            language=language,
            provider=self.sv.provider,
            debug=False,
        )
        self._load_ms = (time.monotonic() - t0) * 1000
        self._recognizer = recognizer
        log.info("SenseVoice %s loaded in %.0f ms (%d threads, language=%s)",
                 self.sv.model.name, self._load_ms, self.sv.num_threads,
                 language or "auto")

    def _ensure(self):
        with self._lock:
            if self._recognizer is None:
                self._load()
            return self._recognizer

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Load the model and transcribe silence once, at startup.

        `timeout` is accepted and ignored, and that is not an oversight: the
        whisper backend spends it waiting on a server that starts independently,
        and there is nothing here to wait *for*. Loading either works or raises
        something a person can act on.

        The silent probe is worth its ~50 ms. It forces the first ONNX
        allocation and the first feature-extractor pass at startup rather than
        inside the first real turn, which is where a cold start is charged
        against the 2500 ms budget and shows up as the assistant being slow
        exactly once, in a way nobody can reproduce afterwards.
        """
        try:
            self._ensure()
            probe = np.zeros(self.rate // 2, dtype=np.int16)
            t0 = time.monotonic()
            self._decode(probe)
            log.info("SenseVoice warm (%.0f ms on 500 ms of silence)",
                     (time.monotonic() - t0) * 1000)
            return True
        except SttUnavailable as exc:
            log.error("%s", exc)
            return False
        except Exception:
            log.exception("SenseVoice failed to start")
            return False

    # ── transcription ────────────────────────────────────────────────

    def _decode(self, audio: np.ndarray) -> tuple[str, str | None]:
        """One recognition pass. Returns (text, reported language)."""
        recognizer = self._ensure()
        samples = as_float32(audio)
        with self._lock:
            stream = recognizer.create_stream()
            # The true sample rate, not the model's. sherpa-onnx resamples when
            # they differ; AIA's capture already decimates 48 kHz to 16 kHz, so
            # they should not, and `Config.__post_init__` checks it.
            stream.accept_waveform(self.rate, samples)
            recognizer.decode_stream(stream)
            result = stream.result

        # sherpa-onnx parses SenseVoice's inline `<|zh|><|NEUTRAL|>...` markers
        # into fields and hands back clean text. `strip_meta` is belt and
        # braces for a version that does not — see its docstring for what the
        # alternative looks like in the journal.
        text = strip_meta(result.text or "")
        # `parse_lang_tag`, NOT `strip_meta` — see that function for why. The
        # two look interchangeable on a field like "<|yue|>" and one of them
        # returns the empty string.
        return text, parse_lang_tag(getattr(result, "lang", None))

    def listen(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """Transcribe an utterance. See `SttBackend.listen`.

        `language` is accepted for the confirmation path, which knows what
        language it asked its question in. It is a genuinely weaker hint here
        than it was under Whisper: this model does not translate audio into the
        language it was told to expect, so naming one cannot produce the
        '确定' -> 'Trading' failure that made that path a correctness fix. It is
        still honoured for the reply language, because a one-word answer is very
        little audio to identify a language from and the caller does know.

        Never raises for anything about the audio — an empty `Transcript` is
        how "nothing usable was said" is reported, and the assistant's response
        to that is to apologise and keep listening.
        """
        if audio is None or len(audio) == 0:
            log.info("stt: empty audio")
            return Transcript("", language or self.cfg.default_language, 0.0)

        duration_ms = 1000.0 * len(audio) / self.rate
        if duration_ms < _MIN_AUDIO_MS:
            log.info("stt: %.0f ms is too short to transcribe", duration_ms)
            return Transcript("", language or self.cfg.default_language, 0.0)

        t0 = time.monotonic()
        try:
            text, reported = self._decode(audio)
        except SttUnavailable:
            # The model went away underneath a running assistant, which should
            # not be survivable quietly — but it must not take the process
            # down mid-turn either. Logged loudly, reported as a dead turn.
            log.exception("SenseVoice is unavailable")
            return Transcript("", language or self.cfg.default_language, 0.0)
        except Exception:
            log.exception("SenseVoice transcription failed")
            return Transcript("", language or self.cfg.default_language, 0.0)
        ms = (time.monotonic() - t0) * 1000

        # Which language to answer in, best source first:
        #
        #   1. What the model reported. It decided this from the audio, which
        #      is the only place the answer actually is, and it is the only
        #      source that can say `yue`.
        #   2. What the caller named — it is holding the floor and knows.
        #   3. The script of the text, for a model that reported nothing.
        #
        # `detect_script` is still consulted third and not first: it reads a
        # code-switched "帮我 search the weather" as English, which is the right
        # reply language for a sentence carried in English, but it cannot see
        # past Han to tell Cantonese from Mandarin.
        lang = (
            _REPLY_IN.get(reported or "")
            or (language if language in self.cfg.supported_languages else None)
            or detect_script(text)
            or self.cfg.default_language
        )
        if lang not in self.cfg.supported_languages:
            # detect_script said "other" — Cyrillic, Thai, Devanagari. This
            # model does not know those languages, so this is not a
            # mis-detection to retry, it is noise that decoded into something.
            lang = self.cfg.default_language

        result = Transcript(text, lang, ms, confidence=None, detected=reported)
        log.info("stt %s (%.0f ms audio, RTF %.2f)",
                 result, duration_ms, ms / max(duration_ms, 1.0))
        return result

    # ── housekeeping ─────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._recognizer = None

    def describe(self) -> dict:
        return {
            "engine": self.name,
            "num_threads": self.sv.num_threads,
            "language": "detected per utterance" if self.sv.is_auto else self.sv.language,
            "recognised_languages": list(self.sv.recognised_languages),
            "use_itn": self.sv.use_itn,
            "provider": self.sv.provider,
            "loaded": self._recognizer is not None,
            "load_ms": None if self._load_ms is None else round(self._load_ms),
        }
