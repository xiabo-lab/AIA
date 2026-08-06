"""What every speech-to-text backend has in common.

The assistant asks one question of this subsystem — *what did they say* — and
this module is the whole of the vocabulary that question is asked in. A backend
supplies `listen()`; everything else in AIA sees `Transcript` and nothing about
whisper.cpp, ONNX, or how the audio got there.

Splitting this out is not tidiness. The two backends differ in kind, not in
degree: one is an HTTP call to a resident server that must already be running,
the other is an in-process ONNX graph that has to be loaded exactly once. If
`main.py` knew which it had, every one of those differences would have leaked
into the voice loop.

**Language codes here mean "a language AIA can answer in", and there are two of
them.** That is not the same set the recogniser can *hear*: SenseVoice
distinguishes Cantonese from Mandarin and AIA cannot answer in Cantonese,
because Piper ships no `yue` voice. So a transcript carries both — `language`
for choosing a voice, `detected` for what was actually spoken. See
`Transcript.detected`, and `aia/tts/language.py` for where that decision lives.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class SttUnavailable(RuntimeError):
    """The backend cannot transcribe at all — no model, no server, no wheel.

    Distinct from a failed utterance. A failed utterance is an empty
    `Transcript` and the assistant apologises and carries on; this means the
    subsystem is not going to work at any point in the future without someone
    intervening, so it is raised out of `wait_ready()` at startup where there
    is a person watching, and never from the voice path.
    """


class Transcript:
    """What was said, in what language, and how long it took to find out."""

    __slots__ = ("text", "language", "ms", "confidence", "detected")

    def __init__(self, text: str, language: str, ms: float,
                 confidence: float | None = None, detected: str | None = None):
        self.text = text
        # The language to *answer* in: "en" or "zh". Not necessarily the
        # language that was spoken — see `detected`.
        self.language = language
        self.ms = ms
        # Mean per-word probability where the backend computes one, else None.
        # The spec asks STT to return a confidence score; whisper produces one
        # only in its slow verbose mode, and SenseVoice does not expose per-word
        # probabilities at all, so `None` is the honest answer more often than
        # not and callers must treat it as "unknown", never as "zero".
        self.confidence = confidence
        # The language the recogniser actually reported, before it was folded
        # onto a voice AIA owns: "zh", "yue", "en", or None when the backend
        # cannot tell. This is the only place Cantonese survives as itself, and
        # it exists so the journal can show that Cantonese was recognised as
        # Cantonese even though the reply came back in Mandarin.
        self.detected = detected

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def __repr__(self) -> str:
        conf = "-" if self.confidence is None else f"{self.confidence:.2f}"
        spoken = "" if self.detected in (None, self.language) else f"/{self.detected}"
        return f"<Transcript {self.language}{spoken} {self.ms:.0f}ms conf={conf} {self.text!r}>"


class SttBackend(ABC):
    """A speech recogniser, as the rest of AIA sees one.

    Three methods, and only the first is on the voice path. Implementations
    must hold their model in memory for the life of the process — loading it
    per utterance misses the latency budget outright, which is the reason the
    whisper backend is a server at all.
    """

    #: Shown on the settings page. Set by each implementation.
    name: str = "stt"

    @abstractmethod
    def listen(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """Transcribe one utterance of 16 kHz mono int16 audio.

        `language` names the spoken language for a caller that genuinely knows
        it, which is the confirmation prompt: the assistant has just asked a
        question out loud and is holding the floor for the answer, so there is
        nothing to detect. Every other caller passes None and gets detection.

        **Must not raise for anything to do with the audio.** No speech, 80 ms
        of speech, a click, silence — all of those are an empty `Transcript`,
        because the assistant's answer to them is to apologise and keep
        listening, not to end the turn in an exception. Only a backend that has
        genuinely stopped working may raise.
        """

    @abstractmethod
    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Get the model loaded, and say whether it worked.

        Called once at startup, before the microphone is open, so that a
        missing model is a clear line in the journal rather than a turn that
        fails silently minutes later. Returning False is fatal to startup;
        raising `SttUnavailable` is the same thing with a better message.
        """

    def close(self) -> None:
        """Release whatever was held. Safe to call twice, and on a failed load."""

    def describe(self) -> dict:
        """What the settings page shows for this backend."""
        return {"engine": self.name}


# Scripts that mean the language was detected wrongly. Whisper's automatic
# detection ranges over all 99 languages it knows, and on Mandarin speech it
# reaches for its CJK neighbours often enough to matter — real examples from
# this device: 총치 (Korean), よいしょ and じゃあ (Japanese). None of that can
# route, and none of it is a language this assistant supports.
#
# SenseVoice knows five languages rather than 99 and tags each utterance with
# the one it used, so it has a better answer available than the script of its
# own output. It still runs text through here, because ja and ko are two of its
# five and its Mandarin can land in either.
_FOREIGN_RANGES = (
    ("぀", "ヿ"),  # hiragana + katakana
    ("가", "힯"),  # hangul syllables
    ("ᄀ", "ᇿ"),  # hangul jamo
    ("Ѐ", "ӿ"),  # cyrillic
    ("฀", "๿"),  # thai
    ("؀", "ۿ"),  # arabic
    ("ऀ", "ॿ"),  # devanagari
)

# SenseVoice emits its metadata as inline tokens — `<|zh|><|NEUTRAL|>
# <|Speech|><|withitn|>`. sherpa-onnx parses them into separate fields and
# hands back clean text, but a model or a wrapper version that does not would
# otherwise put nineteen literal characters of angle brackets in front of every
# command, where they would defeat the router and read, in the journal, as the
# user having said something very strange.
_META_TOKEN = re.compile(r"<\|[^|>]*\|>")


def strip_meta(text: str) -> str:
    """Remove any `<|tag|>` markers a recogniser left in its own output."""
    return _META_TOKEN.sub("", text).strip()


def parse_lang_tag(raw: str | None) -> str | None:
    """Read the language code out of whatever the recogniser called it.

    sherpa-onnx reports SenseVoice's language as `"<|yue|>"` on current builds
    and as a bare `"yue"` on others, and older ones do not report it at all.

    This is not `strip_meta`, and confusing the two is a bug that has already
    happened here: `strip_meta("<|zh|>")` returns the empty string, because its
    job is to take tags *out* of a transcript. Applied to the language field it
    deletes the entire answer, and the failure is silent — every utterance
    reports no detected language, falls through to guessing from the script,
    and Cantonese quietly stops being distinguishable from Mandarin, which is
    the one thing this backend was chosen for.
    """
    if not raw:
        return None
    code = raw.strip().strip("<>|").strip().lower()
    return code or None


def detect_script(text: str) -> str | None:
    """Classify a transcript as `en`, `zh`, or `other`.

    `other` means the recogniser decoded in a language this assistant does not
    support — the caller re-runs the pass with an explicit language rather than
    handing the router text it can never match.

    Note a code-switched command like "Play 周杰伦" resolves to `en`: the
    carrier sentence is English and only the proper noun is not, so an English
    reply is the right one.

    Cantonese is `zh` here and that is correct for what this function is for.
    It reads *script*, and written Cantonese is Han — 帮我搵下歌词 is Han
    throughout. Telling Cantonese from Mandarin is the recogniser's job, and
    SenseVoice does it from the audio; this cannot be made to do it from the
    text and must not be asked to.

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


def as_float32(audio: np.ndarray) -> np.ndarray:
    """int16 PCM to the [-1, 1] float32 every ONNX recogniser wants.

    Scaled by 32768 rather than 32767. The asymmetry is deliberate and it is
    the reason this is a named function instead of one expression inlined at
    the call site: int16 runs to -32768, so dividing by 32767 puts the most
    negative sample at -1.00003 and any downstream clamp turns the loudest part
    of the loudest utterance into a flat top. Dividing by 32768 cannot overflow
    in either direction, at the cost of 0.003% of amplitude that nothing can
    hear.
    """
    if audio.dtype == np.float32:
        return audio
    return (audio.astype(np.float32) / 32768.0)
