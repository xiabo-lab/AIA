"""The fast path: match a transcript to a command without touching the LLM.

This exists because of one measurement. Qwen2.5 3B decodes at 5.67 tok/s on
this Pi, so a tool call plus a spoken reply is ~5.3 s of generation alone —
routing "暂停" through a language model could never meet the 2.5 s target. The
LLM's job is conversation; a known command should never reach it.

## Matching in pinyin, not characters

Transcripts are wrong in specific, predictable ways, and comparing characters
handles them badly. Measured against real output from this system:

    heard          intended       characters   pinyin
    不放歌曲        播放歌曲          0.75        0.90
    播放周结轮      播放周杰伦         0.60        1.00
    今天天气怎么样   播放歌曲          0.00        0.19

Whisper's mistakes in Mandarin are overwhelmingly *homophone* mistakes — it
picks the wrong characters for the right sounds. 周结轮 and 周杰伦 are both
`zhoujielun`, so in pinyin the error disappears completely, while genuinely
unrelated speech stays far away. Comparing characters would need a threshold
low enough (0.60) to accept nonsense.

This is also why the plan's "Mandarin proper nouns are the weak spot" risk is
less alarming than it looked: the failure is in the writing system, not the
recognition, and the fast path does not care how a name is spelled.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from aia.plugins.base import CommandSpec, Plugin, Registry

log = logging.getLogger(__name__)

SLOT = re.compile(r"\{(\w+)\}")

# Punctuation Whisper likes to add, which no spoken phrase contains.
_STRIP = re.compile(r"[\s,.!?;:'\"，。！？、；：…—\-]+")

try:
    from pypinyin import lazy_pinyin

    _HAS_PINYIN = True
except ImportError:  # pragma: no cover
    _HAS_PINYIN = False
    log.warning("pypinyin not installed — Mandarin matching will be much weaker")


def _has_han(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def normalise(text: str) -> str:
    """Fold a transcript to the form comparisons happen in.

    Chinese becomes toneless pinyin, so homophone transcription errors
    collapse. Everything else is lowercased and stripped of punctuation and
    spacing. Full-width characters are normalised first — Whisper emits
    full-width digits and punctuation in Chinese output.
    """
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _STRIP.sub("", text)
    if _HAS_PINYIN and _has_han(text):
        return "".join(lazy_pinyin(text))
    return text


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Intent:
    plugin: Plugin
    command: CommandSpec
    arguments: dict[str, str]
    score: float
    matched: str

    def __repr__(self) -> str:
        return (f"<Intent {self.plugin.name}.{self.command.name} "
                f"{self.arguments} score={self.score:.2f}>")


class FastRouter:
    """Phrase matcher built from the plugin registry.

    Deliberately dumb and deterministic: ~50 ms, no model, no network, and it
    either matches confidently or declines so the caller can fall back to the
    LLM. Ambiguity is not resolved here — that is what the slow path is for.
    """

    def __init__(self, registry: Registry, threshold: float = 0.78,
                 argument_threshold: float = 0.82):
        self.registry = registry
        # Whole-utterance commands ("暂停") are matched end to end, so a loose
        # threshold risks firing on unrelated speech.
        self.threshold = threshold
        # Commands with an argument match only their trigger, which is short —
        # 播放 is two syllables — so the bar is higher to compensate.
        self.argument_threshold = argument_threshold

    def match(self, text: str, language: str = "en") -> Intent | None:
        target = normalise(text)
        if not target:
            return None

        # Rank by score, then prefer a command that took no argument. Both
        # matter: "播放歌曲" is an exact whole-utterance match for `resume`
        # (1.00) *and* an exact trigger match for `play {query}` with the
        # argument "歌曲" (also 1.00). The first is right — it consumed the
        # whole utterance, while the second only matched its two-syllable
        # trigger and invented an argument out of the remainder. Whole-phrase
        # matches are more specific, so they win ties.
        # A longer trigger breaks the remaining ties: "音量调到{level}" and
        # "音量{level}" both match "音量调到五十" perfectly, but only the first
        # leaves a clean "五十" as the argument rather than "调到五十".
        def rank(intent: Intent) -> tuple[float, int, int]:
            return (
                intent.score,
                0 if intent.command.takes_argument else 1,
                len(SLOT.sub("", intent.matched)),
            )

        best: Intent | None = None
        for plugin, command in self.registry.all_commands():
            for lang, phrases in command.phrases.items():
                for phrase in phrases:
                    found = self._match_phrase(plugin, command, phrase, text, target)
                    if found and (best is None or rank(found) > rank(best)):
                        best = found

        if best is None:
            return None

        floor = self.argument_threshold if best.command.takes_argument else self.threshold
        if best.score < floor:
            log.debug("best fast-path candidate %s scored %.2f, below %.2f",
                      best, best.score, floor)
            return None

        log.info("fast path: %s", best)
        return best

    def _match_phrase(self, plugin: Plugin, command: CommandSpec, phrase: str,
                      raw: str, target: str) -> Intent | None:
        slot = SLOT.search(phrase)

        if slot is None:
            return Intent(plugin, command, {}, similarity(target, normalise(phrase)), phrase)

        # A slot always runs to the end of the utterance: speech has no
        # reliable delimiter for "the song name stops here", so anything after
        # the slot in a template is not matchable and is not allowed.
        trigger = phrase[: slot.start()]
        if phrase[slot.end():].strip():
            log.warning("phrase %r has text after its slot; ignoring that part", phrase)

        argument = self._split_argument(raw, trigger)
        if argument is None:
            return None
        head, tail = argument
        if not tail.strip():
            return None

        return Intent(
            plugin, command, {slot.group(1): tail.strip()},
            similarity(normalise(head), normalise(trigger)), phrase,
        )

    @staticmethod
    def _split_argument(raw: str, trigger: str) -> tuple[str, str] | None:
        """Split a transcript into (trigger-ish head, argument tail).

        Cutting by character count is wrong for English, where the trigger is
        whole words and the transcript may space them differently, and cutting
        by words is wrong for Chinese, which has no spaces. So each is handled
        in its own units.
        """
        raw = raw.strip()
        if _has_han(trigger):
            head_len = len(_STRIP.sub("", trigger))
            compact = _STRIP.sub("", raw)
            if len(compact) <= head_len:
                return None
            # The head is compared fuzzily, so an off-by-one from a dropped or
            # inserted syllable still scores well.
            return compact[:head_len], compact[head_len:]

        words = raw.split()
        trigger_words = trigger.split()
        if len(words) <= len(trigger_words):
            return None
        return " ".join(words[: len(trigger_words)]), " ".join(words[len(trigger_words):])
