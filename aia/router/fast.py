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
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import ceil

from aia.plugins.base import CommandSpec, Plugin, Registry

log = logging.getLogger(__name__)

SLOT = re.compile(r"\{(\w+)\}")

# How alike an utterance and a trigger must be before the utterance counts as
# that trigger with its argument left off. See `_is_bare_trigger`.
#
# Derived, after being guessed once from a single counter-example. A trigger
# of length t swallows an argument of length x while 2t / (2t + x) >= k, so
# x <= 2t(1 - k)/k. At 0.95 that is 1.1 to 1.8 normalised characters across
# the triggers actually declared, worst at 把音量调到 (1.8). Every real
# argument is longer, because one Chinese syllable is at least two letters of
# pinyin: 五 is "wu", 我 is "wo". Confirmed on the device — the tightest cases
# keep their argument: 把音量调到五 -> volume "五", 搜索歌曲我 -> search_song
# "我", 播放我 -> play "我".
#
# The margin is real but thin, about 0.2 characters at worst. Raising this
# loses bare-trigger detection; lowering it starts eating one-syllable
# arguments. Re-derive rather than nudge.
BARE_TRIGGER = 0.95

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
        self._validate_phrases()
        self._whole_cache: list[str] | None = None
        self._whole_lengths: list[int] = []
        self._trigger_cache: list[str] | None = None

    def match(self, text: str) -> Intent | None:
        """The best command this utterance matches, or None.

        Takes no language. It used to accept one and never read it, which
        implied a scoping that does not happen — every phrase in every
        language is always considered, and that is deliberate: it is what
        lets "帮我 search the weather" match at all, and what lets one
        command follow another in a different language. Verified identical
        for "en" and "zh" across the routing corpus before the argument was
        removed.
        """
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
                len(normalise(SLOT.sub("", intent.matched))),
            )

        candidates: list[Intent] = []
        for plugin, command in self.registry.all_commands():
            for phrases in command.phrases.values():
                for phrase in phrases:
                    found = self._match_phrase(plugin, command, phrase, text, target)
                    if found:
                        candidates.append(found)

        # Rank first, then ask whether the argument is really an argument,
        # rather than asking it of every candidate as it is built. Only the
        # winner's answer is ever used, and asking in rank order brings the
        # question from 19 times an utterance down to 0.8. Losing candidates
        # are reached only when the winner turns out to have eaten a
        # command — which is precisely when the runner-up is what was meant.
        best = max(candidates, key=rank, default=None)
        while best is not None and best.arguments and self._is_command(
                next(iter(best.arguments.values()))):
            log.debug("%s took a command as its argument; passing over it", best)
            candidates.remove(best)
            best = max(candidates, key=rank, default=None)

        if best is None:
            return None

        # A bare trigger is a command with its argument missing, not a
        # command whose argument is the rest of itself. "搜索歌曲" is the
        # whole of `search_song`'s trigger and nothing else, and the only
        # match it can produce is the shorter 搜索 eating 歌曲 and calling
        # it a song name — so the app is sent looking for a song called
        # "song". Declining hands it to the slow path, which can ask which
        # song, and that is the answer the speaker was about to give.
        if best.command.takes_argument and self._is_bare_trigger(target):
            log.debug("%r is a trigger with no argument; declining", text)
            return None

        floor = self.argument_threshold if best.command.takes_argument else self.threshold
        if best.command.min_score is not None:
            floor = max(floor, best.command.min_score)
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

    def _validate_phrases(self) -> None:
        """Complain once, at startup, about phrases that cannot work.

        A slot always runs to the end of the utterance, so anything written
        after it is silently unmatchable. That used to be reported from
        `_match_phrase`, which runs for every phrase on every utterance — so
        one bad declaration would have logged a warning per phrase per turn
        for the life of the process, about something that cannot change while
        it runs. It is a declaration error, and this is where declarations
        are read.
        """
        for _, command in self.registry.all_commands():
            for phrases in command.phrases.values():
                for phrase in phrases:
                    slot = SLOT.search(phrase)
                    if slot and phrase[slot.end():].strip():
                        log.warning(
                            "%s: phrase %r has text after its slot, which can "
                            "never match — a slot runs to the end of the "
                            "utterance. That part is ignored.",
                            command.name, phrase)

    def _whole_phrases(self) -> list[str]:
        """Every phrase that is a complete command on its own, normalised.

        Built once and kept sorted by length, which is what lets
        `_resembles` skip most of them without looking at them. The registry
        is fixed for the life of the process — it is built from the plugin
        list at startup — so rebuilding this would re-run pinyin over all
        ~145 phrases to reach an answer that cannot have changed.
        """
        if self._whole_cache is None:
            self._whole_cache = sorted(
                (normalise(phrase)
                 for _, command in self.registry.all_commands()
                 if not command.takes_argument
                 for phrases in command.phrases.values()
                 for phrase in phrases),
                key=len,
            )
            self._whole_lengths = [len(p) for p in self._whole_cache]
        return self._whole_cache

    def _triggers(self) -> list[str]:
        """The lead-in of every phrase that takes an argument, normalised."""
        if self._trigger_cache is None:
            self._trigger_cache = []
            for _, command in self.registry.all_commands():
                for phrases in command.phrases.values():
                    for phrase in phrases:
                        slot = SLOT.search(phrase)
                        if slot is None:
                            continue
                        folded = normalise(phrase[: slot.start()])
                        if folded:
                            self._trigger_cache.append(folded)
        return self._trigger_cache

    def _is_bare_trigger(self, target: str) -> bool:
        """Is the whole utterance just some command's trigger and nothing more?

        Near-identity, not the ordinary argument threshold. A short argument
        moves the ratio very little — 查找歌曲夜曲 against the trigger 查找歌曲
        is 0.85, comfortably over the 0.82 an argument match needs — so
        anything looser than this throws away real commands to catch the
        empty ones. At 0.95 the utterance has to be the trigger with at most
        a syllable's worth of difference, which is what "no argument" means.
        """
        return any(similarity(target, trigger) >= BARE_TRIGGER
                   for trigger in self._triggers())

    def _is_command(self, tail: str) -> bool:
        """Is this "argument" just the name of another command?

        A slot runs to the end of the utterance, so a short trigger will
        happily eat the rest of a longer command and call it a query: 搜索
        matches the front of 搜索歌词 and hands back 歌词, which is how
        "search lyric" ends up opening the Song Search window and searching
        for the word "lyrics" — the one confusion these commands must never
        have. A query is a thing out in the world; 歌词 is a button on this
        screen. When the tail is itself a command, the split was wrong, and
        declining lets the whole-utterance match win instead.

        Only whole-utterance commands count. Comparing against triggers too
        would reject "play the song called stop it" for containing a verb.
        """
        folded = normalise(tail)
        return bool(folded) and self._resembles(folded)

    def _resembles(self, folded: str) -> bool:
        """Does `folded` reach the threshold against any whole phrase?

        The same answer as a plain `similarity()` loop over every phrase,
        reached without doing most of the work: there are ~145 phrases to
        try and `ratio()` is quadratic in the length of what it compares.

        Two cuts. First, length: `ratio()` is `2 * matches / (len(a) +
        len(b))` and `matches` cannot exceed the shorter string, so only
        phrases within a band around `len(folded)` can reach the threshold
        at all — at 0.78 that band is 0.64x to 1.56x, which is 69 of the 145
        phrases for a typical tail, and they are kept sorted by length so it
        is two bisections rather than a scan. Second, within the band,
        difflib's own linear upper bounds reject a phrase by counting
        characters before anything aligns them, and the matcher is built
        once with the tail as `b` so difflib's b-side index is built once
        rather than per phrase.

        Worth being honest about the size of this. Per utterance over the
        routing corpus, each configuration in its own interpreter and
        repeated: 5.5-6.0 ms with no guard at all, 5.8-6.2 ms as it stands,
        7.0-7.4 ms with a plain `ratio()` against every phrase. Identical
        verdicts in all three. So the guards themselves are not measurably
        more expensive than not having them — those two ranges overlap and
        nothing should be claimed from the difference — while the plain
        loop is consistently about a millisecond worse. On a 2.5 s budget
        none of it matters; this is written this way because it is no harder
        to read, not because the router was short of time.

        Measure in separate processes if you revisit this. Timing several
        configurations in one interpreter reported 5.7 ms for whichever ran
        first and ~38 ms for every one after it, which reads exactly like a
        catastrophic regression in whatever you happened to measure second.
        """
        phrases = self._whole_phrases()
        length = len(folded)
        lo = bisect_left(self._whole_lengths, ceil(self.threshold * length / (2 - self.threshold)))
        hi = bisect_right(self._whole_lengths, int((2 - self.threshold) * length / self.threshold))
        if lo >= hi:
            return False

        matcher = SequenceMatcher(None, "", folded)
        for phrase in phrases[lo:hi]:
            matcher.set_seq1(phrase)
            if (matcher.real_quick_ratio() >= self.threshold
                    and matcher.quick_ratio() >= self.threshold
                    and matcher.ratio() >= self.threshold):
                return True
        return False

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
