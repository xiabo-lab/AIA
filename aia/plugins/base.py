"""The plugin contract.

An application exposes itself to AIA as a list of commands, each with the
phrases that invoke it. That one declaration feeds both halves of the router:
the fast matcher compiles the phrases into patterns, and the LLM (M2) will
receive the same list as a tool schema. One source of truth, so a command
cannot work when spoken plainly but be invisible to the model, or vice versa.

Adding an application means dropping a module in here. Nothing in core/ or
router/ knows what a music player is.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandSpec:
    """One thing an application can be asked to do."""

    name: str
    description: str
    handler: Callable[..., "Result"]

    # Same thing said in each language, for the confirmation question. Without
    # it a Mandarin confirmation reads "这会Power off the Raspberry Pi。确定吗？"
    # — half the sentence in the wrong language, at the exact moment the user
    # most needs to understand what they are agreeing to.
    speech: dict[str, str] = field(default_factory=dict)

    # Spoken forms, per language. `{slot}` marks a free-text argument — there
    # is at most one, and it always runs to the end of the utterance, because
    # anything after it cannot be delimited reliably in speech.
    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Argument name -> human description, for the LLM tool schema in M2.
    params: dict[str, str] = field(default_factory=dict)

    # Commands that destroy state need confirmation before running; the spec
    # requires it and the router refuses to fast-path them.
    confirm: bool = False

    # A score this command needs before it may fire, overriding the router's
    # own floor upwards. For a command with a near-twin — 搜索歌词 against
    # 搜索歌曲, one syllable apart and 0.80 alike in pinyin — the default
    # floor is what lets one answer for the other, and no threshold that
    # both can share fixes it. Raising it here refuses the near-miss and
    # lets the slow path ask, rather than guessing between two commands
    # that do unrelated things. Only ever raises: a spec cannot make itself
    # easier to trigger than the router allows.
    min_score: float | None = None

    # This command is *meant* to leave audio stopped. Music is paused while
    # the assistant listens (see audio/ducking.py) and normally resumed
    # afterwards — but resuming after "pause" would undo the very thing that
    # was asked for, so these opt out of the restore.
    stops_playback: bool = False

    # Whether the reply is spoken. Off by default, which is the unusual choice
    # and the deliberate one.
    #
    # Nearly every command here has a visible result: the music changes, the
    # window changes, the volume changes. Saying "下一首。" over the top of a
    # track that has audibly already changed tells the room nothing it cannot
    # see and hear, and it costs the tail of every turn — measured at 1250 ms
    # of `speaker.wait()` on a 3663 ms turn, during which the assistant is
    # busy and the music stays ducked.
    #
    # So speech is reserved for replies that carry information no other
    # channel does: an answer to a question (`now_playing`), and the acts with
    # no visible result to watch because the screen is about to go away
    # (`shutdown`, `reboot`, `quit`). Everything else acts silently and writes
    # to the panel, which is instant and does not hold the floor.
    speaks: bool = False

    def describe(self, language: str) -> str:
        return self.speech.get(language) or self.description

    @property
    def takes_argument(self) -> bool:
        return bool(self.params)


@dataclass
class Result:
    """What happened, and what to say about it."""

    ok: bool
    speech: dict[str, str] = field(default_factory=dict)

    def say(self, language: str, fallback: str = "en") -> str:
        return self.speech.get(language) or self.speech.get(fallback) or ""

    @classmethod
    def done(cls, en: str, zh: str) -> "Result":
        return cls(True, {"en": en, "zh": zh})

    @classmethod
    def failed(cls, en: str, zh: str) -> "Result":
        return cls(False, {"en": en, "zh": zh})


class Plugin(ABC):
    """An application AIA can control."""

    name: str = "plugin"
    description: str = ""

    @abstractmethod
    def commands(self) -> list[CommandSpec]:
        """Every command this application accepts."""

    def available(self) -> bool:
        """Whether the application can be reached right now.

        Checked before dispatch so an unavailable app produces the spec's
        "Kodama-Lite is not currently running" rather than a stack trace.
        """
        return True

    def manifest(self) -> dict:
        """Serialisable description, for logs and the M2 tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "commands": [
                {
                    "name": c.name,
                    "description": c.description,
                    "params": c.params,
                    "phrases": {lang: list(p) for lang, p in c.phrases.items()},
                }
                for c in self.commands()
            ],
        }


class Registry:
    """Every loaded plugin, and the command lookup over them."""

    def __init__(self, plugins: list[Plugin]):
        self.plugins = plugins
        self._by_name = {p.name: p for p in plugins}
        total = sum(len(p.commands()) for p in plugins)
        log.info("registry: %d plugin(s), %d command(s) — %s",
                 len(plugins), total, ", ".join(self._by_name))

    def get(self, name: str) -> Plugin | None:
        return self._by_name.get(name)

    def all_commands(self) -> list[tuple[Plugin, CommandSpec]]:
        return [(p, c) for p in self.plugins for c in p.commands()]

    def manifests(self) -> list[dict]:
        return [p.manifest() for p in self.plugins]
