"""Which commands are allowed to speak, and which lyrics command may write.

Both are declaration-level policies rather than logic, and both are the kind
that erode by accident: `speaks=True` is one word to add to a new command, and
a floor is one number to round down. So they are asserted where a diff can see
them.
"""

from __future__ import annotations

import unittest

from aia.plugins.base import Registry
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System

# Every command whose reply is spoken aloud, and nothing else. `now_playing`
# answers a question — the answer exists nowhere but in the reply. The other
# three take the screen away, so there is nothing left to look at.
SPEAKING = {"now_playing", "shutdown", "reboot", "quit"}


def registry() -> Registry:
    return Registry([KodamaLite(), System()])


class TestWhoSpeaks(unittest.TestCase):
    def test_exactly_the_declared_set_speaks(self):
        speaking = {c.name for _, c in registry().all_commands() if c.speaks}
        self.assertEqual(speaking, SPEAKING)

    def test_everything_else_is_silent(self):
        quiet = {c.name for _, c in registry().all_commands() if not c.speaks}
        self.assertNotIn("next", quiet & SPEAKING)
        # The ones most likely to be given a voice back by reflex: their
        # result is audible or visible the moment it happens.
        for name in ("next", "pause", "resume", "volume", "like", "lyrics"):
            self.assertIn(name, quiet, f"{name} should act silently")


class TestLyricsFloors(unittest.TestCase):
    """The write must be harder to trigger than the reads next to it.

    Measured on the Pi: a misheard "search lyrics" reaches 0.889 against
    `save lyrics`. Anything at or below that lets a request to read perform a
    write instead.
    """

    def floors(self) -> dict[str, float | None]:
        return {c.name: c.min_score for _, c in registry().all_commands()}

    def test_save_is_above_the_measured_near_miss(self):
        self.assertIsNotNone(self.floors()["save_lyrics"])
        self.assertGreater(self.floors()["save_lyrics"], 0.889)

    def test_save_is_stricter_than_the_reads(self):
        floors = self.floors()
        self.assertGreater(floors["save_lyrics"], floors["search_lyrics"])

    def test_the_reads_still_have_a_raised_floor(self):
        """Raising `save` must not quietly become the only guard: 搜索歌词 and
        搜索歌曲 are 0.80 alike and that is what `search_lyrics` defends."""
        self.assertIsNotNone(self.floors()["search_lyrics"])
        self.assertGreater(self.floors()["search_lyrics"], 0.78)


if __name__ == "__main__":
    unittest.main()
