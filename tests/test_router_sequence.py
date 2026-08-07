"""More than one command in a single breath.

Both cases here were captured from real use, not invented: "下一首and现在播放
什么？" (SenseVoice renders the spoken "and" literally) and "暂停小爱同学搜索
歌", where the wake word arrives mid-sentence because the speaker summoned the
assistant again without waiting for the first command to finish.

The property that matters most is not that chains work — it is that nothing
else changed. `match_sequence` tries the whole utterance first and only splits
when the router had already declined, so every transcript that routes today
still routes to exactly the same command.
"""

from __future__ import annotations

import unittest

from aia.core.config import CONFIG
from aia.plugins.base import Registry
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System
from aia.router.fast import MAX_CHAIN, FastRouter

WAKE = CONFIG.wake.variants


def router() -> FastRouter:
    return FastRouter(Registry([KodamaLite(), System()]), wake_words=WAKE)


def names(intents) -> list[str]:
    return [i.command.name for i in intents]


class TestSingleCommandsAreUntouched(unittest.TestCase):
    """The regression guarantee, stated as a test."""

    SINGLES = [
        "下一首", "暂停音乐", "播放周杰伦的歌", "搜索歌词", "现在播放什么",
        "音量调到五十", "pause the music", "what's playing", "show lyrics",
        "play some music",
    ]

    def test_sequence_agrees_with_match_on_every_single_command(self):
        r = router()
        for text in self.SINGLES:
            one = r.match(text)
            many = r.match_sequence(text)
            with self.subTest(text=text):
                if one is None:
                    self.assertEqual(many, [])
                else:
                    self.assertEqual(names(many), [one.command.name])

    def test_an_argument_is_never_split_on_a_connective(self):
        """A slot runs to the end of the utterance. "播放五月天和陈奕迅" is one
        search for one query, and splitting it would make it two wrong ones."""
        r = router()
        found = r.match_sequence("播放五月天和陈奕迅")
        self.assertEqual(names(found), ["play"])
        self.assertEqual(found[0].arguments["query"], "五月天和陈奕迅")


class TestChains(unittest.TestCase):
    def test_the_captured_and_utterance(self):
        found = router().match_sequence("下一首and现在播放什么？")
        self.assertEqual(names(found), ["next", "now_playing"])

    def test_a_wake_word_in_the_middle_is_a_boundary(self):
        found = router().match_sequence("暂停小爱同学下一首")
        self.assertEqual(names(found), ["pause", "next"])

    def test_chinese_connective(self):
        found = router().match_sequence("暂停音乐然后现在播放什么")
        self.assertEqual(names(found), ["pause", "now_playing"])

    def test_order_is_preserved(self):
        found = router().match_sequence("现在播放什么and下一首")
        self.assertEqual(names(found), ["now_playing", "next"])


class TestChainsThatMustNotHappen(unittest.TestCase):
    def test_all_or_nothing(self):
        """Half a garbled sentence is a command nobody asked for."""
        found = router().match_sequence("下一首and的天气怎么样")
        self.assertEqual(found, [])

    def test_a_command_needing_confirmation_is_refused_in_a_chain(self):
        """关机 must be asked for on its own, not reached as a tail."""
        with self.assertLogs("aia.router.fast", level="WARNING"):
            found = router().match_sequence("下一首and关机")
        self.assertEqual(found, [])

    def test_confirmation_still_works_on_its_own(self):
        found = router().match_sequence("关机")
        self.assertEqual(names(found), ["shutdown"])
        self.assertTrue(found[0].command.confirm)

    def test_too_many_segments_is_a_sentence_not_a_list(self):
        text = "and".join(["下一首"] * (MAX_CHAIN + 1))
        self.assertEqual(router().match_sequence(text), [])

    def test_nonsense_still_declines(self):
        self.assertEqual(router().match_sequence("今天天气怎么样"), [])


class TestSegmenting(unittest.TestCase):
    def test_empty_segments_are_dropped(self):
        r = router()
        self.assertEqual(r._segments("下一首,"), ["下一首"])
        self.assertEqual(r._segments("，、,"), [])

    def test_he_is_not_a_separator(self):
        """和 is a conjunction and also a syllable in names."""
        self.assertEqual(router()._segments("播放周杰伦和五月天"),
                         ["播放周杰伦和五月天"])

    def test_works_without_wake_words_configured(self):
        bare = FastRouter(Registry([KodamaLite(), System()]))
        self.assertEqual(names(bare.match_sequence("下一首and现在播放什么")),
                         ["next", "now_playing"])


if __name__ == "__main__":
    unittest.main()
