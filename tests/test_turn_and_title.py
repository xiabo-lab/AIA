"""What the latency budget is about, and how much of a title is worth saying.

Both were measured wrong in the same way: something true was reported as
though it were something else. A turn that answered quickly and then spoke for
three seconds was logged as a latency failure, and a title full of upload
credits was read out in full because it was, technically, the title.
"""

from __future__ import annotations

import unittest

from aia.core.state import Machine, Turn
from aia.plugins.kodama import TITLE_LIMIT, _short_title


class TestTurnVerdict(unittest.TestCase):
    def turn(self, **marks) -> Turn:
        t = Turn()
        t.marks.update(marks)
        return t

    def test_speaking_time_is_not_latency(self):
        """The case from the journal: audio started at 4379 ms and the reply
        took three more seconds to say. Only the first number is the wait."""
        t = self.turn(captured=3471, stt=3708, routed=3738, acted=3811,
                      audio_out=4379)
        _, over = t.report(5000)
        self.assertFalse(over)

    def test_slow_to_first_audio_is_still_over(self):
        """The fix must not make everything green."""
        t = self.turn(captured=3471, audio_out=4379)
        line, over = t.report(2500)
        self.assertTrue(over)
        self.assertIn("OVER by 1879ms", line)

    def test_verdict_ignores_time_spent_after_audio_started(self):
        early = self.turn(audio_out=1000)
        line, over = early.report(2500)
        self.assertFalse(over)
        self.assertIn("1000ms to audio", line)

    def test_a_turn_that_never_reached_audio_falls_back_to_the_total(self):
        """An empty transcript, or a turn that died. Nothing better to offer."""
        t = self.turn(captured=100)
        line, over = t.report(2500)
        self.assertIn("total", line)
        self.assertFalse(over)

    def test_the_total_is_still_visible(self):
        t = self.turn(audio_out=1000)
        line, _ = t.report(2500)
        self.assertIn("total", line)
        self.assertIn("audio_out=1000", line)

    def test_machine_warns_on_the_judged_number_not_the_total(self):
        machine = Machine(budget_ms=2500)
        turn = machine.begin_turn()
        turn.marks.update(audio_out=1000)
        with self.assertLogs("aia.core.state", level="INFO") as captured:
            machine.end_turn()
        self.assertTrue(any(r.levelname == "INFO" for r in captured.records))
        self.assertFalse(any(r.levelname == "WARNING" for r in captured.records))


class TestShortTitle(unittest.TestCase):
    def test_the_two_real_titles(self):
        self.assertEqual(
            _short_title("《單車》陳奕迅｜Cover by CCG｜Acoustic R&B / Lo-fi"),
            "《單車》陳奕迅")
        self.assertEqual(_short_title("年轮（R&B版）-陶宏杰"), "年轮（R&B版）-陶宏杰")

    def test_a_plain_title_is_left_alone(self):
        for title in ("晴天", "Hotel California", "七里香"):
            self.assertEqual(_short_title(title), title)

    def test_a_hyphen_is_not_a_separator(self):
        """It appears inside real names — Jay-Z, Lo-fi, R&B版."""
        self.assertEqual(_short_title("Jay-Z"), "Jay-Z")

    def test_slash_only_splits_when_spaced(self):
        self.assertEqual(_short_title("Acoustic R&B/Lo-fi"), "Acoustic R&B/Lo-fi")
        self.assertEqual(_short_title("單車 / 陳奕迅"), "單車")

    def test_long_titles_are_cut_on_a_boundary(self):
        long = "這是一首非常長的歌名，長到沒有人想要整句聽完它的名字"
        short = _short_title(long)
        self.assertLessEqual(len(short), TITLE_LIMIT)
        self.assertTrue(long.startswith(short))

    def test_never_returns_empty(self):
        """A title that is nothing but separators still has to say something."""
        self.assertTrue(_short_title("｜｜｜"))
        self.assertEqual(_short_title("  晴天  "), "晴天")

    def test_blurb_brackets_are_dropped(self):
        """The third real title, which is a line of the lyrics in brackets."""
        self.assertEqual(
            _short_title("K.D 翻唱《不如》【不如我們擁抱後分手，不如眼淚有空偷偷流...】♫"),
            "K.D 翻唱《不如》")

    def test_round_brackets_are_kept(self):
        """(Live) and （R&B版） say what the recording is."""
        self.assertEqual(_short_title("年轮（R&B版）"), "年轮（R&B版）")
        self.assertEqual(_short_title("Hotel California (Live)"),
                         "Hotel California (Live)")

    def test_a_cut_never_leaves_a_bracket_open(self):
        for title in ("K.D 翻唱《不如》【不如我們擁抱後分手，不如眼淚有空偷偷流...】♫",
                      "這是一首歌《非常非常非常長的副標題還沒有結束",
                      "歌名（一個很長很長很長很長的說明"):
            short = _short_title(title)
            with self.subTest(title=title):
                for opener, closer in (("《", "》"), ("（", "）"), ("【", "】")):
                    if opener in short:
                        self.assertIn(closer, short.split(opener, 1)[1])

    def test_decoration_is_stripped(self):
        self.assertEqual(_short_title("晴天 ♫"), "晴天")
        self.assertEqual(_short_title("~晴天~"), "晴天")


if __name__ == "__main__":
    unittest.main()
