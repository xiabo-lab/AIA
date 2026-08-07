"""The two commands added on 2026-08-06, and the confusion each could cause.

`karaoke_exit` is a near-twin of `karaoke` by construction — every one of its
phrases contains the whole of one of the toggle's. That is the shape that has
already cost this project a wrong route once, when a misheard "search lyrics"
reached `save_lyrics`, so it is checked here rather than assumed.
"""

from __future__ import annotations

import unittest

from aia.core.config import CONFIG
from aia.plugins.base import Registry
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System
from aia.router.fast import FastRouter


def router() -> FastRouter:
    return FastRouter(Registry([KodamaLite(), System()]),
                      wake_words=CONFIG.wake.variants)


def routed(text: str) -> str | None:
    found = router().match(text)
    return found.command.name if found else None


class TestKaraokeExit(unittest.TestCase):
    def test_exit_phrases_reach_exit(self):
        for text in ("退出卡拉OK", "关闭卡拉OK", "退出卡拉OK模式", "关掉卡拉OK",
                     "退出全屏歌词", "exit karaoke", "close karaoke",
                     "leave karaoke", "quit karaoke"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "karaoke_exit")

    def test_entering_still_reaches_the_toggle(self):
        for text in ("卡拉OK", "卡拉OK模式", "开启卡拉OK", "全屏歌词",
                     "karaoke", "karaoke mode"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "karaoke")

    def test_the_twins_are_not_close_enough_to_swap(self):
        """The margin that keeps them apart, stated as a number."""
        from aia.router.fast import normalise, similarity
        self.assertLess(similarity(normalise("退出卡拉OK"), normalise("卡拉OK")),
                        0.78)

    def test_exit_sends_off_rather_than_toggling(self):
        """A toggle would reopen karaoke for anyone misheard twice."""
        sent = []
        kodama = KodamaLite()
        kodama._control = lambda action, argument=None: sent.append(
            (action, argument)) or None
        kodama.karaoke_exit()
        self.assertEqual(sent, [("karaoke", "off")])

    def test_off_is_a_word_the_player_understands(self):
        """`parseSwitch` in voiceControl.ts accepts exactly these for false."""
        self.assertIn("off", ("off", "false", "0", "no", "关", "关闭", "取消"))


class TestLibraryCommands(unittest.TestCase):
    """home / play_local / play_liked, added against Kodama-Lite 0.1.42."""

    def test_home(self):
        for text in ("回到主页", "返回主页", "回主页", "主页", "go home",
                     "back to home", "take me home"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "home")

    def test_local_and_usb_are_one_command(self):
        """One action, because there is one offline library: the USB stick.
        The audio cache is an inventory, not a browsable list."""
        for text in ("播放本地音乐", "播放本地歌曲", "播放USB歌曲",
                     "播放U盘的歌", "play local music", "play from usb"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "play_local")

    def test_liked(self):
        for text in ("播放我喜欢的歌曲", "播放我喜欢的歌", "播放收藏的歌曲",
                     "play my liked songs", "play liked music"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "play_liked")

    def test_they_do_not_steal_a_song_request(self):
        """Every one of these phrases begins with 播放, which is also `play`'s
        trigger. "播放本地音乐" is an exact whole-utterance match here *and* an
        exact trigger match for `play` with the argument "本地音乐"; the
        router's tie-break has to prefer the one that consumed the whole
        utterance. If that ever changes, asking for a song by name breaks."""
        for text, query in (("播放周杰伦的歌", "周杰伦的歌"),
                            ("播放五月天", "五月天"),
                            ("播放月亮代表我的心", "月亮代表我的心")):
            found = router().match(text)
            with self.subTest(text=text):
                self.assertEqual(found.command.name, "play")
                self.assertEqual(found.arguments["query"], query)

    def test_they_send_the_action_the_player_whitelists(self):
        """These names must match CONTROL_ACTIONS in Kodama-Lite's
        playback/server.rs, or the endpoint answers 400."""
        sent = []
        kodama = KodamaLite()
        kodama._control = lambda action, argument=None: sent.append(action) or None
        kodama.home()
        kodama.play_local()
        kodama.play_liked()
        self.assertEqual(sent, ["home", "play_local", "play_liked"])

    def test_none_of_them_speak(self):
        """Each has a visible or audible result — a screen change, or music."""
        specs = {c.name: c for _, c in
                 Registry([KodamaLite(), System()]).all_commands()}
        for name in ("home", "play_local", "play_liked", "karaoke_exit"):
            self.assertFalse(specs[name].speaks, f"{name} should be silent")


class TestNetworkStatus(unittest.TestCase):
    def test_phrases_route(self):
        for text in ("网络状态", "有没有网络", "能上网吗", "network status",
                     "are we online", "is the internet working"):
            with self.subTest(text=text):
                self.assertEqual(routed(text), "network")

    def test_it_speaks(self):
        """The answer exists nowhere but in the reply."""
        spec = next(c for _, c in Registry([KodamaLite(), System()]).all_commands()
                    if c.name == "network")
        self.assertTrue(spec.speaks)

    def test_three_answers_not_two(self):
        """A link that is up while names fail is the case that actually
        happened, and it is neither 'online' nor 'offline'."""
        import aia.plugins.system as system_module

        system = System()
        cases = {
            (True, True): ("Online.", "网络正常。"),
            (True, False): ("Connected, but name lookups are failing.",
                            "已连接，但是域名解析失败。"),
            (False, False): ("Offline.", "网络已断开。"),
            (False, True): ("Offline.", "网络已断开。"),
        }
        original_reach, original_resolve = system_module._reachable, system_module._resolves
        try:
            for (route, names), (en, zh) in cases.items():
                system_module._reachable = lambda *a, _r=route, **k: _r
                system_module._resolves = lambda *a, _n=names, **k: _n
                result = system.network()
                with self.subTest(route=route, names=names):
                    self.assertEqual(result.say("en"), en)
                    self.assertEqual(result.say("zh"), zh)
        finally:
            system_module._reachable = original_reach
            system_module._resolves = original_resolve

    def test_a_hanging_resolver_is_reported_as_a_failure(self):
        """getaddrinfo has no timeout of its own."""
        import time

        import aia.plugins.system as system_module

        original_reach, original_resolve = system_module._reachable, system_module._resolves
        system_module.PROBE_TIMEOUT_S_original = system_module.PROBE_TIMEOUT_S
        try:
            system_module._reachable = lambda *a, **k: True
            system_module._resolves = lambda *a, **k: time.sleep(30)
            system_module.PROBE_TIMEOUT_S = 0.1
            started = time.monotonic()
            result = System().network()
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertIn("name lookups", result.say("en"))
        finally:
            system_module._reachable = original_reach
            system_module._resolves = original_resolve
            system_module.PROBE_TIMEOUT_S = system_module.PROBE_TIMEOUT_S_original


if __name__ == "__main__":
    unittest.main()
