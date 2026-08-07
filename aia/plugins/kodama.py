"""Kodama-Lite, driven over MPRIS.

Kodama-Lite already publishes an MPRIS service on the session D-Bus — see its
`src-tauri/src/subsystems/media.rs`, which exists so a paired car head unit can
drive playback over Bluetooth AVRCP. AIA is just another MPRIS client, so
transport control works today with **no changes to Kodama-Lite at all**.
Verified on the device: `Next`, `PlayPause`, `Previous`, `Stop` with `CanPlay`,
`CanPause`, `CanGoNext`, `CanGoPrevious` and `CanSeek` all true, and a
play/pause round trip moves the player and comes back.

What MPRIS does *not* reach is anything that lives in the Kodama-Lite
frontend: searching for a song, switching playlist, shuffle, repeat, lyrics,
and the volume slider (which its `subsystems/volume.rs` documents as
deliberately owned by the UI). Those need a control endpoint added to
Kodama-Lite itself — planned as M5, and the reason `play` here resumes rather
than searching.

`playerctl` is shelled out to rather than binding D-Bus directly: measured at
10 ms per call, which is negligible against a 2.5 s budget, and it avoids
another native dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

import requests

from aia.core.config import CONFIG, KodamaConfig
from aia.plugins.base import CommandSpec, Plugin, Result

log = logging.getLogger(__name__)


# Where Kodama-Lite publishes its control endpoint. Its stream server binds a
# random port under a random per-launch token — deliberately, so knowing the
# port is not enough to drive the app — which also means the URL cannot be
# guessed. It writes it here, mode 0600, for a process running as the same
# user. See `publish_control_endpoint` in its playback/server.rs.
CONTROL_STATE = Path.home() / ".local/state/kodama-lite/control.json"

# Spoken numbers. Whisper usually returns digits ("音量调到50%") but not
# always, and a Chinese numeral is the natural way to say it.
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def parse_level(text: str) -> int | None:
    """Read a 0-100 level out of a spoken phrase, in either language.

    Handles "50", "fifty percent", "百分之五十", "五十", "一百". Returns None
    when there is no number, so the caller can ask rather than guess.
    """
    if not text:
        return None

    # "百分之五十" is 50 percent, not 100 — strip the "percent of" marker
    # before any digit or numeral hunting, or the 百 in it reads as a value.
    text = text.replace("百分之", "").replace("百分比", "")

    # The sign is part of the number. Without it "-10" reads as 10 and
    # turns the volume *up*, which is the opposite of what was asked;
    # clamping a negative to 0 at least does something defensible.
    digits = re.search(r"-?\d+", text)
    if digits:
        return max(0, min(100, int(digits.group())))

    words = {"zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
             "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
             "ninety": 90, "hundred": 100, "full": 100, "half": 50,
             "max": 100, "maximum": 100, "mute": 0}
    lowered = text.lower()
    for word, value in words.items():
        if re.search(rf"\b{word}\b", lowered):
            return value

    # Chinese numerals up to 一百. 五十 = 50, 五十五 = 55, 十五 = 15.
    cn = re.sub(r"[^零〇一两二三四五六七八九十百]", "", text)
    if not cn:
        return None
    if "百" in cn:
        return 100
    if "十" in cn:
        before, _, after = cn.partition("十")
        tens = _CN_DIGITS.get(before, 1) if before else 1
        units = _CN_DIGITS.get(after, 0) if after else 0
        return max(0, min(100, tens * 10 + units))
    if len(cn) == 1 and cn in _CN_DIGITS:
        return _CN_DIGITS[cn]
    return None


def _session_bus_env() -> dict:
    """Environment with a session bus address, whatever we were started from.

    AIA runs as a service, and a service (or an SSH login) inherits no
    DBUS_SESSION_BUS_ADDRESS — playerctl would then find no players at all and
    every command would look like "Kodama-Lite is not running". Kodama-Lite
    runs as a systemd *user* service, so its bus is the per-uid socket.
    """
    env = dict(os.environ)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    return env


class KodamaLite(Plugin):
    name = "kodama"
    description = "Music player (Kodama-Lite)"

    def __init__(self, cfg: KodamaConfig | None = None):
        self.cfg = cfg or CONFIG.kodama

    def _run(self, *args: str, timeout: float | None = None) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["playerctl", "-p", self.cfg.player, *args],
                capture_output=True, text=True,
                timeout=timeout if timeout is not None else self.cfg.playerctl_timeout_s,
                env=_session_bus_env(),
            )
        except FileNotFoundError:
            log.error("playerctl is not installed")
            return False, ""
        except subprocess.TimeoutExpired:
            log.warning("playerctl %s timed out", " ".join(args))
            return False, ""
        return proc.returncode == 0, proc.stdout.strip()

    def available(self) -> bool:
        ok, _ = self._run("status")
        return ok

    # ── the control endpoint, for everything MPRIS cannot reach ──────

    def _control_url(self) -> str | None:
        """Current control URL, re-read each time.

        Not cached: the token and port are regenerated on every launch of
        Kodama-Lite, so a cached URL survives exactly until the app restarts
        and then silently fails. Reading a small file per command costs
        nothing against a 2.5 s budget.
        """
        try:
            data = json.loads(CONTROL_STATE.read_text())
        except FileNotFoundError:
            log.debug("no control endpoint at %s", CONTROL_STATE)
            return None
        except (OSError, ValueError) as exc:
            log.warning("cannot read %s: %s", CONTROL_STATE, exc)
            return None
        return data.get("url")

    def _control(self, action: str, argument: str | None = None) -> str | None:
        """Send a command. Returns None on success, else why it failed.

        Tri-state rather than a bool because the three failures want three
        different things said, and telling them apart used to cost a second
        `playerctl` spawn: every caller answered a False by asking
        `available()` all over again to guess which had happened.

        Note the app answers 202 the moment the event is on its bus, before
        the view plane has acted — deliberately, so the assistant is not
        waiting on the UI to speak. What it must not do is answer 202 to an
        action it has never heard of, because then this cannot tell a real
        command from a feature the installed version predates. Kodama-Lite
        rejects unknown actions with 400 from v0.1.39; against anything older
        every command here reports success it did not have.
        """
        url = self._control_url()
        if not url:
            return "no-endpoint"
        payload = {"action": action}
        if argument is not None:
            payload["argument"] = argument
        try:
            resp = requests.post(url, json=payload,
                                 timeout=self.cfg.control_timeout_s)
        except requests.RequestException as exc:
            log.warning("control %s failed: %s", action, exc)
            return "unreachable"
        if resp.status_code == 400:
            log.warning("control %s was rejected — the app does not know that "
                        "action", action)
            return "no-endpoint"
        if resp.status_code >= 400:
            log.warning("control %s returned %s", action, resp.status_code)
            return "unreachable"
        return None

    def _control_failed(self, reason: str) -> Result:
        """What to say about a control command that did not land."""
        if reason == "unreachable":
            return self._unavailable()
        # Either no endpoint file at all, or the app rejected the action.
        # Both mean the same thing to the person listening: this build cannot
        # do that.
        return Result.failed(
            "That needs a newer version of Kodama-Lite.",
            "这个功能需要更新版本的 Kodama-Lite。",
        )

    # ── command implementations ──────────────────────────────────────
    # Each returns what to say. Confirmations are phrased for speech, not for
    # a screen: short, and describing what happened rather than restating the
    # command back at the user.

    def _unavailable(self) -> Result:
        return Result.failed(
            "Kodama-Lite is not currently running.",
            "Kodama-Lite 没有在运行。",
        )

    def pause(self) -> Result:
        ok, _ = self._run("pause")
        return Result.done("Paused.", "已暂停。") if ok else self._unavailable()

    def resume(self) -> Result:
        ok, _ = self._run("play")
        return Result.done("Playing.", "开始播放。") if ok else self._unavailable()

    def toggle(self) -> Result:
        """Flip playback, and say which way it went.

        Reads the state, then asks for the opposite outright rather than
        sending `play-pause` and looking up what happened. Looking it up does
        not work: MPRIS status lags a transition by 88-156 ms on this device,
        so the read came back with the state from *before* the toggle and the
        assistant announced the opposite of what it had just done — measured
        wrong 4 times out of 4, in both directions.

        Naming the target state also makes the reply true by construction:
        what is announced is what was asked for, not what a second query
        guessed a moment too early.
        """
        ok, status = self._run("status")
        if not ok:
            return self._unavailable()
        playing = status.lower() == "playing"
        if not self._run("pause" if playing else "play")[0]:
            return self._unavailable()
        return Result.done("Paused." if playing else "Playing.",
                           "已暂停。" if playing else "开始播放。")

    def next_track(self) -> Result:
        if not self._run("next")[0]:
            return self._unavailable()
        return Result.done("Next track.", "下一首。")

    def previous_track(self) -> Result:
        if not self._run("previous")[0]:
            return self._unavailable()
        return Result.done("Previous track.", "上一首。")

    def stop(self) -> Result:
        ok, _ = self._run("stop")
        return Result.done("Stopped.", "已停止。") if ok else self._unavailable()

    def now_playing(self) -> Result:
        ok, title = self._run("metadata", "xesam:title")
        if not ok:
            return self._unavailable()
        if not title:
            return Result.done("Nothing is playing.", "现在没有播放。")
        _, artist = self._run("metadata", "xesam:artist")
        if artist:
            return Result.done(f"{title}, by {artist}.", f"{title}，{artist}。")
        return Result.done(f"{title}.", f"{title}。")

    def play(self, query: str) -> Result:
        failed = self._control("play", query)
        if failed:
            return self._control_failed(failed)
        return Result.done(f"Playing {query}.", f"正在播放{query}。")

    def search(self, query: str) -> Result:
        failed = self._control("search", query)
        if failed:
            return self._control_failed(failed)
        return Result.done(f"Searching for {query}.", f"正在搜索{query}。")

    def volume(self, level: str) -> Result:
        value = parse_level(level)
        if value is None:
            return Result.failed("What volume?", "音量调到多少？")
        failed = self._control("volume", str(value))
        if failed:
            return self._control_failed(failed)
        return Result.done(f"Volume {value} percent.", f"音量百分之{value}。")

    def shuffle(self, state: str = "") -> Result:
        failed = self._control("shuffle", state or None)
        if failed:
            return self._control_failed(failed)
        return Result.done("Shuffle toggled.", "随机播放已切换。")

    def repeat(self, mode: str = "") -> Result:
        failed = self._control("repeat", mode or None)
        if failed:
            return self._control_failed(failed)
        return Result.done("Repeat toggled.", "循环模式已切换。")

    def like(self) -> Result:
        failed = self._control("like")
        if failed:
            return self._control_failed(failed)
        return Result.done("Liked.", "已点赞。")

    def lyrics(self) -> Result:
        failed = self._control("lyrics")
        if failed:
            return self._control_failed(failed)
        return Result.done("Showing lyrics.", "正在显示歌词。")

    def search_lyrics(self) -> Result:
        """The karaoke stage's magnifier: look the lyric up again.

        Deliberately not `lyrics`. That one shows what has already been
        found; this one goes back out to the sources for the track playing
        now. They are two buttons on the same screen and they have to stay
        two commands — see the note on the CommandSpec pair below.
        """
        failed = self._control("lyrics_search")
        if failed:
            return self._control_failed(failed)
        return Result.done("Searching for lyrics.", "正在搜索歌词。")

    def save_lyrics(self) -> Result:
        """The green tick: commit the lyric on screen to the cache.

        Nothing reaches Kodama-Lite's persistent lyrics cache until this is
        pressed, so this is the one lyrics command that writes anything.
        """
        failed = self._control("lyrics_save")
        if failed:
            return self._control_failed(failed)
        return Result.done("Lyrics saved.", "歌词已保存。")

    def search_song(self, query: str) -> Result:
        """Song Search, with a name — the same endpoint `search` uses.

        A separate command rather than more phrases on `search` because the
        trigger has to be long enough to be told apart from 搜索歌词 by
        something more than one syllable. `search` stays as the general
        "搜索X" form; this one exists so that saying the word 歌曲 out loud
        cannot land in the lyrics half of the app.
        """
        failed = self._control("search", query)
        if failed:
            return self._control_failed(failed)
        return Result.done(f"Searching for {query}.", f"正在搜索{query}。")

    def karaoke(self, state: str = "") -> Result:
        failed = self._control("karaoke", state or None)
        if failed:
            return self._control_failed(failed)
        return Result.done("Karaoke mode.", "卡拉OK模式。")

    def quit_app(self) -> Result:
        failed = self._control("quit")
        if failed:
            return self._control_failed(failed)
        return Result.done("Closing Kodama-Lite.", "正在关闭 Kodama-Lite。")

    def commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(
                name="pause", description="Pause playback", handler=self.pause,
                stops_playback=True,
                phrases={
                    "en": ("pause", "pause music", "pause the music", "stop music"),
                    "zh": ("暂停", "暂停音乐", "暂停播放"),
                },
            ),
            CommandSpec(
                name="resume", description="Resume playback", handler=self.resume,
                # The Chinese list is longer than it looks like it needs to be
                # because a phrase is matched as a whole: 播放歌曲 scores only
                # 0.75 against the phrase 播放, since the trailing 歌曲 is two
                # thirds more syllables. Spoken forms are declared, not
                # inferred, so each natural way of saying it gets an entry.
                phrases={
                    "en": ("resume", "play", "play music", "play some music",
                           "continue", "resume music", "keep playing"),
                    "zh": ("继续", "播放", "播放音乐", "播放歌曲", "继续播放",
                           "开始播放", "放歌", "放音乐", "来点音乐"),
                },
            ),
            CommandSpec(
                name="toggle", description="Toggle play/pause", handler=self.toggle,
                phrases={"en": ("play pause", "toggle playback"), "zh": ("暂停或播放",)},
            ),
            CommandSpec(
                name="next", description="Skip to the next track", handler=self.next_track,
                phrases={
                    "en": ("next", "next track", "next song", "skip", "skip this song"),
                    "zh": ("下一首", "下一曲", "切歌", "跳过"),
                },
            ),
            CommandSpec(
                name="previous", description="Go back to the previous track",
                handler=self.previous_track,
                phrases={
                    "en": ("previous", "previous track", "previous song", "go back", "last song"),
                    "zh": ("上一首", "上一曲", "返回上一首"),
                },
            ),
            CommandSpec(
                name="stop", description="Stop playback", handler=self.stop,
                stops_playback=True,
                phrases={"en": ("stop", "stop playback"), "zh": ("停止", "停止播放")},
            ),
            CommandSpec(
                name="now_playing", description="Say what is currently playing",
                handler=self.now_playing, speaks=True,
                phrases={
                    "en": ("what's playing", "what is playing", "what song is this",
                           "now playing", "what's this song"),
                    "zh": ("现在播放什么", "这是什么歌", "正在播放什么"),
                },
            ),

            # ── via the control endpoint ─────────────────────────────
            # A `{slot}` runs to the end of the utterance. Note the trigger
            # phrases here overlap with the whole-phrase ones above —
            # "播放歌曲" is `resume`, "播放周杰伦" is `play`. The router
            # prefers the whole-utterance match on a tie, which is what
            # keeps those apart.
            CommandSpec(
                name="play", description="Search for a song and play it",
                handler=self.play, params={"query": "song, artist or album"},
                phrases={
                    "en": ("play {query}", "put on {query}", "i want to hear {query}",
                           "listen to {query}"),
                    "zh": ("播放{query}", "我想听{query}", "来一首{query}",
                           "放一首{query}", "听{query}"),
                },
            ),
            CommandSpec(
                name="search", description="Search without playing",
                handler=self.search, params={"query": "what to search for"},
                phrases={
                    "en": ("search for {query}", "search {query}", "find {query}",
                           "look for {query}"),
                    "zh": ("搜索{query}", "搜寻{query}", "查找{query}", "帮我搜索{query}"),
                },
            ),
            CommandSpec(
                name="volume", description="Set the volume, 0-100",
                handler=self.volume, params={"level": "0-100"},
                phrases={
                    "en": ("volume {level}", "set volume to {level}",
                           "turn volume to {level}", "turn the volume to {level}"),
                    "zh": ("音量{level}", "音量调到{level}", "把音量调到{level}",
                           "声音调到{level}"),
                },
            ),
            CommandSpec(
                name="shuffle", description="Toggle shuffle", handler=self.shuffle,
                phrases={
                    "en": ("shuffle", "shuffle mode", "toggle shuffle", "random play"),
                    "zh": ("随机播放", "随机模式", "打开随机播放", "切换随机播放"),
                },
            ),
            CommandSpec(
                name="repeat", description="Toggle repeat mode", handler=self.repeat,
                phrases={
                    "en": ("repeat", "repeat mode", "toggle repeat", "loop"),
                    "zh": ("循环播放", "单曲循环", "重复播放", "切换循环"),
                },
            ),
            CommandSpec(
                name="like", description="Like the current track", handler=self.like,
                phrases={
                    "en": ("like this song", "like this", "favourite this song",
                           "add to favourites", "thumbs up"),
                    "zh": ("点赞", "喜欢这首歌", "收藏这首歌", "加入收藏"),
                },
            ),
            # ── the three lyrics/song commands that must not overlap ──
            # These are one screen's worth of buttons and three different
            # actions, and two of them are one syllable apart in Mandarin:
            # 搜索歌词 is `sousuogeci` and 搜索歌曲 is `sousuogequ`, which
            # score 0.80 against each other in the pinyin the router
            # compares — above the 0.78 it needs to fire. So the separation
            # is not left to the threshold:
            #
            #   * `lyrics` no longer answers to "搜索歌词" at all. It used
            #     to carry both, which meant the show and search phrases
            #     could never be told apart because they were one command.
            #   * `search_lyrics` and `save_lyrics` take no argument, so
            #     they are matched end to end and carry a raised `min_score`
            #     — a misheard 曲 lands 0.80 and is refused rather than
            #     opening the Song Search window.
            #   * `search_song` requires a name after the trigger, and its
            #     four-syllable trigger beats the two-syllable 搜索 on the
            #     router's longest-trigger tie-break, so the name arrives
            #     clean instead of as "歌曲晴天".
            CommandSpec(
                name="lyrics", description="Show lyrics for the current track",
                handler=self.lyrics,
                phrases={
                    "en": ("show lyrics", "lyrics", "show the lyrics",
                           "display lyrics", "show me the lyrics"),
                    "zh": ("显示歌词", "歌词", "看歌词", "显示一下歌词"),
                },
            ),
            CommandSpec(
                name="search_lyrics",
                description="Search again for the current track's lyrics",
                handler=self.search_lyrics, min_score=0.85,
                phrases={
                    "en": ("search lyric", "search lyrics", "search for lyrics",
                           "search the lyrics", "find lyrics", "find the lyrics",
                           "look up the lyrics", "search for the lyrics"),
                    "zh": ("搜索歌词", "搜寻歌词", "查找歌词", "找歌词",
                           "重新搜索歌词", "搜一下歌词"),
                },
            ),
            # `save_lyrics` sits higher than its neighbours because it is the
            # only one of the three that *writes*. Measured on this device:
            # SenseVoice heard "search lyrics" as "Se lyrics.", which scores
            # 0.889 against `save lyrics` and only 0.800 against `search
            # lyrics` — the dropped syllables leave the shared noun carrying
            # the match, and "se" is genuinely closer to "save" than to
            # "search". So a request to read was one floor away from writing
            # to the lyrics cache.
            #
            # No scoring change separates the two. Comparing verbs alone is
            # worse, not better: "se" vs "save" is 0.667 while the correctly
            # intended "say" vs "save" is 0.571, so the wrong reading wins
            # there too. The scores simply overlap, because the recogniser —
            # not the router — lost the information.
            #
            # 0.90 is derived from the corpus rather than picked: across 71
            # real captures every genuine save scores exactly 1.000 (9 of
            # them), and the highest any non-save utterance reaches against
            # this command is 0.889. Any floor in between refuses the
            # near-miss; 0.90 is the round number with the most headroom.
            # The cost is measured too, and it is one capture: a degraded
            # "Say the lyrics." at 0.880 is now declined rather than obeyed.
            # That is the intended direction — an unwanted write is worse
            # than being asked to repeat yourself.
            CommandSpec(
                name="save_lyrics",
                description="Save the current lyrics to the lyrics cache",
                handler=self.save_lyrics, min_score=0.90,
                phrases={
                    "en": ("save lyric", "save lyrics", "save the lyrics",
                           "save these lyrics", "keep these lyrics",
                           "confirm lyrics", "confirm the lyrics"),
                    "zh": ("保存歌词", "保存这个歌词", "保存这首歌的歌词",
                           "储存歌词", "确认歌词"),
                },
            ),
            CommandSpec(
                name="search_song", description="Search for a song by name",
                handler=self.search_song, params={"query": "song name"},
                phrases={
                    "en": ("search song {query}", "search songs {query}",
                           "search for song {query}", "search for the song {query}",
                           "find song {query}", "find the song {query}"),
                    "zh": ("搜索歌曲{query}", "搜寻歌曲{query}",
                           "查找歌曲{query}", "找歌曲{query}",
                           "搜索一下歌曲{query}"),
                },
            ),
            CommandSpec(
                name="karaoke", description="Toggle full-screen karaoke",
                handler=self.karaoke,
                phrases={
                    "en": ("karaoke", "karaoke mode", "full screen lyrics"),
                    "zh": ("卡拉OK", "卡拉OK模式", "全屏歌词", "开启卡拉OK"),
                },
            ),
            CommandSpec(
                name="quit", description="Close Kodama-Lite", handler=self.quit_app,
                confirm=True, stops_playback=True, speaks=True,
                speech={"en": "close Kodama-Lite", "zh": "关闭音乐播放器"},
                phrases={
                    "en": ("close kodama", "quit kodama", "exit kodama",
                           "close the music player"),
                    "zh": ("退出软件", "关闭软件", "退出音乐播放器", "关闭播放器"),
                },
            ),
        ]
