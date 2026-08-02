"""Pause whatever is playing while the assistant listens.

The microphone and the speakers share a room. With music playing, the
captured utterance is the command *plus* the song, and Whisper transcribes
the mixture — so a command given over music is often not understood at all.
Ducking is not a nicety here; without it the assistant is close to unusable
whenever it is doing the main thing it is for.

This pauses through MPRIS rather than by muting the output, so the player
knows it is paused: the screen shows it, the track position stops, and a
resume picks up exactly where it left off. Muting would leave the song
playing silently and drop several seconds of it.

Every player on the bus is ducked, not just Kodama-Lite — anything that is
making noise interferes equally, and a future app should not have to be
taught about this. Only players that were actually *playing* are remembered,
so restoring never starts something the user had deliberately paused.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)


def _env() -> dict:
    """Environment with a session bus address.

    A systemd service or an SSH login inherits no DBUS_SESSION_BUS_ADDRESS,
    and playerctl would then find no players and silently duck nothing.
    """
    env = dict(os.environ)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    return env


class Ducker:
    def __init__(self, timeout: float = 1.5):
        # Deliberately short. This runs between the wake word firing and the
        # user starting to speak, so a stalled call must not eat the start of
        # the command — better to miss the duck than to miss the words.
        self.timeout = timeout
        self._paused: list[str] = []

    def _playerctl(self, *args: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["playerctl", *args], capture_output=True, text=True,
                timeout=self.timeout, env=_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("playerctl %s: %s", " ".join(args), exc)
            return False, ""
        return proc.returncode == 0, proc.stdout.strip()

    def duck(self) -> bool:
        """Pause everything currently playing. True if anything was paused."""
        self._paused = []
        ok, listing = self._playerctl("-l")
        if not ok or not listing:
            return False

        for player in listing.splitlines():
            player = player.strip()
            if not player:
                continue
            playing, status = self._playerctl("-p", player, "status")
            if not playing or status.lower() != "playing":
                continue
            if self._playerctl("-p", player, "pause")[0]:
                self._paused.append(player)

        if self._paused:
            log.info("ducked: %s", ", ".join(self._paused))
        return bool(self._paused)

    def restore(self) -> None:
        """Resume only the players this ducked, and only if still paused.

        The "still paused" check matters: a spoken "play something else" has
        already started playback by the time this runs, and blindly issuing
        play again would be redundant at best. Anything the user has since
        touched by hand is left alone too.
        """
        players, self._paused = self._paused, []
        for player in players:
            ok, status = self._playerctl("-p", player, "status")
            if ok and status.lower() == "paused":
                self._playerctl("-p", player, "play")
        if players:
            log.info("restored: %s", ", ".join(players))

    def forget(self) -> None:
        """Drop the memory without resuming — the command wanted it stopped."""
        if self._paused:
            log.info("leaving %s paused", ", ".join(self._paused))
        self._paused = []
