"""Device-level commands.

Deliberately tiny, and deliberately not a shell. The spec is explicit that
voice must not reach a shell and that destructive actions need confirmation,
so this exposes exactly one irreversible operation — shutdown — as a named
command with `confirm=True`, and nothing that takes free text.

If this file ever grows a command that passes user speech to a subprocess
argument, that is the moment to stop and reconsider: a recogniser's output is
untrusted input, and "no remote shell access by voice" is a security
requirement rather than a stylistic one.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from aia.plugins.base import CommandSpec, Plugin, Result

log = logging.getLogger(__name__)


class System(Plugin):
    name = "system"
    description = "The Raspberry Pi itself"

    def available(self) -> bool:
        return shutil.which("systemctl") is not None

    @staticmethod
    def _privileged(verb: str) -> list[str] | None:
        """How to run `systemctl <verb>`, or None if this process may not.

        Plain `systemctl poweroff` is refused here. logind hands the decision
        to polkit, which only grants power-off without a password to an
        *active, local* session — and AIA is typically started over SSH or as
        a service, so it fails with "Interactive authentication required".
        That is a permission problem, not a broken command, and it must be
        detected *before* announcing success rather than discovered by the
        user when the machine stays on.
        """
        try:
            if subprocess.run(["sudo", "-n", "true"], capture_output=True,
                              timeout=5).returncode == 0:
                return ["sudo", "-n", "systemctl", verb]
        except (OSError, subprocess.SubprocessError):
            pass

        action = {"poweroff": "power-off", "reboot": "reboot"}.get(verb, verb)
        try:
            authorised = subprocess.run(
                ["pkcheck", "--action-id", f"org.freedesktop.login1.{action}",
                 "--process", str(os.getpid())],
                capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.SubprocessError):
            authorised = False
        return ["systemctl", verb] if authorised else None

    def _power(self, verb: str, en: str, zh: str, done_en: str, done_zh: str) -> Result:
        """Shared path for poweroff/reboot.

        Reached only after the user has confirmed: the router refuses to
        fast-path a `confirm=True` command, and main.py holds the floor for an
        explicit yes.

        The command itself is deferred a few seconds so the spoken
        confirmation finishes playing — poweroff takes the audio stack down
        almost immediately, which would otherwise cut the assistant off
        mid-sentence and leave the user unsure whether anything happened.
        Authorisation, though, is checked *now*, so a refusal is reported as a
        refusal.
        """
        argv = self._privileged(verb)
        if argv is None:
            log.error("not authorised to %s (polkit needs an active local "
                      "session; passwordless sudo would also do)", verb)
            return Result.failed(
                f"I don't have permission to {en}.",
                f"我没有权限{zh}。",
            )
        try:
            subprocess.Popen(
                ["sh", "-c", f"sleep 3 && exec {' '.join(argv)}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            log.error("%s failed to start: %s", verb, exc)
            return Result.failed(f"I couldn't {en}.", f"无法{zh}。")
        log.warning("%s confirmed — running `%s` in 3 seconds",
                    verb, " ".join(argv))
        return Result.done(done_en, done_zh)

    def shutdown(self) -> Result:
        return self._power("poweroff", "shut down", "关机",
                           "Shutting down. Goodbye.", "正在关机，再见。")

    def reboot(self) -> Result:
        return self._power("reboot", "restart", "重启",
                           "Restarting.", "正在重启。")

    def commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(
                name="shutdown", description="Power off the Raspberry Pi",
                handler=self.shutdown, confirm=True, stops_playback=True, speaks=True,
                speech={"en": "power off the Raspberry Pi", "zh": "关闭树莓派"},
                phrases={
                    "en": ("shut down", "shutdown", "power off", "turn off the pi",
                           "shut down the pi"),
                    "zh": ("关机", "关闭电源", "把树莓派关机", "关掉设备"),
                },
            ),
            CommandSpec(
                name="reboot", description="Restart the Raspberry Pi",
                handler=self.reboot, confirm=True, stops_playback=True, speaks=True,
                speech={"en": "restart the Raspberry Pi", "zh": "重启树莓派"},
                phrases={
                    "en": ("reboot", "restart the pi", "reboot the pi"),
                    "zh": ("重启", "重新启动", "重启设备"),
                },
            ),
        ]
