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
import shutil
import subprocess

from aia.plugins.base import CommandSpec, Plugin, Result

log = logging.getLogger(__name__)


class System(Plugin):
    name = "system"
    description = "The Raspberry Pi itself"

    def available(self) -> bool:
        return shutil.which("systemctl") is not None

    def shutdown(self) -> Result:
        """Power the device off.

        Reached only after the user has confirmed — the router refuses to
        fast-path a `confirm=True` command, and main.py requires an explicit
        yes on the following turn.

        Deferred by a few seconds so the spoken confirmation actually
        finishes playing; `systemctl poweroff` kills the audio stack roughly
        instantly, which otherwise cuts the assistant off mid-sentence and
        leaves the user unsure whether it worked.
        """
        try:
            subprocess.Popen(
                ["sh", "-c", "sleep 3 && systemctl poweroff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.error("shutdown failed: %s", exc)
            return Result.failed("I couldn't shut down.", "无法关机。")
        log.warning("shutdown confirmed — powering off in 3 seconds")
        return Result.done("Shutting down. Goodbye.", "正在关机，再见。")

    def reboot(self) -> Result:
        try:
            subprocess.Popen(
                ["sh", "-c", "sleep 3 && systemctl reboot"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.error("reboot failed: %s", exc)
            return Result.failed("I couldn't restart.", "无法重启。")
        log.warning("reboot confirmed — restarting in 3 seconds")
        return Result.done("Restarting.", "正在重启。")

    def commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(
                name="shutdown", description="Power off the Raspberry Pi",
                handler=self.shutdown, confirm=True,
                phrases={
                    "en": ("shut down", "shutdown", "power off", "turn off the pi",
                           "shut down the pi"),
                    "zh": ("关机", "关闭电源", "把树莓派关机", "关掉设备"),
                },
            ),
            CommandSpec(
                name="reboot", description="Restart the Raspberry Pi",
                handler=self.reboot, confirm=True,
                phrases={
                    "en": ("reboot", "restart the pi", "reboot the pi"),
                    "zh": ("重启", "重新启动", "重启设备"),
                },
            ),
        ]
