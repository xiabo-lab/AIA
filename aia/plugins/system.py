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
import socket
import subprocess
import threading
import time

from aia.plugins.base import CommandSpec, Plugin, Result

log = logging.getLogger(__name__)

# What "online" is asked of. The address is contacted directly so the answer
# does not depend on name resolution, and the name is one the device actually
# needs — the player streams from it, and the failure that prompted this
# command was `Failed to resolve 'www.youtube.com'` while the link itself was
# up. Those two states look identical from the sofa and want different fixes.
PROBE_ADDRESS = ("1.1.1.1", 53)
PROBE_NAME = "www.youtube.com"

# Both probes run at once and the pair is given this long. A network answer is
# worth waiting a moment for, but not the whole 2.5 s turn — and an
# unreachable host is exactly the case that would otherwise sit until a TCP
# timeout that nobody chose.
PROBE_TIMEOUT_S = 1.0


def _reachable(address: tuple[str, int], timeout: float) -> bool:
    try:
        with socket.create_connection(address, timeout=timeout):
            return True
    except OSError as exc:
        log.debug("no route to %s: %s", address, exc)
        return False


def _resolves(name: str) -> bool:
    try:
        socket.getaddrinfo(name, 443, proto=socket.IPPROTO_TCP)
        return True
    except OSError as exc:
        log.debug("cannot resolve %s: %s", name, exc)
        return False


def _in_background(fn, *args) -> tuple[threading.Thread, dict]:
    """Run `fn` on a daemon thread and hand back somewhere to read the answer.

    A plain thread rather than `ThreadPoolExecutor`, and the reason is the
    whole point of this probe: the executor's context manager calls
    `shutdown(wait=True)` on the way out, so a `getaddrinfo` that hangs blocks
    the turn for as long as the resolver takes no matter what timeout the
    caller passed to `result()`. Measured at 30 s against a stub. A daemon
    thread is simply abandoned, which is the correct treatment for an answer
    that has already arrived too late to use.
    """
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn(*args)
        except Exception:
            log.debug("network probe raised", exc_info=True)
            box["value"] = False

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


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

    def network(self) -> Result:
        """Is there internet, and if not, which half is broken.

        Three answers rather than two, because the two failures want different
        things done about them. A link that is up while names do not resolve is
        what the 5G hotspot actually did: the player asked for
        `www.youtube.com`, got `Temporary failure in name resolution`, retried
        for four minutes and died, while everything else looked connected.
        "Offline" would have been the wrong thing to say and "online" worse.

        Nothing else in AIA touches the network — this is the only command that
        does, and only when it is asked.
        """
        deadline = time.monotonic() + PROBE_TIMEOUT_S
        route_thread, route_box = _in_background(
            _reachable, PROBE_ADDRESS, PROBE_TIMEOUT_S)
        names_thread, names_box = _in_background(_resolves, PROBE_NAME)

        # Both are already running; joining in turn against one shared deadline
        # costs the slower of the two rather than the sum.
        for thread in (route_thread, names_thread):
            thread.join(max(0.0, deadline - time.monotonic()))

        # A probe that has not answered has not answered. `getaddrinfo` has no
        # timeout of its own, so this is the only thing bounding it.
        has_route = route_box.get("value", False)
        has_names = names_box.get("value", False)

        log.info("network: route=%s names=%s", has_route, has_names)
        if has_route and has_names:
            return Result.done("Online.", "网络正常。")
        if has_route:
            return Result.done(
                "Connected, but name lookups are failing.",
                "已连接，但是域名解析失败。",
            )
        return Result.done("Offline.", "网络已断开。")

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
            # Speaks, for the same reason `now_playing` does: the answer exists
            # nowhere but in the reply. Nothing on screen changes when you ask
            # whether there is internet.
            CommandSpec(
                name="network", description="Say whether the internet is reachable",
                handler=self.network, speaks=True,
                phrases={
                    "en": ("network status", "is the internet working",
                           "are we online", "check the network",
                           "do we have internet"),
                    "zh": ("网络状态", "有没有网络", "网络怎么样",
                           "网络正常吗", "检查网络", "能上网吗"),
                },
            ),
        ]
