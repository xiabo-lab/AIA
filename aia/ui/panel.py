"""Client for the conversation overlay.

Owns the overlay subprocess and writes messages to it. Every method is
non-blocking and swallows its own failures: the display is a convenience,
and losing it must never cost the user a command. If the overlay dies —
no compositor, no layer-shell, a toolkit error — the assistant carries on
speaking, and the only difference is that nothing appears on screen.

The overlay runs under the *system* interpreter, not the virtualenv, because
PyGObject is a system package and the venv was built without
`--system-site-packages`. Keeping it out of process makes that a feature
rather than a problem to work around.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _system_python() -> str | None:
    """An interpreter that can import gi.

    The venv's python usually cannot — PyGObject is installed as a Debian
    package. Prefer an explicit override, then the usual system paths.
    """
    override = os.environ.get("AIA_OVERLAY_PYTHON")
    candidates = [override] if override else []
    candidates += ["/usr/bin/python3", shutil.which("python3"), sys.executable]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "import gi; gi.require_version('GtkLayerShell','0.1')"],
                capture_output=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


class Panel:
    """The conversation feed: on screen, and written down.

    Both destinations hang off this one object because this is already the
    place every line of the conversation passes through — `main.py` calls
    `user()` and `aia()` at exactly the moments a transcript and a reply exist,
    and threading a second recorder through those same nine call sites would be
    two things to keep in step instead of one.

    The `history` sink is optional and, like the overlay, must never cost the
    user a command: `ConversationLog.record` enqueues and returns without
    touching a disk.
    """

    def __init__(self, enabled: bool = True, history=None):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._history = history
        if not enabled:
            log.info("conversation overlay disabled")
            return

        interpreter = _system_python()
        if interpreter is None:
            log.warning("no interpreter with GtkLayerShell — overlay disabled. "
                        "Install: sudo apt install gir1.2-gtklayershell-0.1")
            return

        env = dict(os.environ)
        # A service inherits no Wayland display; without these the overlay
        # starts, finds no compositor and exits, and the assistant would look
        # like it had silently lost its screen.
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        env["PYTHONPATH"] = str(ROOT)

        try:
            self._proc = subprocess.Popen(
                [interpreter, "-u", "-m", "aia.ui.overlay"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1,
                cwd=str(ROOT),
            )
            log.info("conversation overlay up (%s)", interpreter)
        except OSError as exc:
            log.warning("could not start the overlay: %s", exc)
            self._proc = None

    def _send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        if proc.poll() is not None:
            # It exited on its own; stop trying rather than raising on every
            # turn for the rest of the session.
            log.warning("overlay exited (%s); no further display", proc.returncode)
            self._proc = None
            return
        try:
            with self._lock:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
        except (OSError, ValueError) as exc:
            log.warning("overlay write failed: %s", exc)
            self._proc = None

    def _record(self, role: str, text: str, language: str | None) -> None:
        if self._history is None:
            return
        try:
            self._history.record(role, text, language)
        except Exception:
            # Same contract as the overlay: losing the transcript is a shame,
            # dropping the turn over it would be a fault.
            log.warning("could not record the conversation", exc_info=True)

    def status(self, text: str) -> None:
        """A transient line — replaced by the next real message.

        Not recorded. "Listening…" is the overlay saying it is awake, not
        something anybody said, and storing it would put a line of furniture
        between every real turn of the transcript.
        """
        self._send({"role": "status", "text": text})

    def user(self, text: str, language: str | None = None) -> None:
        self._send({"role": "user", "text": text})
        self._record("user", text, language)

    def aia(self, text: str, language: str | None = None) -> None:
        self._send({"role": "aia", "text": text})
        self._record("aia", text, language)

    def hide(self) -> None:
        self._send({"cmd": "hide"})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
