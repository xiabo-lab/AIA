"""The conversation and settings web UI.

A single static page and two JSON endpoints, served from a daemon thread inside
the assistant process. Deliberately the smallest thing that answers the two
questions people actually have — *what did it hear me say?* and *what is it
running?* — because this shares a Pi 5 with a wake recogniser, whisper.cpp and
a music player, all of which want the same four cores.

Why this and not more of the GTK overlay: the overlay is a layer-shell surface
with keyboard mode NONE, which is what stops it stealing focus from the
full-screen player, and it is therefore a thing you cannot scroll. Scrollback
and a settings page need input. Putting them in a browser also keeps PyGObject
out of the venv, which `ui/panel.py` already explains at length.

**Loopback only.** See `UiConfig`. This serves a transcript of everything said
in the room and there is no authentication in front of it.

Failure here is never fatal. A port already in use, a missing page file — the
assistant logs it and carries on listening, exactly as it does when the overlay
cannot start.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aia.core.config import UiConfig

log = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parent / "web" / "index.html"

# Anything above this and a client that has been away for hours would pull the
# whole retention window in one request. The page asks again immediately when
# it receives a full page, so nothing is lost — it just arrives in batches.
MAX_LIMIT = 500


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 keeps this simple: no keep-alive bookkeeping, and a browser
    # polling once a second does not need it.
    protocol_version = "HTTP/1.0"
    server_version = "AIA"

    # Injected by `WebUI.start`.
    ui: "WebUI" = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        """Route access logs to DEBUG.

        The default writes every request to stderr, which on this device is the
        journal — a page polling once a second would put 86,000 lines a day
        into the log people read to find out why a turn failed.
        """
        log.debug("%s %s", self.address_string(), fmt % args)

    # ── responses ────────────────────────────────────────────────────

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The transcript changes every turn; a cached feed would show a
        # conversation that has already moved on.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The tab was closed mid-response. Not worth a stack trace.
            log.debug("client went away mid-response")

    def _json(self, payload: dict, code: int = 200) -> None:
        # ensure_ascii=False so Mandarin arrives as Mandarin rather than as
        # escapes; the charset says how to read it.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    # ── routes ───────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        route = urlparse(self.path)
        params = parse_qs(route.query)

        if route.path in ("/", "/index.html"):
            self._page()
        elif route.path == "/api/feed":
            self._feed(params)
        elif route.path == "/api/system":
            self._system()
        elif route.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        """Nothing here accepts input.

        Stated as a refusal rather than left to fall through as a 501, because
        "the UI cannot change anything" is a property worth being explicit
        about: the only way to talk to this assistant is to talk to it.
        """
        self._json({"error": "this interface is read-only"}, 405)

    def _page(self) -> None:
        body = self.ui.page()
        if body is None:
            self._json({"error": "the page is missing from this deployment"}, 500)
            return
        self._send(200, body, "text/html; charset=utf-8")

    def _feed(self, params: dict) -> None:
        try:
            since = int(params.get("since", ["0"])[0])
        except ValueError:
            since = 0
        try:
            limit = int(params.get("limit", [str(self.ui.cfg.backlog)])[0])
        except ValueError:
            limit = self.ui.cfg.backlog
        limit = max(1, min(limit, MAX_LIMIT))

        messages = self.ui.history.recent(since_id=since, limit=limit)
        self._json({
            "messages": messages,
            "state": self.ui.state(),
            "retention_hours": self.ui.retention_hours,
            # The browser and the Pi can disagree about the time, and the
            # timestamps below are the Pi's. Sending its clock lets the page
            # render ages that match the device rather than the viewer.
            "now": time.time(),
        })

    def _system(self) -> None:
        try:
            self._json(self.ui.info.snapshot())
        except Exception:
            log.exception("could not build the system snapshot")
            self._json({"error": "system information is unavailable"}, 500)


class WebUI:
    """Owns the HTTP server thread."""

    def __init__(self, cfg: UiConfig, *, history, info, state, retention_hours: float):
        self.cfg = cfg
        self.history = history
        self.info = info
        # A callable rather than a value: the state changes several times per
        # turn and the server must never hold a stale copy of it.
        self.state = state
        self.retention_hours = retention_hours
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._page: bytes | None = None

    def page(self) -> bytes | None:
        """The single page, read once and held.

        One file, served from a fixed path — there is no static directory and
        no path joining anywhere in this module, so there is nothing for a
        crafted URL to traverse into.
        """
        if self._page is None:
            try:
                self._page = PAGE.read_bytes()
            except OSError as exc:
                log.error("could not read %s: %s", PAGE, exc)
                return None
        return self._page

    def start(self) -> None:
        if not self.cfg.enabled:
            log.info("web UI disabled")
            return

        handler = type("_BoundHandler", (_Handler,), {"ui": self})
        try:
            self._server = ThreadingHTTPServer((self.cfg.host, self.cfg.port), handler)
        except OSError as exc:
            # Usually the port is taken by a previous instance that has not
            # finished dying. Not a reason to refuse to listen to anybody.
            log.warning("could not start the web UI on %s:%d — %s",
                        self.cfg.host, self.cfg.port, exc)
            self._server = None
            return

        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="aia-web", daemon=True)
        self._thread.start()
        log.info("web UI at http://%s:%d", self.cfg.host, self.cfg.port)

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            log.debug("shutting down the web UI failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
