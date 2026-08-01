"""The conversation overlay: a panel showing what was heard and what was said.

Runs as its own process, reading one JSON object per line on stdin. Two
reasons it is not a thread inside the assistant:

  * GTK wants to own the main loop, and the assistant's main loop is a
    blocking read on the microphone. Neither yields to the other.
  * A toolkit crash, or a compositor that refuses a surface, then takes out
    only the display. The assistant keeps listening and answering, which is
    the part that matters.

It is a **layer-shell** surface rather than an ordinary window. Kodama-Lite
runs full-screen and undecorated, so a normal window would either sit behind
it or steal its focus; a layer-shell overlay sits above everything by
definition and, with keyboard mode NONE, never takes focus at all. labwc —
the compositor on this Pi — implements wlr-layer-shell, so this works.

Protocol, one JSON object per line:

    {"role": "user",   "text": "播放五月天"}
    {"role": "aia",    "text": "正在播放五月天。"}
    {"role": "status", "text": "Listening…"}
    {"cmd": "hide"}

`status` is transient — the next message replaces it — which is what makes
"is it even hearing me?" answerable before the transcript exists.
"""

from __future__ import annotations

import json
import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import GLib, Gtk, GtkLayerShell  # noqa: E402

# How long the panel stays up after the last message before fading out.
IDLE_HIDE_MS = 5000
FADE_MS = 400
FADE_STEPS = 20

# The display is 1920x440 — very wide and very short. A panel much taller
# than this covers the app it is annotating.
WIDTH = 1280
MAX_TURNS = 3

CSS = b"""
.aia-panel {
    background-color: #ffffff;
    border-radius: 18px;
    border: 1px solid #d8d8d8;
    padding: 18px 26px;
}
.aia-speaker {
    color: #6a6a6a;
    font-size: 14px;
    font-weight: bold;
}
.aia-text {
    color: #000000;
    font-size: 24px;
}
.aia-status {
    color: #6a6a6a;
    font-size: 20px;
    font-style: italic;
}
"""


class Overlay:
    def __init__(self) -> None:
        self.window = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)
        self.window.set_default_size(WIDTH, -1)
        self.window.set_app_paintable(True)

        GtkLayerShell.init_for_window(self.window)
        GtkLayerShell.set_layer(self.window, GtkLayerShell.Layer.OVERLAY)
        # Never take focus: the user is talking to the assistant, not typing
        # into it, and stealing focus from the player would be a real bug.
        GtkLayerShell.set_keyboard_mode(self.window, GtkLayerShell.KeyboardMode.NONE)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.BOTTOM, True)
        GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.BOTTOM, 40)
        # No exclusive zone — this floats over the app rather than shrinking it.
        GtkLayerShell.set_exclusive_zone(self.window, 0)

        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(
            self.window.get_screen(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.box.get_style_context().add_class("aia-panel")
        self.box.set_size_request(WIDTH, -1)
        self.window.add(self.box)

        self._turns: list[Gtk.Widget] = []
        self._status: Gtk.Widget | None = None
        self._hide_timer: int | None = None
        self._fade_timer: int | None = None

    # ── rendering ────────────────────────────────────────────────────

    def _row(self, speaker: str, text: str, style: str) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        if speaker:
            label = Gtk.Label(label=speaker, xalign=0.0)
            label.get_style_context().add_class("aia-speaker")
            row.pack_start(label, False, False, 0)
        body = Gtk.Label(label=text, xalign=0.0)
        body.get_style_context().add_class(style)
        # Long transcripts must wrap rather than stretch the panel off-screen.
        body.set_line_wrap(True)
        body.set_max_width_chars(60)
        row.pack_start(body, False, False, 0)
        row.show_all()
        return row

    def _clear_status(self) -> None:
        if self._status is not None:
            self.box.remove(self._status)
            self._status = None

    def show_message(self, role: str, text: str) -> None:
        text = text.strip()
        if not text:
            return

        self._cancel_fade()
        self._clear_status()

        if role == "status":
            self._status = self._row("", text, "aia-status")
            self.box.pack_start(self._status, False, False, 0)
        else:
            speaker = "You" if role == "user" else "AIA"
            row = self._row(speaker, text, "aia-text")
            self.box.pack_start(row, False, False, 0)
            self._turns.append(row)
            # Keep the panel short enough for a 440px-tall screen.
            while len(self._turns) > MAX_TURNS:
                self.box.remove(self._turns.pop(0))

        self.box.set_opacity(1.0)
        self.window.show_all()
        self._restart_hide_timer()

    # ── hiding ───────────────────────────────────────────────────────

    def _restart_hide_timer(self) -> None:
        if self._hide_timer is not None:
            GLib.source_remove(self._hide_timer)
        self._hide_timer = GLib.timeout_add(IDLE_HIDE_MS, self._begin_fade)

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            GLib.source_remove(self._fade_timer)
            self._fade_timer = None

    def _begin_fade(self) -> bool:
        self._hide_timer = None
        step = [0]

        def tick() -> bool:
            step[0] += 1
            if step[0] >= FADE_STEPS:
                self._fade_timer = None
                self.hide()
                return False
            self.box.set_opacity(1.0 - step[0] / FADE_STEPS)
            return True

        self._fade_timer = GLib.timeout_add(FADE_MS // FADE_STEPS, tick)
        return False  # one-shot

    def hide(self) -> None:
        self._cancel_fade()
        if self._hide_timer is not None:
            GLib.source_remove(self._hide_timer)
            self._hide_timer = None
        self.window.hide()
        for row in self._turns:
            self.box.remove(row)
        self._turns.clear()
        self._clear_status()
        self.box.set_opacity(1.0)

    # ── input ────────────────────────────────────────────────────────

    def _handle(self, payload: dict) -> bool:
        if payload.get("cmd") == "hide":
            self.hide()
        else:
            self.show_message(payload.get("role", "aia"), payload.get("text", ""))
        return False  # do not repeat

    def read_stdin(self) -> None:
        """Pump stdin on a worker thread, marshalling onto the GTK loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            GLib.idle_add(self._handle, payload)
        # The assistant closed the pipe, or exited. Nothing left to show.
        GLib.idle_add(Gtk.main_quit)


def main() -> int:
    overlay = Overlay()
    threading.Thread(target=overlay.read_stdin, daemon=True).start()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
