#!/usr/bin/env bash
#
# Open AIA's conversation UI as an app window on the Pi's display.
#
# Chromium against the loopback server aia/ui/server.py runs, maximised — which
# is what Kodama-Lite does, and it is deliberate rather than a compromise.
#
# **Do not make this --kiosk.** Full-screen covers the taskbar, and on this
# machine that is a one-way door: labwc here has no Close binding at all (its
# rc.xml binds Maximize, UnMaximize, volume and magnify — there is no Alt+F4),
# and window.close() from the page does not close a Chromium app window either,
# tested. On a touch-only display a full-screen window with no panel and no
# titlebar cannot be dismissed. Maximised costs 28 px of 440 and leaves both
# the taskbar and a titlebar close button reachable by finger.
#
# Three flags are load-bearing and were each found by trying without them:
#
#   --ozone-platform=wayland  Chromium probes for X first and exits with
#                             "Missing X server or $DISPLAY" when there is
#                             none. The desktop session usually sets enough
#                             for it to guess right; a launcher run from ssh
#                             or a .desktop file does not, so it is passed
#                             explicitly whenever Wayland is what we are on.
#   --start-maximized         and not --start-fullscreen, which gave a
#                             full-screen surface with the page still laid out
#                             at the default window size: black borders all
#                             round, the content in a 945x430 box in the middle.
#   --user-data-dir           its own profile. Sharing the default one means
#                             this window joins the browsing session — it
#                             would restore tabs, offer to reopen pages after
#                             a crash, and inherit whatever zoom was last set
#                             on 127.0.0.1.
#
# The touch flags matter on this display too: without --disable-pinch a stray
# two-finger touch zooms an appliance UI that has no way to zoom back.

set -euo pipefail

URL="http://127.0.0.1:8090"
PROFILE="${XDG_DATA_HOME:-$HOME/.local/share}/aia-ui"

# Already open. A second window on the same profile does not stack usefully —
# it opens behind the first, and both carry the same taskbar entry.
if pgrep -f -- "--user-data-dir=$PROFILE" >/dev/null 2>&1; then
    echo "AIA UI is already open" >&2
    exit 0
fi

# Wait for the server rather than racing it. Clicking this at boot, or straight
# after `systemctl --user restart aia`, otherwise lands on Chromium's own "site
# can't be reached" page — and an --app window has no address bar or reload
# button to get off it with. Ten seconds is far longer than AIA takes to bind
# the port once it is up, and if it is not running at all the error page is the
# honest outcome.
for _ in $(seq 1 20); do
    if curl -fsS -o /dev/null --max-time 1 "$URL/api/feed?since=0" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

platform=()
[ -n "${WAYLAND_DISPLAY:-}" ] && platform=(--ozone-platform=wayland)

exec chromium \
    "${platform[@]}" \
    --start-maximized \
    --app="$URL" \
    --user-data-dir="$PROFILE" \
    --class=aia-ui \
    --force-device-scale-factor=1 \
    --no-first-run \
    --no-default-browser-check \
    --noerrdialogs \
    --disable-session-crashed-bubble \
    --disable-infobars \
    --disable-pinch \
    --overscroll-history-navigation=0
