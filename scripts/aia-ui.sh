#!/usr/bin/env bash
#
# Open AIA's conversation UI full-screen on the Pi's display.
#
# Chromium in kiosk mode against the loopback server aia/ui/server.py runs. The
# window covers the whole 1920x440 output, taskbar included — which is the
# point, and which is also the thing to know before pressing it: there is no
# panel to click while this is up. Alt+F4 closes it.
#
# Three flags are load-bearing and were each found by trying without them:
#
#   --ozone-platform=wayland  Chromium probes for X first and exits with
#                             "Missing X server or $DISPLAY" when there is
#                             none. The desktop session usually sets enough
#                             for it to guess right; a launcher run from ssh
#                             or a .desktop file does not, so it is passed
#                             explicitly whenever Wayland is what we are on.
#   --kiosk                   --start-fullscreen alone gave a full-screen
#                             surface with the page still laid out at the
#                             default window size: black borders all round,
#                             the content in a 945x430 box in the middle.
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

# Already open. A second kiosk window on the same profile does not stack
# usefully — it opens behind the first with no way to reach either.
if pgrep -f -- "--user-data-dir=$PROFILE" >/dev/null 2>&1; then
    echo "AIA UI is already open" >&2
    exit 0
fi

# Wait for the server rather than racing it. Clicking this at boot, or straight
# after `systemctl --user restart aia`, otherwise lands on Chromium's own
# "site can't be reached" page, which then needs a reload nobody can reach in
# kiosk mode. Ten seconds is far longer than AIA takes to bind the port once it
# is up, and if it is not running at all the error page is the honest outcome.
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
    --kiosk \
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
