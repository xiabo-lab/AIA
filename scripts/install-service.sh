#!/usr/bin/env bash
#
# Install AIA as systemd *user* services, so it starts with the desktop
# session — the same way kodama-lite is started.
#
#   ./scripts/install-service.sh            # install, enable, start
#   ./scripts/install-service.sh --uninstall
#
# User services rather than system ones, deliberately: the assistant needs
# the session bus (to drive the music player over MPRIS) and the Wayland
# display (for the conversation overlay). A system service has neither, and
# would have to be handed both by hand.
#
# Note this means AIA starts when the user session does. On a Pi that boots
# to an auto-login desktop that is effectively "on boot". On one that boots
# to a console with no login, nothing starts — and that is correct, because
# there would be no compositor and no session bus to talk to.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
UNITS=(aia-whisper.service aia.service)

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

if [[ "${1:-}" == "--uninstall" ]]; then
  log "Stopping and removing"
  for unit in "${UNITS[@]}"; do
    systemctl --user disable --now "$unit" 2>/dev/null || true
    rm -f "$UNIT_DIR/$unit"
  done
  systemctl --user daemon-reload
  echo "Removed. Anything started by hand is unaffected."
  exit 0
fi

# ── preflight ─────────────────────────────────────────────────────────
log "Checking prerequisites"
missing=0
check() { [[ -e "$2" ]] && echo "  ok   $1" || { echo "  MISSING  $1 ($2)"; missing=1; }; }
check "virtualenv"        "$ROOT/.venv/bin/python"
check "whisper-server"    "$ROOT/vendor/whisper.cpp/build/bin/whisper-server"
check "whisper model"     "$ROOT/models/ggml-base.bin"
check "wake-word model"   "$ROOT/models/vosk-model-small-cn-0.22"
check "English voice"     "$ROOT/models/en_US-lessac-medium.onnx"
check "Mandarin voice"    "$ROOT/models/zh_CN-huayan-medium.onnx"
(( missing )) && { warn "Run scripts/bench_m0.sh and scripts/get_wake_model.sh first."; exit 1; }

# The units hardcode %h/AI_Assit; anywhere else and they would silently point
# at nothing.
[[ "$ROOT" == "$HOME/AI_Assit" ]] || {
  warn "The units expect $HOME/AI_Assit but this checkout is at $ROOT."
  warn "Either move it, or edit systemd/*.service before installing."
  exit 1
}

# ── install ───────────────────────────────────────────────────────────
log "Installing units into $UNIT_DIR"
mkdir -p "$UNIT_DIR"
for unit in "${UNITS[@]}"; do
  install -m 0644 "$ROOT/systemd/$unit" "$UNIT_DIR/$unit"
  echo "  $unit"
done
chmod +x "$ROOT/scripts/whisper-server.sh"
systemctl --user daemon-reload

# Anything already running by hand holds the microphone, and the microphone
# allows exactly one reader — the service would fail to start with an error
# that looks nothing like the real cause.
log "Stopping any hand-started instance"
pkill -f "aia.main" 2>/dev/null || true
pkill -f "aia.ui.overlay" 2>/dev/null || true
pkill -f "whisper-server" 2>/dev/null || true
sleep 2

log "Enabling and starting"
systemctl --user enable --now aia-whisper.service
systemctl --user enable --now aia.service

sleep 12
log "Status"
for unit in "${UNITS[@]}"; do
  printf '  %-22s %s\n' "$unit" "$(systemctl --user is-active "$unit")"
done

cat <<'NOTE'

  Starts automatically with the desktop session from now on.

    journalctl --user -u aia -f        watch a conversation happen
    systemctl --user restart aia       after changing the code
    systemctl --user stop aia          free the microphone (for wake_test.py)
    ./scripts/install-service.sh --uninstall

NOTE
