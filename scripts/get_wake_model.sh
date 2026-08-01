#!/usr/bin/env bash
#
# Fetch the Mandarin wake-word model (Vosk small, ~42 MB download / 66 MB on
# disk). This is the default wake-word backend — see aia/audio/wake.py for why
# a general recogniser is doing a wake word's job.
#
#   ./scripts/get_wake_model.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
NAME="vosk-model-small-cn-0.22"

mkdir -p "$MODELS"
if [[ -d "$MODELS/$NAME" ]]; then
  echo "already have $NAME"
  exit 0
fi

command -v unzip >/dev/null || { echo "need unzip: sudo apt install -y unzip" >&2; exit 1; }

echo "downloading $NAME ..."
curl -fL --progress-bar -o "$MODELS/$NAME.zip" \
  "https://alphacephei.com/vosk/models/$NAME.zip"
unzip -q "$MODELS/$NAME.zip" -d "$MODELS"
rm -f "$MODELS/$NAME.zip"
echo "installed $MODELS/$NAME"
