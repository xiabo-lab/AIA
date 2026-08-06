#!/usr/bin/env bash
#
# Fetch the SenseVoiceSmall model AIA transcribes with.
#
#   ./scripts/get_sensevoice.sh
#
# Run once, on the Pi. The model is ~230 MB of ONNX and is deliberately not in
# git — `git archive | ssh` is how this project deploys, and putting a quarter
# of a gigabyte of weights through that on every deploy would be unkind.
#
# STT is offline. This is the only step that touches a network, it happens
# once, and nothing at runtime downloads anything or calls an API. If this
# script has not been run, AIA says so at startup and exits rather than
# quietly falling back to something that phones home.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"

# Keep this name in step with SenseVoiceConfig.directory in aia/core/config.py.
# The two are not derived from each other, so a rename here without a rename
# there presents as "model is missing" against a directory that is plainly on
# disk.
NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"

DEST="$MODELS/$NAME"

# The INT8 weights specifically. The archive also carries fp32 `model.onnx`,
# which works and is roughly twice the size and slower — a wrong choice that
# would present as disappointing benchmark numbers rather than as an error.
MODEL="$DEST/model.int8.onnx"
TOKENS="$DEST/tokens.txt"

if [[ -s "$MODEL" && -s "$TOKENS" ]]; then
  echo "SenseVoice is already at $DEST"
  echo "  $(du -h "$MODEL" | cut -f1)  $(basename "$MODEL")"
  exit 0
fi

mkdir -p "$MODELS"
cd "$MODELS"

echo "Fetching $NAME (~230 MB)…"
# --location because the release URL redirects to a CDN, and a redirect
# followed as a 40-byte HTML body would extract as a corrupt archive.
curl -fL --progress-bar -o "${NAME}.tar.bz2" "$URL"

echo "Extracting…"
tar xf "${NAME}.tar.bz2"
rm -f "${NAME}.tar.bz2"

# Prove what was asked for is what arrived, rather than trusting that tar
# exited zero over an archive with a different layout inside it.
for f in "$MODEL" "$TOKENS"; do
  [[ -s "$f" ]] || { echo "expected $f in the archive and it is not there" >&2; exit 1; }
done

echo
echo "SenseVoice ready:"
echo "  model  $MODEL  ($(du -h "$MODEL" | cut -f1))"
echo "  tokens $TOKENS"
echo
echo "Check it end to end with:  .venv/bin/python scripts/stt_test.py --selftest"
