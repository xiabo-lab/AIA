#!/usr/bin/env bash
#
# Run whisper-server in the foreground with AIA's canonical flags.
#
# The single definition of those flags. Both `run_services.sh` (for a hand
# start) and the `aia-whisper.service` unit (for boot) exec this, so the
# assistant cannot end up transcribing with different settings depending on
# how it happened to be started — which would be an unpleasant thing to
# debug, since the symptom would be latency or accuracy quietly changing.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN="$ROOT/vendor/whisper.cpp/build/bin/whisper-server"
MODEL="$ROOT/models/ggml-base.bin"
PORT="${AIA_WHISPER_PORT:-8081}"

# See docs/PLAN.md. 512 caps Whisper's padded 30 s encoder window to ~10 s —
# a 4.1x speedup with no measured accuracy cost. 256 makes the decoder loop.
AUDIO_CTX=512

[[ -x "$BIN" ]]   || { echo "missing $BIN — run scripts/bench_m0.sh first" >&2; exit 1; }
[[ -s "$MODEL" ]] || { echo "missing $MODEL — run scripts/bench_m0.sh first" >&2; exit 1; }

# `-l auto` detects the spoken language per request, which is what makes
# switching between English and Mandarin mid-conversation work. It costs
# ~390 ms against naming a language outright; aia/stt/engine.py documents the
# cheaper scheme that was tried and removed.
exec "$BIN" \
  -m "$MODEL" \
  -t "$(nproc)" \
  -ac "$AUDIO_CTX" \
  -l auto \
  --port "$PORT"
