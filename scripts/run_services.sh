#!/usr/bin/env bash
#
# Start/stop the resident model servers AIA talks to.
#
# Both models must stay in memory — that is an architectural requirement, not
# an optimisation. Loading ggml-base costs ~96 ms and a Piper voice up to
# 1190 ms; paying either per utterance misses the latency budget outright.
# (Piper is resident too, but as a child of the assistant process — see
# aia/tts/piper.py — so it is not managed here.)
#
#   ./scripts/run_services.sh start|stop|status|logs

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/.bench"
mkdir -p "$LOGS"

WHISPER="$ROOT/vendor/whisper.cpp/build/bin/whisper-server"
LLAMA="$ROOT/vendor/llama.cpp/build/bin/llama-server"
WHISPER_MODEL="$ROOT/models/ggml-base.bin"
LLAMA_MODEL="$ROOT/models/qwen2.5-3b-instruct-q4_k_m.gguf"

WHISPER_PORT=8081
LLAMA_PORT=8080
THREADS=$(nproc)

# See docs/PLAN.md: 512 caps Whisper's padded 30 s encoder window to ~10 s,
# a 4.1x speedup with no measured accuracy cost. 256 makes the decoder loop.
AUDIO_CTX=512

start_whisper() {
  pgrep -f "whisper-server" >/dev/null && { echo "whisper-server already running"; return; }
  [[ -x "$WHISPER" ]] || { echo "missing $WHISPER — run scripts/bench_m0.sh first"; return 1; }
  # Explicit -l en, not auto: auto-detect costs a second encoder pass (~400 ms
  # per utterance). The client overrides language per request and falls back to
  # the other language on a script mismatch — see aia/stt/engine.py.
  # `-l auto` detects the spoken language per request, which is what makes
  # switching between English and Mandarin mid-conversation work. It costs
  # ~390 ms against naming a language outright; aia/stt/engine.py documents the
  # cheaper scheme that was tried and why it was removed.
  #
  # Temperature fallback is deliberately left ON. Disabling it was tested to
  # cap a pathological 10.4 s pass and made things worse — Whisper stopped
  # producing detectable garbage and started producing confident
  # mistranslations instead. Fallback only engages when a decode genuinely
  # fails, which auto-detect makes rare.
  setsid nohup "$WHISPER" -m "$WHISPER_MODEL" -t "$THREADS" -ac "$AUDIO_CTX" \
    -l auto --port "$WHISPER_PORT" > "$LOGS/whisper-server.log" 2>&1 < /dev/null &
  echo "whisper-server starting on :$WHISPER_PORT"
}

start_llama() {
  pgrep -f "llama-server" >/dev/null && { echo "llama-server already running"; return; }
  [[ -x "$LLAMA" ]] || { echo "missing $LLAMA — run scripts/bench_m0.sh first"; return 1; }
  # --cache-reuse keeps the plugin-manifest system prompt in the KV cache
  # between turns. Without it every turn re-prefills ~800 tokens at 28 tok/s,
  # which is ~28 s of latency that no amount of routing can hide.
  setsid nohup "$LLAMA" -m "$LLAMA_MODEL" -t "$THREADS" \
    --port "$LLAMA_PORT" --ctx-size 4096 --cache-reuse 256 \
    > "$LOGS/llama-server.log" 2>&1 < /dev/null &
  echo "llama-server starting on :$LLAMA_PORT (M2 uses this; M1 does not)"
}

case "${1:-start}" in
  start)
    start_whisper
    [[ "${2:-}" == "--with-llm" ]] && start_llama
    ;;
  stop)
    pkill -f whisper-server && echo "stopped whisper-server"
    pkill -f llama-server   && echo "stopped llama-server"
    ;;
  status)
    pgrep -f whisper-server >/dev/null && echo "whisper-server: up" || echo "whisper-server: down"
    pgrep -f llama-server   >/dev/null && echo "llama-server:   up" || echo "llama-server:   down"
    ;;
  logs)
    tail -n 40 "$LOGS"/*-server.log
    ;;
  *)
    echo "usage: $0 start [--with-llm] | stop | status | logs" >&2
    exit 2
    ;;
esac
