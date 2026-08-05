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

# Whether systemd owns a process, so this script can defer to it instead of
# fighting it. `run_services.sh stop` used to SIGTERM the server out from
# under aia-whisper.service, and systemd counts a SIGTERM exit as *success*,
# so `Restart=on-failure` never fired: the unit went inactive and stayed
# there, while aia.service carried on listening and answering with no
# transcriber behind it. Measured — active, pid 1068, then inactive, pid 0,
# NRestarts 0, and the assistant still "running".
unit_active() { systemctl --user is-active --quiet "$1" 2>/dev/null; }

# `pkill -f` matches any command line *containing* the pattern, which
# includes the shell that invoked it — a `bash -c '… pkill -f llama-server …'`
# kills itself, and does so with an exit code that reads like the target
# refusing to die. `-x` matches the process name exactly and cannot do that.
# Kept for the two servers, whose names are short enough to survive comm's
# 15-character truncation ("whisper-server" is 14, "llama-server" is 12).
running() { pgrep -x "$1" >/dev/null; }

# See docs/PLAN.md: 512 caps Whisper's padded 30 s encoder window to ~10 s,
# a 4.1x speedup with no measured accuracy cost. 256 makes the decoder loop.
AUDIO_CTX=512

start_whisper() {
  if unit_active aia-whisper.service; then
    echo "aia-whisper.service is running it — leaving systemd in charge"
    return 0
  fi
  running whisper-server && { echo "whisper-server already running"; return; }
  [[ -x "$WHISPER" ]] || { echo "missing $WHISPER — run scripts/bench_m0.sh first"; return 1; }
  # Flags live in scripts/whisper-server.sh, which the systemd unit also
  # execs, so a hand start and a boot start cannot end up differing.
  setsid nohup "$ROOT/scripts/whisper-server.sh" \
    > "$LOGS/whisper-server.log" 2>&1 < /dev/null &
  echo "whisper-server starting on :$WHISPER_PORT"
  return 0
}

start_llama() {
  running llama-server && { echo "llama-server already running"; return; }
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
    # Stop it the way it was started. Killing a systemd-managed server
    # directly leaves the unit inactive with nothing to restart it, and the
    # assistant deaf but still answering.
    if unit_active aia-whisper.service; then
      systemctl --user stop aia-whisper.service && echo "stopped aia-whisper.service"
    elif pkill -x whisper-server; then
      echo "stopped whisper-server"
    fi
    pkill -x llama-server && echo "stopped llama-server"
    ;;
  status)
    if unit_active aia-whisper.service; then
      echo "whisper-server: up (aia-whisper.service)"
    elif running whisper-server; then
      echo "whisper-server: up (started by hand)"
    else
      echo "whisper-server: down"
    fi
    running llama-server && echo "llama-server:   up" || echo "llama-server:   down"
    ;;
  logs)
    tail -n 40 "$LOGS"/*-server.log
    ;;
  *)
    echo "usage: $0 start [--with-llm] | stop | status | logs" >&2
    exit 2
    ;;
esac
