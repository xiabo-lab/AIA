#!/usr/bin/env bash
#
# M0 — hardware gate for AIA.
#
# Every performance number in docs/PLAN.md comes from published benchmarks on
# someone else's Raspberry Pi. This script measures *this* Pi and prints a
# PASS/FAIL table against the assumptions the design rests on. If a row fails,
# re-plan before writing any AIA code — that is the entire point of M0.
#
# What it measures, and why that number matters:
#
#   whisper.cpp base, ~2 s clip   the biggest fixed cost on the fast path
#   llama.cpp prefill (pp)        decides whether a plugin-manifest system
#                                 prompt is affordable at all
#   llama.cpp decode (tg)         decides whether the LLM can ever be on the
#                                 fast path (spoiler: it cannot)
#   Piper time-to-first-audio     the tail of every single response
#   MPRIS reachability            whether Kodama-Lite can be driven today
#
# Usage:
#   ./scripts/bench_m0.sh [--skip-download] [--quick]
#
# First run needs ~3 GB of disk and ~20 minutes, mostly compiling. Later runs
# reuse everything in models/ and vendor/.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
VENDOR="$ROOT/vendor"
WORK="$ROOT/.bench"

SKIP_DOWNLOAD=0
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --quick)         QUICK=1 ;;
    -h|--help)       sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$MODELS" "$VENDOR" "$WORK"

# ── Thresholds ────────────────────────────────────────────────────────
# Derived from the latency budget in docs/PLAN.md. Loosened slightly from
# the published figures so normal run-to-run variance does not fail a Pi
# that is actually fine.
T_WHISPER_MS=900       # transcribe ~2 s of audio within the budget slot
T_LLAMA_PP=25          # tok/s prefill (published: ~33)
T_LLAMA_TG=4.5         # tok/s decode  (published: ~5-8)
T_PIPER_MS=400         # to first audio sample, warm
T_DISPATCH_MS=100      # MPRIS command round trip

# Whisper runs its encoder over a fixed 30 s window no matter how short the
# clip is, which is why a 2 s command cost 1.56 s of encode at the default
# 1500. Capping the audio context to ~10 s cuts that to ~0.38 s and — measured
# on this hardware, 2026-08-01 — does not degrade the transcript at all.
# Do not lower it further: at 256 the decoder starts falling back and looping,
# and total time goes *up* despite the faster encode.
WHISPER_AC=512

# ── Output helpers ────────────────────────────────────────────────────
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YEL=$'\033[33m'; RST=$'\033[0m'
[[ -t 1 ]] || { BOLD=""; DIM=""; RED=""; GRN=""; YEL=""; RST=""; }

RESULTS=()   # "status|label|measured|threshold"
record() { RESULTS+=("$1|$2|$3|$4"); }
step()   { printf '\n%s══ %s%s\n' "$BOLD" "$1" "$RST"; }
info()   { printf '%s   %s%s\n' "$DIM" "$1" "$RST"; }
warn()   { printf '%s   ! %s%s\n' "$YEL" "$1" "$RST"; }

# Download to $1 from $2, unless it already exists.
fetch() {
  local dest="$1" url="$2"
  [[ -s "$dest" ]] && { info "have $(basename "$dest")"; return 0; }
  if (( SKIP_DOWNLOAD )); then
    warn "missing $(basename "$dest") and --skip-download given"
    return 1
  fi
  info "downloading $(basename "$dest")"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url" || {
    warn "download failed: $url"; rm -f "$dest.part"; return 1
  }
  mv "$dest.part" "$dest"
}

# Median of numbers on stdin. Keeps one slow first run from skewing a result.
median() {
  sort -g | awk '{a[NR]=$1} END{ if(NR==0){print "0";exit}
    print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2 }'
}

# ── 0. Platform ───────────────────────────────────────────────────────
step "Platform"

MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
info "$MODEL"
case "$MODEL" in
  *"Raspberry Pi 5"*) ;;
  *) warn "not a Pi 5 — every threshold below assumes Pi 5 (8 GB)" ;;
esac

MEM_MB=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)
info "RAM: ${MEM_MB} MB"
(( MEM_MB < 7000 )) && warn "under 8 GB; the 3B model plus Kodama-Lite will be tight"

# Throttling is the classic silent benchmark-wrecker. Bit 0 = currently
# throttled, bit 2 = currently capped, bits 16+ = has happened since boot.
if command -v vcgencmd >/dev/null 2>&1; then
  THR=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
  info "throttle flags: $THR"
  if [[ "$THR" != "0x0" ]]; then
    warn "throttling detected ($THR) — results will understate this Pi."
    warn "check the PSU and that the active cooler is fitted, then re-run."
  fi
fi

# NVMe vs SD card shows up mostly in model load time, i.e. boot budget.
ROOTDEV=$(findmnt -no SOURCE / 2>/dev/null || echo "?")
info "root device: $ROOTDEV"
[[ "$ROOTDEV" == *mmcblk* ]] && warn "booting from SD; 2 GB model load will be slow (boot budget risk)"

TEMP=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2 || echo "?")
info "temp at start: $TEMP"

# ── 1. Build dependencies ─────────────────────────────────────────────
step "Build dependencies"

MISSING=()
for c in git cmake make g++ curl python3; do
  command -v "$c" >/dev/null 2>&1 || MISSING+=("$c")
done
if (( ${#MISSING[@]} )); then
  warn "missing: ${MISSING[*]}"
  echo "   install with: sudo apt update && sudo apt install -y git cmake build-essential curl python3"
  exit 1
fi
info "all present"
NPROC=$(nproc)
info "building with -j$NPROC"

# ── 2. whisper.cpp ────────────────────────────────────────────────────
step "whisper.cpp — speech recognition"

WHISPER_DIR="$VENDOR/whisper.cpp"
if [[ ! -d "$WHISPER_DIR" ]]; then
  (( SKIP_DOWNLOAD )) && { warn "not built and --skip-download given"; }
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR" \
    || warn "clone failed"
fi

WHISPER_BIN=""
if [[ -d "$WHISPER_DIR" ]]; then
  if [[ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]]; then
    info "compiling (a few minutes)..."
    cmake -B "$WHISPER_DIR/build" -S "$WHISPER_DIR" -DCMAKE_BUILD_TYPE=Release \
      >"$WORK/whisper_cmake.log" 2>&1 \
      && cmake --build "$WHISPER_DIR/build" -j"$NPROC" --config Release \
        >>"$WORK/whisper_cmake.log" 2>&1 \
      || warn "build failed — see $WORK/whisper_cmake.log"
  fi
  [[ -x "$WHISPER_DIR/build/bin/whisper-cli" ]] && WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
fi

fetch "$MODELS/ggml-base.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"

# ── 3. Piper ──────────────────────────────────────────────────────────
# Done before the whisper run, because Piper generates the Mandarin test
# clip that whisper is then timed on. That keeps M0 self-contained with no
# external audio assets — but it does mean the clip is synthetic and clean.
# M0 measures SPEED honestly; it does not measure accuracy. Real accuracy
# needs the actual USB mic in the actual room (see the STT gate in the plan).
step "Piper — speech synthesis"

PIPER_BIN=""
if [[ ! -x "$VENDOR/piper/piper" ]]; then
  if fetch "$WORK/piper.tar.gz" \
      "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"; then
    tar -xzf "$WORK/piper.tar.gz" -C "$VENDOR" && info "extracted piper"
  fi
fi
[[ -x "$VENDOR/piper/piper" ]] && PIPER_BIN="$VENDOR/piper/piper"

PV="https://huggingface.co/rhasspy/piper-voices/resolve/main"
fetch "$MODELS/en_US-lessac-medium.onnx"      "$PV/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
fetch "$MODELS/en_US-lessac-medium.onnx.json" "$PV/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
fetch "$MODELS/zh_CN-huayan-medium.onnx"      "$PV/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx"
fetch "$MODELS/zh_CN-huayan-medium.onnx.json" "$PV/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"

# Warm per-utterance synthesis cost, with model load factored out.
#
# Piper is dominated by loading the ONNX voice: measured here, 452 ms for the
# English voice and 1190 ms for the Mandarin one, against 298 ms / 133 ms to
# actually synthesise a sentence. AIA keeps Piper resident, so it pays the load
# once at boot. Timing a fresh `piper` process per utterance — which is what an
# earlier version of this script did — reported 672 ms and failed a threshold
# the real design never has to meet.
#
# Separated by running N utterances through one process: T(n) = load + n*synth,
# so synth = (T(10) - T(1)) / 9.
piper_run_ms() {
  local voice="$1" n="$2" start end
  start=$(date +%s%N)
  { for _ in $(seq "$n"); do echo "Playing Hotel California by the Eagles."; done; } \
    | "$PIPER_BIN" --model "$voice" --output_dir "$WORK/tts" >/dev/null 2>&1
  end=$(date +%s%N)
  echo $(( (end - start) / 1000000 ))
}

piper_warm_ms() {
  local voice="$1" t1 t10
  mkdir -p "$WORK/tts"
  t1=$(piper_run_ms "$voice" 1)
  t10=$(piper_run_ms "$voice" 10)
  awk -v a="$t1" -v b="$t10" 'BEGIN{ s=(b-a)/9; printf "%.0f\n", (s<0?0:s) }'
}

if [[ -n "$PIPER_BIN" && -s "$MODELS/en_US-lessac-medium.onnx" ]]; then
  info "measuring warm per-utterance synthesis..."
  for V in en_US-lessac-medium zh_CN-huayan-medium; do
    [[ -s "$MODELS/$V.onnx" ]] || { record SKIP "Piper synth ($V)" "voice missing" "<= ${T_PIPER_MS} ms"; continue; }
    MS=$(piper_warm_ms "$MODELS/$V.onnx")
    case "$V" in en_US-*) PIPER_EN_MS="$MS" ;; zh_CN-*) PIPER_ZH_MS="$MS" ;; esac
    LBL="Piper synth, ${V%%-*}"
    if [[ -n "$MS" && "$MS" != "0" ]]; then
      if (( MS <= T_PIPER_MS )); then
        record PASS "$LBL (warm)" "${MS} ms" "<= ${T_PIPER_MS} ms"
      else
        record FAIL "$LBL (warm)" "${MS} ms" "<= ${T_PIPER_MS} ms"
      fi
    else
      record SKIP "$LBL (warm)" "no output" "<= ${T_PIPER_MS} ms"
    fi
  done

  # Generate the Mandarin test clip for whisper.
  "$PIPER_BIN" --model "$MODELS/zh_CN-huayan-medium.onnx" \
    --output_file "$WORK/zh.wav" <<<"播放周杰伦的歌曲，音量调到百分之五十。" >/dev/null 2>&1
else
  record SKIP "Piper time-to-first-audio" "piper unavailable" "<= ${T_PIPER_MS} ms"
fi

# ── 4. whisper timing ─────────────────────────────────────────────────
step "whisper.cpp timing"

# whisper.cpp requires 16 kHz mono PCM.
prep_wav() {
  local src="$1" dst="$2"
  command -v ffmpeg >/dev/null 2>&1 || return 1
  ffmpeg -y -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$dst" >/dev/null 2>&1
}

# Steady-state transcription cost: total time minus model load.
#
# AIA holds the model resident for the life of the process, so load time is
# paid once at boot and never again on the critical path. Charging it to every
# utterance — which is what timing the CLI naively does — overstates the real
# per-command cost by ~100 ms and measures an architecture we are not building.
#
# NB: no -np here. `--no-prints` suppresses the timing block this parses,
# which would silently turn every whisper row into a SKIP.
whisper_ms() {
  local wav="$1" lang="$2"
  "$WHISPER_BIN" -m "$MODELS/ggml-base.bin" -f "$wav" -l "$lang" \
    -t "$NPROC" -nt -ac "$WHISPER_AC" 2>&1 \
    | awk '
        /load time/  { for(i=1;i<=NF;i++) if($i=="=") { load=$(i+1); break } }
        /total time/ { for(i=1;i<=NF;i++) if($i=="=") { tot=$(i+1);  break } }
        END { if (tot=="") exit; print (load=="" ? tot : tot-load) }'
}

if [[ -n "$WHISPER_BIN" && -s "$MODELS/ggml-base.bin" ]]; then
  REPS=$(( QUICK ? 1 : 3 ))

  # English: whisper.cpp ships jfk.wav (~11 s). Trim to ~2 s so the number
  # is comparable to a real spoken command rather than a long sample.
  EN_WAV="$WHISPER_DIR/samples/jfk.wav"
  if [[ -s "$EN_WAV" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -i "$EN_WAV" -t 2 -ar 16000 -ac 1 "$WORK/en2s.wav" >/dev/null 2>&1 \
        && EN_WAV="$WORK/en2s.wav"
    else
      warn "no ffmpeg; timing the full ~11 s sample instead of a 2 s command"
      warn "install with: sudo apt install -y ffmpeg   (then re-run)"
    fi
    info "English clip..."
    EN_MS=$(for _ in $(seq "$REPS"); do whisper_ms "$EN_WAV" en; done | median)
    if [[ -n "$EN_MS" && "$EN_MS" != "0" ]]; then
      if (( ${EN_MS%.*} <= T_WHISPER_MS )); then
        record PASS "whisper base, English ~2 s" "${EN_MS} ms" "<= ${T_WHISPER_MS} ms"
      else
        record FAIL "whisper base, English ~2 s" "${EN_MS} ms" "<= ${T_WHISPER_MS} ms"
      fi
    else
      record SKIP "whisper base, English ~2 s" "no timing parsed" "<= ${T_WHISPER_MS} ms"
    fi
  else
    record SKIP "whisper base, English ~2 s" "sample missing" "<= ${T_WHISPER_MS} ms"
  fi

  # Mandarin, from the Piper-generated clip.
  if [[ -s "$WORK/zh.wav" ]]; then
    ZH_IN="$WORK/zh.wav"
    prep_wav "$WORK/zh.wav" "$WORK/zh16.wav" && ZH_IN="$WORK/zh16.wav"
    info "Mandarin clip..."
    ZH_MS=$(for _ in $(seq "$REPS"); do whisper_ms "$ZH_IN" zh; done | median)
    if [[ -n "$ZH_MS" && "$ZH_MS" != "0" ]]; then
      if (( ${ZH_MS%.*} <= T_WHISPER_MS )); then
        record PASS "whisper base, Mandarin ~3 s" "${ZH_MS} ms" "<= ${T_WHISPER_MS} ms"
      else
        record FAIL "whisper base, Mandarin ~3 s" "${ZH_MS} ms" "<= ${T_WHISPER_MS} ms"
      fi
    fi
    # Print the transcript: a rough eyeball on whether Mandarin comes out
    # as sense at all. Not an accuracy metric — the audio is synthetic.
    echo "${DIM}   transcript: $("$WHISPER_BIN" -m "$MODELS/ggml-base.bin" -f "$ZH_IN" \
      -l zh -t "$NPROC" -np -nt 2>/dev/null | tr -d '\n' | sed 's/^ *//')${RST}"
  else
    record SKIP "whisper base, Mandarin ~3 s" "no clip generated" "<= ${T_WHISPER_MS} ms"
  fi
else
  record SKIP "whisper base, English ~2 s" "whisper unavailable" "<= ${T_WHISPER_MS} ms"
  record SKIP "whisper base, Mandarin ~3 s" "whisper unavailable" "<= ${T_WHISPER_MS} ms"
fi

# ── 5. llama.cpp ──────────────────────────────────────────────────────
step "llama.cpp — Qwen2.5 3B Instruct Q4_K_M"

LLAMA_DIR="$VENDOR/llama.cpp"
if [[ ! -d "$LLAMA_DIR" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || warn "clone failed"
fi

LLAMA_BENCH=""
if [[ -d "$LLAMA_DIR" ]]; then
  if [[ ! -x "$LLAMA_DIR/build/bin/llama-bench" ]]; then
    info "compiling (this is the slow one — 10+ minutes)..."
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      >"$WORK/llama_cmake.log" 2>&1 \
      && cmake --build "$LLAMA_DIR/build" -j"$NPROC" --config Release \
        >>"$WORK/llama_cmake.log" 2>&1 \
      || warn "build failed — see $WORK/llama_cmake.log"
  fi
  [[ -x "$LLAMA_DIR/build/bin/llama-bench" ]] && LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
fi

GGUF="$MODELS/qwen2.5-3b-instruct-q4_k_m.gguf"
fetch "$GGUF" \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

if [[ -n "$LLAMA_BENCH" && -s "$GGUF" ]]; then
  info "benchmarking prefill (pp512) and decode (tg128)..."
  REPS=$(( QUICK ? 1 : 3 ))
  "$LLAMA_BENCH" -m "$GGUF" -t "$NPROC" -p 512 -n 128 -r "$REPS" \
    2>"$WORK/llama_bench.err" | tee "$WORK/llama_bench.out"

  # llama-bench prints a markdown table; t/s is the second-to-last column.
  PP=$(awk -F'|' '/pp512/ {gsub(/ /,"",$(NF-1)); split($(NF-1),a,"±"); print a[1]}' "$WORK/llama_bench.out" | tail -1)
  TG=$(awk -F'|' '/tg128/ {gsub(/ /,"",$(NF-1)); split($(NF-1),a,"±"); print a[1]}' "$WORK/llama_bench.out" | tail -1)

  if [[ -n "$PP" ]]; then
    awk -v v="$PP" -v t="$T_LLAMA_PP" 'BEGIN{exit !(v>=t)}' \
      && record PASS "llama prefill (pp512)" "${PP} tok/s" ">= ${T_LLAMA_PP} tok/s" \
      || record FAIL "llama prefill (pp512)" "${PP} tok/s" ">= ${T_LLAMA_PP} tok/s"
  else
    record SKIP "llama prefill (pp512)" "not parsed" ">= ${T_LLAMA_PP} tok/s"
  fi

  if [[ -n "$TG" ]]; then
    awk -v v="$TG" -v t="$T_LLAMA_TG" 'BEGIN{exit !(v>=t)}' \
      && record PASS "llama decode (tg128)" "${TG} tok/s" ">= ${T_LLAMA_TG} tok/s" \
      || record FAIL "llama decode (tg128)" "${TG} tok/s" ">= ${T_LLAMA_TG} tok/s"

    # Turn decode speed into the number that actually drives the design.
    awk -v v="$TG" 'BEGIN{
      printf "\033[2m   → a 30-token tool call + reply takes %.1f s of decode\033[0m\n", 30/v }'
    awk -v v="$TG" 'BEGIN{
      if (30/v > 2.5)
        print "\033[2m   → confirms the fast path must bypass the LLM\033[0m" }'
  else
    record SKIP "llama decode (tg128)" "not parsed" ">= ${T_LLAMA_TG} tok/s"
  fi
else
  record SKIP "llama prefill (pp512)" "llama.cpp or model unavailable" ">= ${T_LLAMA_PP} tok/s"
  record SKIP "llama decode (tg128)"  "llama.cpp or model unavailable" ">= ${T_LLAMA_TG} tok/s"
fi

# ── 6. Kodama-Lite over MPRIS ─────────────────────────────────────────
# Six of the required verbs are supposed to work today with no changes to
# Kodama-Lite, via the MPRIS service in its subsystems/media.rs. This is
# the cheapest possible check of that claim.
step "Kodama-Lite — MPRIS reachability"

# An SSH login gets no session bus in its environment, so playerctl would
# find nothing and this would look like a genuine failure. Kodama-Lite runs
# as a systemd *user* service, so its bus is the standard per-uid socket.
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "/run/user/$(id -u)/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"
  info "no session bus in env (SSH?) — using /run/user/$(id -u)/bus"
fi

if command -v playerctl >/dev/null 2>&1; then
  PLAYERS=$(playerctl -l 2>/dev/null)
  if grep -qi kodama <<<"$PLAYERS"; then
    STATUS=$(playerctl -p kodamalite status 2>/dev/null || echo "?")
    record PASS "MPRIS: kodamalite present" "status=$STATUS" "service on session bus"
    info "transport control is available with no changes to Kodama-Lite"

    # Round-trip cost of one command. Read-only on purpose: this runs
    # unattended, and toggling playback to time it would start playing music
    # at whatever hour the benchmark happens to run.
    S=$(date +%s%N)
    for _ in $(seq 10); do playerctl -p kodamalite status >/dev/null 2>&1; done
    E=$(date +%s%N)
    D_MS=$(( (E - S) / 10000000 ))
    if (( D_MS <= T_DISPATCH_MS )); then
      record PASS "MPRIS dispatch round trip" "${D_MS} ms" "<= ${T_DISPATCH_MS} ms"
    else
      record FAIL "MPRIS dispatch round trip" "${D_MS} ms" "<= ${T_DISPATCH_MS} ms"
    fi
  elif [[ -n "$PLAYERS" ]]; then
    record FAIL "MPRIS: kodamalite present" "found: $(tr '\n' ' ' <<<"$PLAYERS")" "service on session bus"
    warn "other players are visible but Kodama-Lite is not — is it running?"
  else
    record SKIP "MPRIS: kodamalite present" "no players; is Kodama-Lite running?" "service on session bus"
    info "start it, then re-run: systemctl --user start kodama-lite"
  fi
else
  record SKIP "MPRIS: kodamalite present" "playerctl not installed" "service on session bus"
  info "install with: sudo apt install -y playerctl"
fi

# ── 7. Summary ────────────────────────────────────────────────────────
step "M0 results"

printf '\n   %-34s %-22s %s\n' "CHECK" "MEASURED" "REQUIRED"
printf '   %s\n' "$(printf '─%.0s' {1..78})"
FAILED=0; SKIPPED=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r st label measured threshold <<<"$r"
  case "$st" in
    PASS) c="$GRN" ;;
    FAIL) c="$RED"; ((FAILED++)) ;;
    *)    c="$YEL"; ((SKIPPED++)) ;;
  esac
  printf '   %s%-6s%s %-27s %-22s %s\n' "$c" "$st" "$RST" "$label" "$measured" "$threshold"
done
echo

# ── Fast-path rollup ──────────────────────────────────────────────────
# The table above checks components. This is the number the project is
# actually judged on: wake word to first audio out, for a known command that
# never touches the LLM. Assembled from what was just measured, plus two
# design constants that are not measurable here (wake-word detect and the
# VAD silence window before endpointing).
WAKE_MS=150
VAD_MS=400
MATCH_MS=50
: "${EN_MS:=}" "${ZH_MS:=}" "${PIPER_EN_MS:=}" "${PIPER_ZH_MS:=}" "${D_MS:=}"

if [[ -n "$EN_MS" && -n "$PIPER_EN_MS" && -n "$D_MS" ]]; then
  printf '\n   %sFast-path budget (wake word → first audio out)%s\n' "$BOLD" "$RST"
  printf '   %s\n' "$(printf '─%.0s' {1..78})"
  printf '   %-38s %10s %10s\n' "" "ENGLISH" "MANDARIN"
  printf '   %-38s %9s %10s\n' "wake word detect (assumed)"        "$WAKE_MS" "$WAKE_MS"
  printf '   %-38s %9s %10s\n' "VAD endpoint window (design)"      "$VAD_MS"  "$VAD_MS"
  printf '   %-38s %9.0f %10.0f\n' "whisper base -ac $WHISPER_AC (measured)" "${EN_MS:-0}" "${ZH_MS:-0}"
  printf '   %-38s %9s %10s\n' "intent match (design)"             "$MATCH_MS" "$MATCH_MS"
  printf '   %-38s %9s %10s\n' "dispatch over MPRIS (measured)"    "$D_MS"    "$D_MS"
  printf '   %-38s %9s %10s\n' "Piper first audio, warm (measured)" "${PIPER_EN_MS:-?}" "${PIPER_ZH_MS:-?}"
  awk -v w=$WAKE_MS -v v=$VAD_MS -v m=$MATCH_MS -v d="$D_MS" \
      -v e="${EN_MS:-0}" -v z="${ZH_MS:-0}" -v pe="${PIPER_EN_MS:-0}" -v pz="${PIPER_ZH_MS:-0}" \
      -v g="$GRN" -v r="$RED" -v x="$RST" 'BEGIN{
    te=w+v+m+d+e+pe; tz=w+v+m+d+z+pz;
    printf "   %-38s %9.2f %10.2f\n", "TOTAL (seconds)", te/1000, tz/1000;
    ok = (te<=2500 && tz<=2500);
    printf "   %s%s%s\n", (ok?g:r), (ok ? "→ meets the < 2.5 s target on both languages" \
      : "→ MISSES the < 2.5 s target — re-plan the fast path"), x }'
fi

TEMP_END=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2 || echo "?")
info "temp at end: $TEMP_END (started $TEMP)"
if command -v vcgencmd >/dev/null 2>&1; then
  THR_END=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
  [[ "$THR_END" != "0x0" ]] && warn "throttled during the run ($THR_END) — numbers are pessimistic"
fi

echo
if (( FAILED > 0 )); then
  printf '%s   %d check(s) FAILED.%s Re-plan before starting M1 — the latency budget\n' "$RED" "$FAILED" "$RST"
  printf '   in docs/PLAN.md assumes every row above passes.\n'
  exit 1
elif (( SKIPPED > 0 )); then
  printf '%s   %d check(s) skipped.%s Resolve those, then re-run before starting M1.\n' "$YEL" "$SKIPPED" "$RST"
  exit 0
else
  printf '%s   All checks passed.%s The design in docs/PLAN.md holds on this Pi — start M1.\n' "$GRN" "$RST"
  exit 0
fi
