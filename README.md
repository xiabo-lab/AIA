# AIA — AI Assistant for Raspberry Pi 5

An offline-first voice assistant that acts as the primary voice interface for the Pi.
Wake word → speech recognition → intent routing → app control → spoken reply, all on-device.

**Status: M0 passed, M1 built and validated except for the live microphone test.**

M0 measured the target Pi and every check passed; the fast path lands at **1.46 s** (English)
and **1.58 s** (Mandarin) against a 2.5 s budget. M1 — the voice loop — is written and each
stage is verified on the device. The one thing that still needs a human is speaking into the
microphone; see "Try it" below. Details in `docs/PLAN.md`.

## Languages

English and Mandarin. Cantonese is deferred — Piper has no Cantonese voice and stock Whisper
is unusable on Cantonese (~49.5% CER). The language layer is kept behind
`aia/tts/language.py` so adding it later is a branch, not a refactor.

## Design in one picture

```
Microphone → Wake Word (always on, ~0.6% CPU)
                  ↓
            VAD endpointing
                  ↓
          STT (whisper.cpp, ggml-base)
                  ↓
         ┌─── Intent Router ───┐
         │                     │
   fast path (~50 ms)     slow path
   phrase match against   Qwen2.5 3B via
   plugin manifests       llama.cpp server
         │                     │
         └────── App Controller ──────┘
                  ↓
        Plugin (Kodama-Lite) → MPRIS + control API
                  ↓
              Piper TTS (streaming) → Speaker
```

The router tier is the important part. Qwen2.5 3B decodes at ~5–8 tok/s on a Pi 5, so a tool
call plus a spoken reply is 4–6 seconds of generation alone. Routing every command through
the LLM cannot meet the 2.5 s target. Known commands ("pause", "next", "play X") are matched
deterministically in ~50 ms and never touch the LLM; only open-ended conversation does.

## Running the M0 benchmark

M0 measures this specific Pi rather than trusting published benchmarks. Run it on the Pi:

```bash
git clone <this repo> ~/AI_Assit && cd ~/AI_Assit
./scripts/bench_m0.sh
```

It fetches what it needs (whisper.cpp, a Qwen2.5 3B GGUF, Piper and a voice) into `models/`
and `vendor/`, runs each stage, and prints a PASS/FAIL table against the plan's assumptions.
Budget ~20 minutes and ~3 GB of disk on the first run; later runs reuse the downloads.

Flags:

```
--skip-download   use whatever is already in models/ and vendor/
--quick           fewer repetitions, rougher numbers
```

## Running it

AIA runs as a pair of systemd **user** services and starts with the desktop
session — user services rather than system ones because it needs the session bus
(to drive the player over MPRIS) and the Wayland display (for the overlay):

```bash
./scripts/install-service.sh           # install, enable, start
journalctl --user -u aia -f            # watch a conversation happen
systemctl --user restart aia           # after changing the code
systemctl --user stop aia              # free the mic for scripts/wake_test.py
```

To run it by hand instead — say, to try an environment variable:

```bash
cd ~/AI_Assit
systemctl --user stop aia              # the microphone allows one reader
./scripts/run_services.sh start        # resident whisper-server on :8081
. .venv/bin/activate
python -m aia.main                     # then say 小艾同学, pause, then speak
```

It now controls music. Anything it doesn't recognise as a command, it repeats back (the LLM
that will handle those arrives in M2). Try both languages in one session — nothing to change.

| say (English) | say (Mandarin) | does |
|---|---|---|
| play / play some music | 播放歌曲 / 放歌 | resume playback |
| pause / pause the music | 暂停 / 暂停音乐 | pause |
| next / skip | 下一首 / 切歌 | next track |
| previous / go back | 上一首 | previous track |
| stop | 停止播放 | stop |
| what's playing | 这是什么歌 | says the current track |

Everything below goes through the control endpoint added to Kodama-Lite in M5, which it
publishes at `~/.local/state/kodama-lite/control.json` (mode 0600):

| say (English) | say (Mandarin) | does |
|---|---|---|
| play hotel california | 播放五月天 | searches, then plays the results as a queue |
| search for X | 搜索五月天 | shows results without playing |
| set volume to fifty | 音量调到五十 | volume — understands 五十, 百分之五十, 50 |
| shuffle / repeat | 随机播放 / 单曲循环 | shuffle, repeat mode |
| like this song | 点赞 | likes the current track |
| show lyrics | 显示歌词 | lyrics |
| karaoke | 卡拉OK模式 | full-screen karaoke |
| close kodama | 退出软件 | quits the app — **asks first** |
| shut down / reboot | 关机 / 重启 | powers off or restarts — **asks first**

Destructive commands need an explicit yes on the following turn; anything ambiguous cancels.

### The wake word

The phrase is **小艾同学**. No engine ships a pretrained Chinese wake word, so this runs a
small Vosk recogniser and matches the phrase in its output. The recogniser hears it as
小爱同学 — a stable homophone, which is all the matching needs. Both spellings are accepted.

Two consequences worth knowing:

* 小爱同学 is also **Xiaomi's** wake word, so a Xiaomi device in the same room will answer
  to this too.
* A general recogniser will false-trigger on background Mandarin (a TV, a podcast) more
  than a purpose-built engine, and costs ~49% of one core while anyone is speaking —
  against 6.1% at idle, which is the state it is in almost all the time.

Porcupine fixes both (0.6% flat, purpose-built) and the backend is already written. It
needs a free access key and a custom keyword file:

```bash
./scripts/get_porcupine_zh.sh          # fetches the Mandarin parameter file, explains the rest
export PICOVOICE_ACCESS_KEY=...        # then set wake.backend = "porcupine"
```

To exercise the pipeline without any wake word, `AIA_NO_WAKE=1 python -m aia.main`.

To check latency without speaking at all — safe to run unattended, synthesises but plays
nothing:

```bash
python scripts/replay.py .bench/en2s.wav .bench/zh16.wav
```

## Repository layout

```
aia/
  core/      state machine, event bus, config
  audio/     wake word, VAD, capture
  stt/       whisper.cpp wrapper
  router/    fast phrase matcher + LLM client
  tts/       Piper synthesis, language resolution
  plugins/   plugin ABC + per-app handlers
docs/PLAN.md the full plan, milestones and open risks
scripts/     bench_m0.sh and setup helpers
models/      ggml / gguf / onnx (gitignored)
vendor/      built third-party binaries (gitignored)
```

## Hardware

Raspberry Pi 5 (8 GB), Pi OS 64-bit, 1920×440 capacitive touch display, USB microphone,
stereo speakers. An active cooler and NVMe boot both matter — the benchmark numbers this
project is designed around assume them, and M0 reports if either is missing.
