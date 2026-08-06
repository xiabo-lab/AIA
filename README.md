# AIA — AI Assistant for Raspberry Pi 5

An offline-first voice assistant that acts as the primary voice interface for the Pi.
Wake word → speech recognition → intent routing → app control → spoken reply, all on-device.

**Status: v0.3.0 — the voice loop and app control work on real speech. The LLM is not built.**

M0 measured the target Pi and every check passed; the fast path landed at **1.46 s** (English)
and **1.58 s** (Mandarin) against a 2.5 s budget. M1 — wake word, capture, endpointing, STT,
routing, TTS — runs on the device and has been reviewed subsystem by subsystem. Anything the
router does not recognise is repeated back rather than answered: that is M2's job and there is
no `aia/llm/` yet. Details and milestones in `docs/PLAN.md`.

Requires **Kodama-Lite v0.1.38** or newer for the lyrics commands. Older versions accept the
request and ignore it — the control endpoint returns 202 for actions it has never heard of, so
the assistant will say it worked.

## Languages

English and Mandarin, decided **per utterance**. There is no mode to be in and nothing to
select: say "play some music", then "播放音乐", then switch back, and each is transcribed in
the language it was spoken in.

That costs about 600 ms against naming a language outright, and it is worth it because the
failure it removes is not a slightly worse transcript. Whisper does not fail when handed audio
in a language it was not asked for — it translates. 下一首 came back as "Next one.", 关机 as
"Guanji.", 播放… as "(Song)": fluent, confident, and impossible to route.

Cantonese is deferred — Piper has no Cantonese voice and stock Whisper is unusable on
Cantonese (~49.5% CER). The language layer is kept behind `aia/tts/language.py` so adding it
later is a branch, not a refactor.

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
   fast path (~9 ms)      slow path — NOT BUILT
   phrase match against   Qwen2.5 3B via
   plugin manifests       llama.cpp server
         │                     │
         └────── App Controller ──────┘
                  ↓
        Plugin (Kodama-Lite) → MPRIS + control API
                  ↓
              Piper TTS (streaming) → Speaker
```

The router tier is the important part. Qwen2.5 3B benchmarks at 5.67 tok/s on an idle Pi 5 —
and **2.4 tok/s measured with whisper-server and the player actually running**, because both
engines ask for all four cores. A 30-token reply is therefore closer to 12 s than to the 5 s
the benchmark suggests. Routing every command through the LLM cannot meet the 2.5 s target
and never could. Known commands ("pause", "next", "play X") are matched deterministically in
**~9 ms on the Pi** and never touch a model; only open-ended conversation will.

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

It also serves a page at **http://127.0.0.1:8090** — the last 24 hours of
conversation, and what it is running. See "Screen, transcript and settings".

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
| search song hotel california | 搜索歌曲七里香 | Song Search window, by name |
| set volume to fifty | 音量调到五十 | volume — understands 五十, 百分之五十, 50 |
| shuffle / repeat | 随机播放 / 单曲循环 | shuffle, repeat mode |
| like this song | 点赞 | likes the current track |
| show lyrics | 显示歌词 | shows the lyrics already found |
| search lyric | 搜索歌词 | looks them up again — the karaoke magnifier |
| save lyric | 保存歌词 | commits them to the cache — the green tick |
| karaoke | 卡拉OK模式 | full-screen karaoke |
| close kodama | 退出软件 | quits the app — **asks first** |
| shut down / reboot | 关机 / 重启 | powers off or restarts — **asks first**

`search lyric` and `search song` are deliberately different actions and never cross: one stays
on the karaoke screen, the other opens Song Search. They are one syllable apart in Mandarin —
搜索歌词 and 搜索歌曲 score 0.80 against each other in the pinyin the router compares, above
the 0.78 it needs to fire — so the separation does not rest on the threshold. It rests on a
raised score floor for the two no-argument lyric commands, a rule that an argument which is
itself a command is not an argument, and a bare trigger declining instead of guessing.

Destructive commands need an explicit yes on the following turn; anything ambiguous cancels.

### Microphone level — check this before anything else

A microphone running near full scale cannot be endpointed. webrtcvad calls every frame speech,
so `silence_ms` is never satisfied and every utterance runs to `max_utterance_ms`, turning a
two-second command into a ten-second turn. Measured: a capture that is 33% voiced at its own
level is **100% voiced** once amplified to peak 0.0 dBFS. It takes barely any clipping — real
captures failed this way at −1.8 and −0.6 dBFS with *no* clipped samples at all.

So capture gain is not a matter of taste:

```bash
amixer -c <card> sset Mic 8                    # not 16 — leave headroom
amixer -c <card> sset 'Auto Gain Control' off  # AGC winds gain into the rail
sudo alsactl store                             # or a reboot reverts both
```

The symptoms in `journalctl --user -u aia` are `samples clipped at full scale` and
`too hot to endpoint`. Two USB microphones also enumerate under the *same* name, differing
only by a card number that moves on re-plug, so AIA warns when more than one input matches.

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

## Screen, transcript and settings

There are two displays, and they answer different questions.

**The overlay** is a Wayland layer-shell strip over whatever is running. It
shows the current turn — "Listening…", what was heard, what was answered — and
fades after five seconds. It never takes focus, which is what stops it stealing
the keyboard from the full-screen player, and which is also why it cannot be
scrolled.

**The web UI**, at `http://127.0.0.1:8090`, is for everything the strip cannot
hold: scrollback through the last 24 hours, and a settings page reporting the
AIA version, the STT model, the Piper voices that actually loaded, the LLM
(there isn't one yet — M2), and **the microphone AIA is really using**, read
back from the open capture stream and the ALSA mixer rather than from config.
Gain and AGC come from `amixer` itself, so what is on screen is what the
hardware is set to.

It is read-only and bound to **loopback**, deliberately: it serves a transcript
of everything said in the room and has no authentication. Open it in Chromium
on the Pi, or forward it over ssh (`ssh -L 8090:127.0.0.1:8090 raspberrypi5`).
`AIA_NO_WEB=1` turns it off.

### As a desktop app

`scripts/aia-ui.sh` opens it as a maximised Chromium app window — the same
shape Kodama-Lite runs in, filling the screen below the taskbar. Install the
launcher on the Pi:

```bash
cp ~/AI_Assit/scripts/aia-ui.desktop ~/Desktop/
cp ~/AI_Assit/scripts/aia-ui.desktop ~/.local/share/applications/
```

**Maximised and not full-screen, on purpose.** Full-screen covers the taskbar,
and on this machine that is a one-way door: labwc's `rc.xml` binds no Close
action at all — there is no Alt+F4 — and `window.close()` does not close a
Chromium app window either. On a touch-only display that leaves a window with
no panel, no titlebar and no way out. Maximised costs 28 px of 440 and keeps
both the taskbar and a titlebar close button under your finger.

The script waits for the server before opening, so pressing it at boot lands on
the UI rather than on Chromium's "site can't be reached" — which an `--app`
window has no address bar to get off. Its other flags are commented in place.

### The taskbar must stay at the top

`position=bottom` in `~/.config/wf-panel-pi/wf-panel-pi.ini` does not work on
this Pi — wf-panel-pi 1.13 with labwc 0.9.8. The panel process starts and stays
running, but nothing is ever drawn and no exclusive zone is reserved, so a
maximised window takes the full height and the taskbar is simply gone. Verified
by screenshot in both positions, and it is not the `monitor=` key, not the
partial config the panel's own preferences dialog writes, and not autohide:
`top` maps instantly, `bottom` never does.

### 24 hours, and the one exception

Both the conversation database and the saved recordings expire after 24 hours.
A sweep runs at startup and every 15 minutes.

The exception is that the **newest 100 recordings survive regardless of age**.
Audio cannot be recaptured — it is a particular speaker, room and microphone on
a particular day — and the last hundred captures are what a misrecognition is
diagnosed from. Conversation text has no such exemption and expires outright.

Recording cleanup only ever looks at `*.wav` directly inside
`.bench/utterances`, never recursively. That is not fussiness: `.bench/` also
holds `wake-trials-pre-phasefix/`, the only surviving audio captured through the
decimator bug, and a wider sweep would take it.

```bash
python -m unittest discover -s tests -t .   # the expiry rules, on any machine
```

Those tests run on the development machine — no microphone, no ALSA, no numpy.
Expiry is the one part of this project that is pure enough to check off-device.

## Repository layout

```
aia/
  core/      state machine, config, system information
  audio/     wake word, VAD, capture
  stt/       whisper.cpp wrapper
  router/    fast phrase matcher (the LLM client lands with M2)
  tts/       Piper synthesis, language resolution
  plugins/   plugin ABC + per-app handlers
  ui/        overlay strip, conversation history, retention, web UI
docs/PLAN.md the full plan, milestones and open risks
scripts/     bench_m0.sh and setup helpers
tests/       retention and history — the off-device testable part
models/      ggml / gguf / onnx (gitignored)
vendor/      built third-party binaries (gitignored)
```

## Hardware

Raspberry Pi 5 (8 GB), Pi OS 64-bit, 1920×440 capacitive touch display, USB microphone,
stereo speakers. An active cooler and NVMe boot both matter — the benchmark numbers this
project is designed around assume them, and M0 reports if either is missing.
