# AIA — AI Assistant for Raspberry Pi 5

An offline-first voice assistant that acts as the primary voice interface for the Pi.
Wake word → speech recognition → intent routing → app control → spoken reply, all on-device.

**Status: v0.4.4 — the voice loop, app control and the conversation UI work on real speech.
The LLM is not built.**

M0 measured the target Pi and every check passed; the fast path landed at **1.46 s** (English)
and **1.58 s** (Mandarin) against a 2.5 s budget. M1 — wake word, capture, endpointing, STT,
routing, TTS — runs on the device and has been reviewed subsystem by subsystem. Anything the
router does not recognise is repeated back rather than answered: that is M2's job and there is
no `aia/llm/` yet. Details and milestones in `docs/PLAN.md`.

Since **v0.4.0** every turn is also written down: the conversation and the saved recordings are
kept for 24 hours and then deleted, and a page on loopback shows the transcript and what AIA is
actually running — the STT model, the Piper voices that loaded, and the microphone the open
capture stream is reading from. See "Screen, transcript and settings".

The version AIA reports is the tag it was deployed from, stamped in by `git archive`; there is
no version constant in the source to bump.

Requires **Kodama-Lite v0.1.38** or newer for the lyrics commands. Older versions accept the
request and ignore it — the control endpoint returns 202 for actions it has never heard of, so
the assistant will say it worked.

## Languages

Mandarin, Cantonese and English, decided **per utterance**. There is no mode to be in and
nothing to select: say "play some music", then "播放音乐", then switch back, and each is
transcribed in the language it was spoken in. Nothing is translated on the way through —
搜索 Taylor Swift 的歌词 reaches the router exactly as it was said.

Getting this wrong is not a slightly worse transcript. A recogniser handed audio in a language
it was not asked for does not fail, it translates: under the old Whisper backend 下一首 came
back as "Next one.", 关机 as "Guanji.", 播放… as "(Song)" — fluent, confident, and impossible
to route. That is why the language is detected on every utterance and never held across a
conversation.

**Cantonese is understood and answered in Mandarin.** The recogniser tags it as `yue` and
transcribes it as Cantonese; Piper still ships no `yue` voice, so the reply comes back in the
Mandarin voice. The two halves are deliberately separate — `_REPLY_IN` in
`aia/stt/sensevoice.py` decides the voice, `aia/tts/language.py` owns the table — so adding a
Cantonese voice later is a row and a model file.

## Design in one picture

```
Microphone → Wake Word (always on, ~0.6% CPU)
                  ↓
            VAD endpointing
                  ↓
     STT (sherpa-onnx, SenseVoiceSmall INT8)
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
                  ↓
   every turn recorded, kept 24 h ─→ overlay strip (the current turn)
                                  └─→ web UI on :8090 (scrollback + settings)
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

## Speech recognition

**SenseVoiceSmall INT8, via sherpa-onnx, loaded once inside the assistant process.** Fetch the
model on the Pi before installing the service — it is ~230 MB and deliberately not in git,
because deploys go over `git archive | ssh`:

```bash
./scripts/get_sensevoice.sh
.venv/bin/python scripts/stt_test.py --selftest
```

STT is offline. That fetch is the only step that touches a network, it happens once, and
nothing at runtime downloads anything or calls an API. A missing model stops AIA at startup
with a line saying so, rather than falling back to something that would.

Measured against reference text, on 17 prompts read in this room through this microphone:

| | SenseVoice | Whisper base |
|---|---|---|
| Mandarin CER | **0.05** | 0.48 |
| Cantonese CER | **0.06** | 0.66 |
| English CER | **0.10** | 0.19 |
| Mixed zh/en CER | **0.12** | 0.38 |
| exact transcripts | **11/17** | 3/17 |
| mean / p95 latency | **150 / 202 ms** | 1112 / 1952 ms |
| RTF | **0.06** | 0.45 |
| CPU | **200%** of one core | 376% |
| peak RSS | 603 MB | **262 MB** |

Cantonese is the reason for the swap and the number that justifies it: **0.06 against 0.66**.
Whisper does not fail on Cantonese so much as answer confidently in something else —
播放陈奕迅嘅歌 came back as 播放陳玉順的歌. SenseVoice costs 341 MB more resident memory and
is worth it.

English is SenseVoice's weakest language, and the errors are not always harmless: it heard
"search lyrics" as "Se lyrics." See the routing note under Commands for what that cost and how
it is guarded.

whisper.cpp is kept as the fallback backend behind the same interface. Switch with
`stt.backend` in `aia/core/config.py`, or for one run — **both halves are needed**, since the
unit is disabled and enabling it alone just burns 272 MB on a backend nothing is using:

```bash
systemctl --user enable --now aia-whisper.service   # it needs its server
AIA_STT_BACKEND=whisper python -m aia.main
```

Measure them against each other on the Pi, on the same recordings, in one process:

```bash
systemctl --user stop aia                                   # the mic allows one reader
.venv/bin/python scripts/stt_test.py record                 # once, with a person
.venv/bin/python scripts/stt_test.py run --backend both
.venv/bin/python scripts/stt_test.py run --threads 1,2,3,4
```

`record` needs somebody in the room: there is no Cantonese audio in this project and none can
be synthesised. Everything after it is repeatable with nobody present.

## Running it

AIA runs as a systemd **user** service and starts with the desktop
session — a user service rather than a system one because it needs the session bus
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
. .venv/bin/activate
python -m aia.main                     # then say 小艾同学, pause, then speak
```

It now controls music. Anything it doesn't recognise as a command, it repeats back (the LLM
that will handle those arrives in M2). Try all three languages in one session — nothing to change.

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

`save lyric` carries a higher floor than the commands around it because it is the only one of
the three that writes. SenseVoice heard "search lyrics" as "Se lyrics.", which scores 0.889
against `save lyrics` and 0.800 against `search lyrics` — the dropped syllables leave the
shared noun carrying the match. No scoring change separates those; the recogniser lost the
information. So the write refuses anything below 0.90, and a near-miss is declined rather than
guessed. Across 71 real captures every genuine save scores exactly 1.000.

Destructive commands need an explicit yes on the following turn; anything ambiguous cancels.

### What it says back — which is usually nothing

Four commands answer out loud: **what's playing**, **shut down**, **reboot** and **close
kodama**. Everything else acts in silence.

That is deliberate. The result of "next" is a different song playing; the result of "volume
fifty" is a different volume. Saying "下一首。" over the top of a track that has audibly
already changed tells the room nothing it cannot hear, and it costs the tail of every turn —
measured at 1250 ms of playback on a 3663 ms turn, during which the assistant is busy and the
music stays ducked. Speech is kept for replies that carry information no other channel does:
an answer to a question, and the three commands that take the screen away.

Two things speak regardless. A command that **asks first** is answered out loud, because the
question held the floor and going quiet at the most consequential moment is the wrong place to
save a second. And a command that could not run at all — the player is closed — says so,
because nothing changed on screen and silence there is indistinguishable from being ignored.

Every reply still reaches the panel and the journal, spoken or not. `CommandSpec.speaks` is
the whole mechanism; it defaults to False.

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

### Sound output — install `pipewire-alsa` or AIA is mute

```bash
sudo apt install pipewire-alsa
```

AIA plays through PortAudio, which talks to ALSA. The Pi's only sink is HDMI and PipeWire owns
it — and holds the ALSA device open for as long as anything is playing. Without the bridge,
ALSA's `default` is simply busy whenever music is on: PortAudio enumerates **zero** output
devices, `sd.default.device` is `-1`, and every reply is synthesised and dropped. The player
keeps working the whole time, because it goes through PipeWire and never touches ALSA.

This is worth stating loudly because of how it fails. Piper reports success, the log prints
`tts[zh] 542 ms to audio: '正在搜索陈慧琳。'` for a reply nobody heard, and the only visible
sign is one line above it:

```
ERROR aia.tts.piper  audio output unavailable, reply not spoken: Error querying device -1
```

`Speaker.warm()` now plays a short quiet tone at startup and says so either way, so a dead
output is a boot-time error rather than something discovered days later by ear:

```
INFO  aia.tts.piper  audio output ready (158 ms for the probe tone)
ERROR aia.tts.piper  AUDIO OUTPUT IS DEAD — every reply will be synthesised and never heard.
```

A failed probe does not stop the service. An assistant that hears and acts but cannot speak is
degraded; one that refuses to start is useless.

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

**Maximising is the compositor's job, not Chromium's.** `--start-maximized` is
honoured on a fresh profile and ignored once Chromium has saved window bounds,
so the same command opens maximised one day and in a 945x430 box the next. A
labwc window rule is deterministic. In `~/.config/labwc/rc.xml`:

```xml
<windowRules>
  <windowRule identifier="aia-ui"><action name="Maximize"/></windowRule>
</windowRules>
```

`aia-ui` is the `--class` the launcher passes, and the same string
`aia-ui.desktop` uses as `StartupWMClass` so the taskbar groups the window
under the AIA icon.

To put it one tap away when Kodama-Lite is covering the desktop, add it to the
panel's launchers in `~/.config/wf-panel-pi/wf-panel-pi.ini` — the entry is the
`.desktop` basename, and the file must be in `~/.local/share/applications`:

```ini
launchers=x-www-browser pcmanfm x-terminal-emulator aia-ui
```

### Two things about this compositor that cost an evening

**labwc here binds no way to close or un-fullscreen a window.** Stock
`rc.xml` gives you Maximize, UnMaximize, volume and magnify — there is no
Alt+F4 and no fullscreen toggle. So a full-screen app with no titlebar is
unrecoverable on a touch-only display. Worth adding:

```xml
<keyboard>
  <keybind key="A-F11"><action name="ToggleFullscreen"/></keybind>
  <keybind key="A-F4"><action name="Close"/></keybind>
</keyboard>
```

**The panel loses its surface when labwc reloads its config, and does not draw
again until it is restarted.** `pkill wf-panel-pi` is enough — `lwrespawn` in
labwc's autostart brings it straight back. Reload labwc with `kill -HUP` on its
pid; `labwc --reconfigure` exits with "LABWC_PID not set" from a plain ssh
session and silently changes nothing, which makes an edit look like it had no
effect when it was simply never loaded.

That pair is worth knowing together, because a full-screen window covering the
panel and a panel that has not been restarted look identical from a screenshot
— and Kodama-Lite in full-screen (its own toolbar toggle, top centre) covers
the taskbar exactly the way a broken panel does.

### The taskbar must stay at the top

`position=bottom` in `~/.config/wf-panel-pi/wf-panel-pi.ini` does not work on
this Pi — wf-panel-pi 1.13 with labwc 0.9.8. The panel process starts and stays
running, but nothing is drawn and **no exclusive zone is reserved**, so a
maximised window takes the full height.

That second half is what makes it a real finding rather than a misreading,
because a panel covered by a full-screen window looks exactly the same in a
screenshot. Under `top` the maximised player sat at y=28 and the panel drew;
under `bottom` the player took all 440 px. Tested four ways, each with the panel
restarted afterwards: the config the preferences dialog writes, the same without
`monitor=`, the full `/etc/xdg` default, and with `autohide=false`. `top` maps
instantly every time; `bottom` never does.

Before blaming the panel, though, check the other cause first: **Kodama-Lite in
full-screen covers the taskbar**, and its toggle is the icon at top centre of
its own toolbar. `Alt+F11` (once bound, see above) gets out of it.

## Repository layout

```
aia/
  core/      state machine, config, system information
  audio/     wake word, VAD, capture
  stt/       backends: SenseVoice (default), whisper.cpp (fallback)
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
