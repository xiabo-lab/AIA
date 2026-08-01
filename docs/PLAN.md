# AIA — Feasibility Research & Implementation Plan

## Context

You asked whether this project is buildable. **Verdict: yes.** With Cantonese deferred to a
later stage (your call — English + Mandarin only for now), the remaining risk is almost
entirely latency, and that is solvable with a routing change.

The stack (whisper.cpp + Qwen2.5 3B + Piper + Pi 5) is sound and the two target apps already
exist locally. I verified each spec requirement against measurements rather than assuming:

**M0 ran on the target Pi (raspberrypi5, 10.0.0.113) on 2026-08-01 and every check passed.**
The table below now reports measured values, not published ones.

| Spec requirement | Measured on this Pi | Status |
|---|---|---|
| Total response < 2.5 s via LLM | Qwen2.5 3B Q4 decode **5.67 tok/s** → a 30-token tool call + reply is **5.3 s** of decode alone | **Not possible as written** — needs tiered routing |
| Fast path < 2.5 s (tiered) | **1.46 s** English, **1.58 s** Mandarin | Met |
| TTS for English + Mandarin | Piper warm synth **304 ms** / **224 ms** | Met |
| STT for English + Mandarin | whisper `base -ac 512` **542 ms** / **747 ms** | Met |
| LLM prefill | **28.4 tok/s** (published ~33) | Met, KV-cache reuse still mandatory |
| Control Kodama-Lite | MPRIS round trip **10 ms**; play/pause verified live | Met, no app changes |
| Wake word, low CPU | Porcupine = 0.6% CPU (published; **not yet measured**) | Open |
| Fits in 8 GB without swap | 3B Q4 = 1.95 GiB + whisper 147 MB + Piper ~100 MB | Met |
| Boot < 30 s | Pi boots from **SD card**, not NVMe | At risk — models must lazy-load behind the wake word |

Thermals stayed sane: 55 °C idle → 70 °C under sustained load, `throttled=0x0` throughout.

**Deferring Cantonese removed the two hardest problems**, and it is worth recording why so the
decision can be revisited deliberately:

- Piper has **zero** Cantonese voices — I checked the manifest directly: 35 languages, `zh_CN`
  only, no `yue`. There was no off-the-shelf path to Cantonese speech output at all.
- Stock Whisper on Cantonese is unusable (**~49.5% CER** on tiny). It would have required a
  fine-tuned model converted to ggml, plus forcing language `zh` rather than `yue`.

Neither problem exists for English or Mandarin.

The good news from reading your code: **Kodama-Lite already accepts inbound transport
commands.** `src-tauri/src/subsystems/media.rs` publishes MPRIS on the session D-Bus and
forwards Play/Pause/Toggle/Next/Previous/Stop/Seek back over the bus as a `media:control`
event. Six of the required verbs work today with zero changes to that app.

### Decisions taken

1. **Languages** — English + Mandarin only this stage. Cantonese deferred.
2. **Latency** — tiered routing; known commands bypass the LLM entirely.
3. **App control** — Kodama-Lite gets a control API now; Pi Dashboard deferred.
4. **Display** — AIA is headless; each app renders its own overlay strip.

---

## Revised architecture

The spec's diagram routes every utterance through the LLM. That is the single cause of the
latency miss. The corrected flow puts a deterministic router in front:

```
Microphone → Wake Word (Porcupine, always on)
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

**Latency budget, fast path — MEASURED on the device (M0, 2026-08-01):**

| Stage | English | Mandarin | Source |
|---|---|---|---|
| Wake word detect | 150 ms | 150 ms | assumed (Porcupine, not yet measured) |
| VAD endpoint window | 400 ms | 400 ms | design constant |
| whisper.cpp `base` `-ac 512` | **542 ms** | **747 ms** | measured, resident model |
| Intent match | 50 ms | 50 ms | design constant |
| Dispatch over MPRIS | **10 ms** | **10 ms** | measured |
| Piper first audio, warm | **304 ms** | **224 ms** | measured, resident model |
| **Total** | **1.46 s** ✅ | **1.58 s** ✅ | |

Both comfortably inside 2.5 s. Two configuration choices are load-bearing and were only
found by measuring:

**1. `--audio-ctx 512` on whisper — a 4.1× encoder speedup, for free.** Whisper pads every
clip to 30 s and runs its encoder over the whole window regardless of how short the
utterance is, so a 2 s command cost 1564 ms of encode at the default 1500. Capping the audio
context to ~10 s drops that to 378 ms, and the transcripts are unchanged — on the Mandarin
test clip the capped run was actually *better* (it got 周 right; the uncapped one did not).
Do not lower it further: at 256 the decoder starts falling back and looping, and total time
goes **up** (3072 ms) despite the faster encode.

**2. Both models must stay resident.** This is an architectural requirement, not an
optimisation. Piper is dominated by ONNX load — 452 ms (en) / 1190 ms (zh) to load against
304 ms / 224 ms to synthesise. Whisper's load is 96 ms. A design that shells out per
utterance pays those every time and misses the budget; the first version of `bench_m0.sh`
did exactly that and reported a spurious 672 ms failure for Piper.

**Do not quantise the models.** `q5_1` is *slower* than `f16` on this hardware — base encode
went 1564 ms → 2212 ms — because ARM NEON handles f16 natively and q5_1 costs a dequantise
step. The smaller file buys nothing on a Pi with 8 GB.

**Slow path** (open conversation) lands at 4–7 s. That is acceptable because the user is
asking a question, not issuing a command — and the "thinking" overlay sets the expectation.
Do not promise < 2.5 s here.

Two techniques are mandatory to hold the fast path:
- **llama.cpp KV cache reuse** (`--keep`, `cache_prompt`). The plugin manifest system prompt
  is ~800 tokens; at 33 tok/s prefill that is 24 s if re-sent each turn. Prefill it once at
  startup and reuse, so per-turn prefill is only the new user text.
- **Streaming TTS.** Start Piper on the first complete sentence, not the full response.

---

## Language handling

Much simpler than the original spec implied, now that it is English + Mandarin:

- **Recognition.** One model: `ggml-base` (multilingual), with `-l auto`. Verified on the
  device: English and Mandarin clips in either order are detected correctly with no state
  carried between them, so the spec's free mid-conversation switching works as written.

  Auto-detect costs ~390 ms per utterance (a whole extra encoder pass), and a sticky-language
  scheme was built to avoid it and then **removed**. It is worth recording why, because the
  idea is tempting enough to reinvent: nothing detects "wrong language" cheaply. Whisper
  emits text in the script you asked for, so a script check never fires — Mandarin forced
  through English came back as fluent English prose, quietly *translated*. Mean word
  probability did separate the bad case (0.17 vs 0.69), but only with temperature fallback
  on, which is also what made that pass take **10.4 s**; disabling fallback to cap the
  latency made Whisper produce a *confident* mistranslation instead (0.77), destroying the
  signal. And reading confidence at all requires `verbose_json`, whose DTW pass costs
  ~390 ms — the entire saving. Auto-detect is correct, bounded, and still leaves ~700 ms of
  headroom. `aia/stt/engine.py` carries the full table.
- **Reply language.** Detect the transcript's script and pick the Piper voice:
  Latin → `en_US-*`, Han → `zh_CN-huayan-medium`.
- **Free mid-conversation switching** is genuinely achievable for this pair, so the spec's
  example ("Play 周杰倫" → "Increase the volume") will work as written.
- **Keep the language layer pluggable.** Voice selection and model choice belong behind a
  single `resolve_language(transcript) -> (stt_model, piper_voice)` function, so adding
  Cantonese later is one branch plus two model files, not a refactor.

---

## Repository layout

New project in `C:\Users\fuwen\Claude\AI_Assit` (currently empty). Python, matching
Pi Dashboard's language and because every dependency has first-class Python bindings.

```
aia/
  core/
    state.py         # idle → listening → thinking → speaking → acting
    bus.py           # in-process async event bus
    config.py
  audio/
    wake.py          # Porcupine; owns the mic, gates everything downstream
    vad.py           # webrtcvad / silero endpointing
    capture.py
  stt/
    engine.py        # pywhispercpp wrapper
  router/
    fast.py          # phrase → intent match, built from plugin manifests
    llm.py           # llama.cpp HTTP client, streaming, tool-call parsing
  tts/
    piper.py         # streaming synthesis, rate + volume control
    language.py      # resolve_language(); the Cantonese seam
  plugins/
    base.py          # Plugin ABC + manifest schema
    kodama.py        # MPRIS transport + control API for the rest
  server.py          # localhost HTTP/WS: overlay feed + plugin registration
  logging.py
models/              # ggml + gguf + onnx, gitignored
systemd/aia.service
```

Single process with asyncio. Do not build separate services with REST between them — the
spec asks for independently replaceable modules, and the plugin ABC plus the internal bus
deliver that without paying IPC latency on the critical path.

---

## Plugin protocol

Each plugin declares a manifest that feeds **both** router tiers — the fast matcher compiles
phrase patterns from it, and the LLM receives it as a tool schema. One source of truth:

```python
{
  "name": "kodama",
  "description": "Music player",
  "commands": [
    {"name": "play", "params": {"query": "str"},
     "phrases": ["play {query}", "播放 {query}"]},
    {"name": "pause", "params": {}, "phrases": ["pause", "stop music", "暫停"]},
    {"name": "volume", "params": {"level": "int:0-100"},
     "phrases": ["volume {level} percent", "音量 {level}"]},
  ]
}
```

Phrases are listed per supported language; the fast matcher tries all of them regardless of
detected language, which costs nothing and tolerates code-switching mid-sentence.

Adding an app means dropping a manifest and a handler into `plugins/` — no core changes,
satisfying success criterion 8.

---

## Kodama-Lite control API

Split by what already works:

**Free today — no app changes. VERIFIED on the device (2026-07-31).** `pause`, `resume`,
`next`, `previous`, `stop`, `seek` via MPRIS. The bus name is
`org.mpris.MediaPlayer2.kodamalite` (`playerctl -p kodamalite`). Introspection confirms
`Next`/`PlayPause`/`Previous`/`Stop` methods with `CanPlay`, `CanPause`, `CanGoNext`,
`CanGoPrevious` and `CanSeek` all `true`, and a `play-pause` round-trip moved the player
Paused → Playing → Paused. Metadata (`xesam:title`, `xesam:artist`, `mpris:length`) is
populated, so AIA can also *report* now-playing without any new API.

Note for any process driving this from a non-login context (SSH, a systemd service without
a session): there is no `DBUS_SESSION_BUS_ADDRESS` in the environment and playerctl will
find nothing. Kodama-Lite runs as a systemd *user* service, so the bus is
`unix:path=/run/user/<uid>/bus`. AIA must set this explicitly.

**Needs adding** — `play(song)`, `search`, `volume`, `shuffle`, `repeat`, `playlists`,
`lyrics`. These live in the frontend (see `src/lib/audioEngine.ts`; volume is deliberately
owned by the UI per the comment in `src-tauri/src/subsystems/volume.rs`), so they cannot be
reached over MPRIS.

The clean path reuses infrastructure Kodama-Lite already has:

1. It already runs an **axum server on 127.0.0.1** for streaming
   (`src-tauri/src/subsystems/playback/server.rs`). Add a `/control` route there — no new
   dependency, no new port strategy.

   **Discovery matters here.** That server deliberately binds port **0** (a random port) and
   namespaces every route under a random **token**: `http://127.0.0.1:<port>/<token>/...`.
   The header comment is explicit that something knowing only the port cannot form a valid
   request. So AIA cannot hardcode an endpoint, and adding a second fixed unauthenticated
   port would throw away a security property the app chose on purpose.

   Instead, have Kodama-Lite write its base URL to a well-known state file on startup —
   `~/.local/state/kodama-lite/control.json`, mode 0600 — and have AIA read it. That keeps
   the random-port + token model intact while making the endpoint discoverable to a local
   process running as the same user, which is exactly the trust boundary we want.
2. Add one `Command` variant and one `AppEvent` variant in **both** `src-tauri/src/protocol.rs`
   and its TypeScript mirror `src/protocol.ts` (the file header documents that they must stay
   in lockstep).
3. The route emits `control:command` over the existing bus; the frontend subscribes in
   `AppShell.tsx` and dispatches into the existing zustand store actions.

Bind to loopback only, and treat it as trusted-local — this is the sandboxing the spec's
security section asks for.

---

## Milestones

- **M0 — Bench on real hardware.** Before writing AIA code, measure on *your* Pi 5: `whisper.cpp`
  `base` on a 2 s Mandarin clip and a 2 s English clip, `llama-bench` on Qwen2.5-3B-Instruct
  Q4_K_M, Piper first-chunk latency, and `playerctl -p kodamalite play-pause` against a running
  Kodama-Lite. Every number above is from published benchmarks, not your device. **If M0
  disagrees, re-plan before M1.**
- **M1 — Voice loop, no apps. BUILT; live mic test outstanding.** Wake word → STT → Piper
  echo, both languages. Every stage verified on the device: capture delivers 0.998× real
  time in uniform 480-sample frames, the wake word costs 5.6% of one core at idle with no
  false fires on room noise, and `scripts/replay.py` puts the whole chain at **1729 ms**
  (English) / **2048 ms** (Mandarin) against the 2.5 s budget. What is untested is a human
  speaking into the microphone — everything above used recorded or synthesised audio.

  Three things were found only by running it, all now fixed and documented in the code:
  a fixed `blocksize` on this USB mic silently **dropped ~19% of all audio** (only
  `blocksize=0` keeps up); the anti-alias filter must carry state across blocks or it
  discontinuities 33×/second; and the sticky-language optimisation had to be removed
  (see below).

  **Wake phrase is 小艾同学**, via a small Vosk Mandarin recogniser matched against its
  output (heard consistently as 小爱同学). No engine ships a pretrained Chinese wake word:
  openWakeWord bundles six English phrases, and Porcupine does Mandarin properly but needs
  a Picovoice key plus a custom keyword file, since only 你好/咖啡/水饺/豪猪 are built in.
  Measured cost: **6.1% of one core at idle, ~49% while anyone is speaking** — the
  asymmetry a purpose-built engine does not have. The Porcupine backend is written and
  one env var away; `scripts/get_porcupine_zh.sh` fetches what is freely downloadable.

  Caveat to keep in view: 小爱同学 is Xiaomi's wake word, so a Xiaomi device in the room
  will answer to it as well.

### Lead worth following: Vosk may beat Whisper on Mandarin

Noticed while testing the wake word, on the same synthesised clips:

    spoken            Whisper base -ac 512     Vosk small-cn
    我想听周杰伦        —                        我想听周杰伦        exact
    播放周杰伦的歌曲    播放周結倫的各取…          播放周杰伦的歌曲    exact
    今天天气怎么样      —                        今天天气怎么样      exact

Whisper mangles 周杰伦 into 周結倫/周结轮 in every configuration tried; the 42 MB Vosk
model gets it right. That is precisely open risk #3 below — Mandarin proper nouns are the
weak spot, and artist and song names are most of what the Kodama-Lite plugin will need to
match. Worth measuring properly against real microphone audio before M4: if it holds up,
routing Mandarin through Vosk and English through Whisper may beat using Whisper for both.
Do not act on it yet — this was synthesised speech, which flatters both engines.
- **M2 — LLM + slow path.** llama.cpp server with KV cache reuse, streaming into Piper.
- **M3 — Plugin framework + fast router. DONE.** Manifest schema (`aia/plugins/base.py`),
  phrase matcher (`aia/router/fast.py`), dispatch wired into the loop. 22/22 on a mixed
  English/Mandarin test set including three adversarial negatives, and **1.91 ms** per
  match — 26× under the 50 ms the budget allowed.

  **Matching happens in pinyin, not characters**, and this turned out to matter more than
  expected. Whisper's Mandarin errors are overwhelmingly homophone errors, so they vanish
  when compared by sound:

        heard          intended       characters   pinyin
        不放歌曲        播放歌曲          0.75        0.90
        播放周结轮      播放周杰伦         0.60        1.00
        今天天气怎么样   播放歌曲          0.00        0.19

  不放歌曲 is the real transcript Whisper produced when the user said 播放歌曲; it now
  routes correctly. 周结轮 and 周杰伦 are *identical* in pinyin. Comparing characters would
  need a threshold of 0.60 — low enough to accept nonsense.

- **M4 — Kodama-Lite via MPRIS. DONE.** Seven commands (pause, resume, toggle, next,
  previous, stop, now-playing) in `aia/plugins/kodama.py`, driven with `playerctl` and
  **no changes to Kodama-Lite**. Verified end to end against the running player: state
  changes stick, `now_playing` reads real metadata, dispatch is 12-22 ms.
- **M5 — Kodama-Lite control API. DONE.** `POST /<token>/control` added to the existing axum
  server, a `control:command` event mirrored in both protocol files, and a cross-store handler
  in `src/lib/voiceControl.ts`. Every branch calls the same store action the on-screen control
  calls — the rule `media:control` already follows. Verified on the device by screenshot:
  karaoke opened with synced lyrics, shuffle lit, volume moved, and a spoken artist name
  loaded a queue.

  Commands: play (search + queue), search, volume, shuffle, repeat, like, lyrics, karaoke,
  quit. Destructive ones (quit, and the `system` plugin's shutdown/reboot) require an explicit
  yes on the next turn; an ambiguous answer cancels.

  **Discovery keeps the app's security model.** That server binds port 0 under a random
  per-launch token precisely so knowing the port is not enough to drive it, so the URL is
  written to `~/.local/state/kodama-lite/control.json` at mode 0600 rather than being made
  guessable. A fixed unauthenticated port would have been simpler and would have discarded
  that property.

  Two traps worth recording. **`cargo build --release` alone produces a broken app**: Tauri
  only embeds `dist/` under the `custom-protocol` feature, so without it the webview loads
  `devUrl` and shows "Could not connect to localhost". Everything else looked healthy —
  HTTP 202s, Rust logging every command — and only a screenshot of the display revealed it.
  And a voice `play` must use `playQueue`, not `playNow`: the latter makes a one-track queue
  where "下一首" silently does nothing while MPRIS still advertises `CanGoNext: true`.

  Known rough edge: the queue is built from flat search results, so after the first few
  tracks an artist query drifts into covers and related uploads. Using the artist page or a
  radio seed would hold the thread better.
- **M6 — Overlay + polish.** Listening/thinking/speaking strip inside Kodama-Lite, logging,
  error phrases, systemd unit.

Deferred: Cantonese (input and output), Pi Dashboard integration, and everything in the
spec's "Future Enhancements".

---

## Verification

- **Latency**: log a monotonic timestamp at each stage transition in `core/state.py`; assert
  wake→audio-out p50 < 2.5 s across 20 fast-path commands. This is the primary success gate.
- **STT**: hold out ~50 command clips (25 English, 25 Mandarin); require WER < 10% English and
  CER < 15% Mandarin before M4.
- **Language switching**: alternate English and Mandarin commands in one session; confirm the
  correct Piper voice is chosen every turn with no settings change.
- **Wake word**: run 8 h with a radio playing; count false activations (target < 1/hour).
- **Kodama control**: script all 15 spec verbs; confirm each changes player state.
- **Memory**: `systemd-cgtop` during a 30 min session with music playing; confirm no swap.
- **Offline**: unplug the 5G adapter — voice loop, local playback and device control must all
  still work.

---

## Open risks

1. **Every performance number is from published benchmarks, not your Pi.** M0 exists to
   de-risk this; treat M0 results as authoritative over this document.
2. **Mic quality dominates STT accuracy** more than model choice. A cheap USB mic at 2 m will
   undo a better model. Budget for a decent one — this is now the top accuracy risk.
3. ~~**Mandarin proper nouns** are the weak spot for `base`.~~ **Largely resolved by M3.**
   Confirmed real (播放 heard as 不放, 周杰伦 as 周结轮/周結倫) but the failure is in the
   *writing system*, not the recognition — the sounds are right. Matching in pinyin makes it
   disappear for command routing. It will return in M5, where a song title has to be matched
   against a real library rather than a fixed phrase list; the same pinyin normalisation
   should be applied to that search.

4. **STT is slower on real speech than on the synthesised clips M0 used.** A live turn
   measured **2115 ms** against 877-1036 ms in the benchmark. Real room audio is harder, and
   a difficult decode triggers Whisper's temperature fallback, which re-decodes and roughly
   doubles the cost — the same root cause that produced 不放. Utterances are now saved under
   `.bench/utterances` (`AIA_SAVE_AUDIO=1`) so this can be measured against real audio
   instead of TTS. This is the main open threat to the latency budget.
4. **Pi Dashboard currently targets a Pi Zero 2W under `cage`** (see the header comment in
   `Pi_dashboard/display.py`), not the Pi 5. When it comes back into scope, confirm which
   device it is meant to run on — the spec assumes the Pi 5.

---

## Sources

- [Running LLMs on Raspberry Pi 5: Real Benchmarks — TinyWeights](https://tinyweights.dev/posts/run-llms-raspberry-pi-5/) — 3B Q4_K_M: 33 tok/s prefill, 8.8 tok/s decode
- [How Well Do LLMs Perform on a Raspberry Pi 5? — Stratosphere Lab](https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5) — ~5 tok/s for 3B models
- [whisper.cpp real-time transcription discussion](https://github.com/ggml-org/whisper.cpp/discussions/166) — base ≈ real-time, small ≈ 0.4–0.6× on Pi-class hardware
- [Piper voices manifest (`rhasspy/piper-voices`)](https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json) — checked directly: `en_US` + 4 `zh_CN` voices confirmed; no `yue`
- [Porcupine Wake Word](https://picovoice.ai/products/voice/wake-word/) — 0.6% CPU on Pi 5
- [openWakeWord](https://github.com/dscripka/openWakeWord) — free alternative; pretrained models are CC-BY-NC-SA

Kept for whenever Cantonese comes back into scope:
[alvanlii/whisper-small-cantonese](https://huggingface.co/alvanlii/whisper-small-cantonese) (7.93 CER) ·
[force `zh`, not `yue`](https://augustinchan.dev/posts/2026-01-13-whisper-cantonese-transcription-zh-not-yue) ·
[LoRA-INT8 Whisper for Cantonese](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431075/) (49.5% → 11.1% CER)
