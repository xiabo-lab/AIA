"""Configuration for AIA.

Values here are not arbitrary defaults — most were forced by measurements on the
target device (see docs/PLAN.md, "M0"). Where a number is load-bearing the
comment says what breaks if you change it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"


@dataclass(frozen=True)
class AudioConfig:
    # The USB PnP sound device (TI PCM2902) refuses 16 kHz outright —
    # portaudio returns paInvalidSampleRate. It offers only 44100 and 48000,
    # so we capture at 48 k and decimate by exactly 3. Do not "simplify" this
    # to a 16 kHz open; it fails on this hardware.
    capture_rate: int = 48000
    target_rate: int = 16000          # what Whisper and openWakeWord both want
    channels: int = 1
    dtype: str = "int16"

    # Substring matched against the portaudio device name. The USB mic
    # enumerates as "USB PnP Sound Device: Audio (hw:2,0)", but the card
    # number moves when other USB audio is plugged in, so match on name.
    device_match: str = "USB"

    # 30 ms at 16 kHz = 480 samples. webrtcvad accepts only 10/20/30 ms
    # frames, and 30 ms is the cheapest of the three.
    frame_ms: int = 30

    @property
    def frame_samples(self) -> int:
        return self.target_rate * self.frame_ms // 1000

    @property
    def capture_block(self) -> int:
        # Capture in whole output frames so decimation never straddles a block.
        return self.capture_rate * self.frame_ms // 1000


@dataclass(frozen=True)
class WakeConfig:
    # No engine ships a pretrained Chinese wake word, so this runs a small
    # Vosk recogniser and matches the phrase in its output. That is not what a
    # purpose-built wake-word engine does, and the trade is explicit: 6.1% of
    # one core at idle (on par with openWakeWord) but ~49% whenever anyone is
    # speaking, and a recogniser listening to everything will false-trigger on
    # background Mandarin — a TV, a podcast, another conversation — far more
    # than a dedicated model. Switch to "porcupine" for a real one.
    backend: str = "vosk"          # vosk | porcupine | openwakeword
    threshold: float = 0.5         # openwakeword only

    # What the user actually says.
    phrase: str = "小艾同学"

    # What the recogniser actually *hears* when they say it. Exact characters
    # do not matter — consistency does. Measured through the Mandarin voice,
    # 小艾同学 comes back as 小爱同学 every time, a stable homophone (小爱 is
    # also how Xiaomi spells its own wake word, so expect cross-triggering if
    # one is in the room). Both spellings are accepted; add more here if the
    # journal shows near-misses on a real voice.
    variants: tuple[str, ...] = ("小爱同学", "小艾同学")

    vosk_model: Path = MODELS / "vosk-model-small-cn-0.22"

    # --- porcupine, for when an access key exists -----------------------
    # Picovoice needs a per-language parameter file, and 小艾同学 is not one of
    # the four built-in Chinese keywords (你好 / 咖啡 / 水饺 / 豪猪), so it also
    # needs a custom .ppn generated in the Picovoice Console. Fetch both with
    # scripts/get_porcupine_zh.sh.
    access_key: str = field(default_factory=lambda: os.environ.get("PICOVOICE_ACCESS_KEY", ""))
    porcupine_params: Path = MODELS / "porcupine_params_zh.pv"
    porcupine_keyword: Path = MODELS / "xiaoai_zh_raspberry-pi.ppn"

    # Fallback English phrase, used only by the openwakeword backend.
    model: str = "hey_jarvis"

    # Set AIA_NO_WAKE=1 to skip wake-word gating entirely and treat any speech
    # as a command. Useful for testing the rest of the pipeline on a machine
    # with no working wake model.
    disabled: bool = field(default_factory=lambda: os.environ.get("AIA_NO_WAKE") == "1")


@dataclass(frozen=True)
class VadConfig:
    # 0-3, higher is more aggressive about calling audio "not speech".
    aggressiveness: int = 2
    # How much trailing silence ends an utterance. This is a straight latency
    # tax on every command — it is counted in the fast-path budget — but too
    # short and normal mid-sentence pauses truncate the user.
    silence_ms: int = 400
    # Speech must persist this long to count as the start of the utterance.
    # The wake word fires while the last syllable of 小艾同学 is still being
    # spoken, and without this that tail starts the utterance — so the pause
    # before the actual command reads as end-of-speech and the turn captures
    # nothing. Two frames.
    onset_ms: int = 60
    # An utterance must contain at least this much voiced audio before it is
    # allowed to end, or be discarded. Guards against a cough or a door ending
    # the turn before the command arrives.
    min_speech_ms: int = 300
    # Audio kept from just before speech onset, so the first phoneme survives.
    preroll_ms: int = 300
    # Give up on an utterance that never ends, so a noisy room cannot pin the
    # assistant in listening mode forever.
    max_utterance_ms: int = 12000
    # Bail out if the user says nothing at all after the wake word. This is
    # the window to start speaking, so it has to tolerate a normal pause.
    max_wait_ms: int = 4000


@dataclass(frozen=True)
class SttConfig:
    host: str = "127.0.0.1"
    port: int = 8081
    model: Path = MODELS / "ggml-base.bin"

    # The single most valuable setting in this file. Whisper runs its encoder
    # over a padded 30 s window no matter how short the clip is; capping the
    # audio context to ~10 s cut encode from 1564 ms to 378 ms on this Pi with
    # no change to the transcripts. Below ~512 the decoder starts falling back
    # and looping and total time goes back *up*, so 512 is a floor, not a knob.
    audio_ctx: int = 512

    # Detect the spoken language per utterance. Costs ~390 ms versus naming a
    # language outright, which buys the spec's "switch language freely mid
    # conversation" requirement. A sticky-language scheme was built to avoid
    # this cost and then removed — stt/engine.py's docstring records why, with
    # the measurements. Turning this off makes AIA monolingual; it does not
    # make it faster in any way worth having.
    auto_detect: bool = True
    default_language: str = "en"
    supported_languages: tuple[str, ...] = ("en", "zh")

    # Word-level probabilities (the spec's "transcription confidence score").
    # Off by default: it forces `verbose_json`, whose DTW alignment pass adds
    # another ~390 ms. Turn it on for diagnostics, not for the fast path.
    verbose: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/inference"


@dataclass(frozen=True)
class TtsConfig:
    binary: Path = VENDOR / "piper" / "piper"
    voices: dict[str, Path] = field(default_factory=lambda: {
        "en": MODELS / "en_US-lessac-medium.onnx",
        "zh": MODELS / "zh_CN-huayan-medium.onnx",
    })
    # Every Piper voice we ship is 22050 Hz; read from the voice's .json at
    # load time rather than trusting this, which is only the fallback.
    sample_rate: int = 22050

    # Piper writes one wav per input line. Point that at tmpfs: the Pi boots
    # from an SD card, and a synthesis-per-utterance write cycle to flash is
    # both slower and unkind to the card.
    scratch: Path = Path("/dev/shm/aia-tts")


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    # Fast-path target from the spec. Exceeding it is not fatal, but the
    # state machine logs a warning so regressions surface in the journal
    # rather than in someone's patience.
    target_latency_ms: int = 2500


CONFIG = Config()
