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

    # How many captured utterances AIA_SAVE_AUDIO keeps before pruning the
    # oldest. Measured at ~148 KB each, so 200 is ~30 MB and roughly a week
    # of ordinary use — long enough that a misrecognition can still be dug up
    # after a weekend, bounded so it cannot grow to a gigabyte of SD card.
    keep_utterances: int = 200

    # Substring matched against the portaudio device name. The card number
    # moves when other USB audio is plugged in, so match on name — but match
    # on a name that identifies *one* capsule. A bare "USB" matched both mics
    # on this Pi and the pick fell to enumeration order, which is how capture
    # ended up on a different microphone than the log named. The two here:
    #
    #   "USB PnP Sound Device"  TI PCM2902,   mixer range 0-16 (23.81 dB max)
    #   "USB Microphone"        Generalplus,  mixer range 0-30 (33.00 dB max)
    #
    # Switching microphones is a one-line change here. Whichever is named, the
    # resolved ALSA card is logged at startup — check it against /proc/asound
    # rather than trusting this string to have meant what you thought.
    device_match: str = "USB PnP Sound Device"

    # 30 ms at 16 kHz = 480 samples. webrtcvad accepts only 10/20/30 ms
    # frames, and 30 ms is the cheapest of the three.
    frame_ms: int = 30

    @property
    def frame_samples(self) -> int:
        return self.target_rate * self.frame_ms // 1000

    @property
    def capture_block(self) -> int:
        """One output frame's worth of audio, at the capture rate.

        This is *not* a blocksize to ask PortAudio for — doing that loses a
        fifth of all audio on this device, see `Microphone`. It is the size the
        capture callback coalesces the device's ragged blocks up to before
        queueing them, which is what lets the queue be bounded by duration
        instead of by a block count that means nothing.
        """
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

    # 小爱 is also how Xiaomi spells its own wake word, so expect
    # cross-triggering if one is in the room.
    #
    # What the recogniser actually returns when this user says the phrase,
    # measured over 40 attempts. The leading 小 is dropped about half the
    # time: it is a third tone, low and flat, and it sits at the very start
    # of the utterance where the recogniser has the least context — while 爱
    # (fourth tone, falling) is loud and survives. So the reliable core of
    # the phrase is 爱同学, not 小爱同学, and the forms below are the shapes
    # that core actually arrives in.
    #
    # These are compared by *sound*, so they only need to cover genuinely
    # different pronunciations, not different spellings: 哎同学 also matches
    # 唉同学 and 爱同学 for free. Duplicates by sound are collapsed at startup,
    # so listing both 小爱同学 and 小艾同学 costs nothing but is also worth
    # nothing — these four are two targets.
    #
    # Do not add a variant shorter than three syllables, and do not expect the
    # threshold alone to keep a three-syllable one safe. 爱同学 is only one
    # syllable away from 同学, which is ordinary speech, so the matcher
    # additionally requires the phrase's *first* syllable to have been heard.
    # See `_covers_onset` in aia/audio/wake.py — without that guard 同学 scores
    # 0.875 here and the assistant wakes up on conversation.
    variants: tuple[str, ...] = ("小爱同学", "小艾同学", "哎同学", "爱同学")

    # How closely the tail of what was heard must *sound* like the phrase.
    # Matching is done on pinyin, so spelling differences cost nothing and
    # this is really a tolerance for dropped or slurred syllables.
    #
    # Lower to catch more, at the price of firing on ordinary speech: 0.70
    # accepts roughly one wrong syllable in four. Measure before changing it —
    # scripts/wake_test.py reports the score of every attempt, so the right
    # value for a given voice and room is something to read off, not guess.
    # Its second phase reads ordinary sentences and reports what they score;
    # that number, not this one, is what says whether a change is safe.
    similarity: float = 0.72

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
    #
    # It was 300 ms, which is about one syllable, and that is what it behaved
    # like: measured across two runs of 27 trials, every missed capture held
    # 300-660 ms of voiced audio while a real phrase holds 1140-1290 ms. Two
    # misses in the last run held 300 and 330 ms — the guard exactly. A breath
    # or a hesitation cleared it, the capture closed, and the command that
    # followed was never heard.
    #
    # 500 ms still admits the shortest real command: at the ~320 ms per
    # syllable these recordings show, 暂停 is around 640 ms. Raising it further
    # would start cutting off genuine two-syllable commands, so this is a floor
    # set by what people say, not by what noise happens to measure.
    min_speech_ms: int = 500
    # ...of which this much has to be *contiguous*. Total voiced time alone
    # cannot tell a command from a scatter of unrelated noises across a few
    # seconds, and one missed capture held 330 ms of speech whose longest
    # unbroken run was 150 ms. Real phrases in these recordings run 1260 ms
    # unbroken.
    min_run_ms: int = 400
    # Audio kept from just before speech onset, so the first phoneme survives.
    #
    # 300 ms was less than the phrase it most needed to protect. At ~320 ms per
    # syllable, 小爱 takes about 640 ms, so an onset detected even slightly
    # late lost the head of the wake phrase and could not get it back — which
    # is precisely the "leading 小 is dropped about half the time" that
    # WakeConfig.variants exists to accommodate. Costs memory, not latency:
    # these frames have already been captured either way.
    preroll_ms: int = 700
    # Give up on an utterance that never ends, so a noisy room cannot pin the
    # assistant in listening mode forever.
    #
    # Capped at what the transcriber will actually read rather than at a round
    # number: SttConfig.audio_ctx of 512 gives Whisper 512 of its 1500 encoder
    # positions, which is 512/1500 x 30 s = 10.2 s. Anything captured beyond
    # that was being discarded downstream in silence.
    max_utterance_ms: int = 10000
    # Bail out if the user says nothing at all after the wake word. This is
    # the window to start speaking, so it has to tolerate a normal pause.
    max_wait_ms: int = 4000
    # Longer window when waiting for a yes/no. The assistant has just asked a
    # question and is holding the floor, so the user has to hear it, take it
    # in, and decide — and silence here cancels an irreversible action, which
    # is not something to do impatiently.
    confirm_wait_ms: int = 8000


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

    # Detect the spoken language on every utterance. Costs ~600 ms against
    # naming one outright, measured over 35 real captures, and it is what
    # lets a Mandarin command follow an English one with nothing to select.
    #
    # A sticky-language scheme held this for a whole conversation to avoid
    # that cost. It was removed: Whisper does not fail on audio in a language
    # it was not asked for, it silently *translates*, so the second language
    # became five minutes of fluent unroutable English. stt/engine.py's
    # docstring carries the measurements and the three cheaper schemes that
    # were tried and rejected — including a bigger model, which is genuinely
    # more accurate and four times too slow.
    #
    # Turning this off makes AIA monolingual. It does not make it faster in
    # any way worth having.
    auto_detect: bool = True

    # Only used when auto_detect is off, or when a transcript is empty and
    # there is no script to read a language from.
    default_language: str = "en"
    supported_languages: tuple[str, ...] = ("en", "zh")

    # How long to wait on the server before giving up. This is a bound on a
    # hang, not a target: it must never abort a request that would have
    # succeeded. Measured over 35 real captures with `base` and `-ac 512`,
    # transcription is 1323 ms median and 1506 ms at p90, so this is ~6x p90.
    #
    # It was 30 s, hardcoded, and `listen()` can transcribe twice — so a
    # wedged whisper-server held the assistant for a minute with the music
    # ducked and "Listening…" on screen. Against a 2.5 s turn budget, a
    # timeout that long is indistinguishable from a hang to the person
    # standing there.
    timeout_s: float = 10.0

    # Word-level probabilities (the spec's "transcription confidence score").
    # Off by default: it forces `verbose_json`, whose DTW alignment pass adds
    # another ~390 ms. Turn it on for diagnostics, not for the fast path.
    verbose: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/inference"

    @property
    def readable_audio_ms(self) -> int:
        """How much audio the encoder will actually look at, in ms.

        Whisper's encoder runs at 50 positions per second — its full 1500 are
        the padded 30 s window — so each position is 20 ms and `audio_ctx`
        converts straight into a duration. `Config.__post_init__` uses this to
        stop the capture cap drifting past it.
        """
        return self.audio_ctx * 20


@dataclass(frozen=True)
class KodamaConfig:
    """Talking to the music player."""

    # The MPRIS bus name to drive. Worth being settable rather than a
    # constant: this app publishes *two* players — its own service and a
    # WebKit media session for the same audio, from the webview — so which one
    # is meant is a real distinction, and the webview's name carries a
    # per-launch instance suffix.
    player: str = "kodamalite"

    # Bounds on a hang, not targets. Both were 3 s and hardcoded, which is
    # over the whole turn's 2.5 s budget on its own. Measured, a `playerctl`
    # call costs 6-7 ms and the control POST is to localhost, so this is two
    # orders of magnitude of slack and still cannot blow the budget by itself.
    playerctl_timeout_s: float = 1.5
    control_timeout_s: float = 1.5


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
    kodama: KodamaConfig = field(default_factory=KodamaConfig)

    # Fast-path target from the spec. Exceeding it is not fatal, but the
    # state machine logs a warning so regressions surface in the journal
    # rather than in someone's patience.
    target_latency_ms: int = 2500

    def __post_init__(self) -> None:
        """Check the invariants that span two config classes.

        These are the ones a comment cannot enforce. `max_utterance_ms` was
        capped at what Whisper will actually read, and the two numbers were
        chosen together and written down in separate dataclasses — so raising
        the capture cap without touching `audio_ctx` silently throws the extra
        audio away *inside* Whisper, with a full transcript of the first ten
        seconds coming back and nothing anywhere to say the rest was dropped.

        Failing at import is deliberate. This can only be reached by editing
        config.py, so it is a developer error caught the first time the
        process starts, not something a user can trigger.
        """
        readable = self.stt.readable_audio_ms
        if self.vad.max_utterance_ms > readable:
            raise ValueError(
                f"vad.max_utterance_ms={self.vad.max_utterance_ms} exceeds what "
                f"whisper will read at stt.audio_ctx={self.stt.audio_ctx} "
                f"({readable} ms). Audio past that is discarded silently. "
                f"Raise audio_ctx or lower max_utterance_ms."
            )


CONFIG = Config()
