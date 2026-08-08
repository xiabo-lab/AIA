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
class MicProfile:
    """A known microphone, and the mixer state it was measured in.

    `match` is a substring of the PortAudio device name. It has to be specific
    enough to identify *one* capsule: a bare "USB" matched both microphones on
    this Pi and the pick fell to enumeration order, which is how capture once
    ran on a different capsule than the log named.

    `gain` is a step on that device's own `Mic` scale, not a dB figure. The
    scales differ — 0-30 on one of these, 0-16 on the other — and only the step
    is portable to amixer, so the dB it lands on is in the comment beside each
    profile. That is the number worth comparing between microphones.

    `None` for `gain` or `agc` leaves that control alone.
    """

    match: str
    gain: int | None = None
    agc: bool | None = None
    note: str = ""


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

    # Known microphones, in preference order: the first one actually plugged in
    # is used, and its mixer settings are applied on every open. Unplug one and
    # plug in the other and nothing here needs editing.
    #
    # Applying the gain from here rather than leaving it to ALSA's stored state
    # is deliberate. `/var/lib/alsa/asound.state` is keyed per card and only
    # holds what `alsactl store` last captured, so a microphone the system has
    # never seen has no entry at all — the Generalplus arrived at 30/30
    # (33.00 dB), which is deep into the range where proximity to full scale
    # drives voiced% to 100% and the endpointer can never terminate. A stored
    # entry can also go stale against the tuning it was meant to hold: the TI's
    # said 16 (23.81 dB) long after this project settled on 8.
    #
    # Both gains below are the level-matched pair from the 2026-08-05 A/B,
    # which is why they land within 0.1 dB of each other.
    microphones: tuple[MicProfile, ...] = (
        # Generalplus, omnidirectional condenser. In use since 2026-08-05, and
        # chosen for coverage rather than measured quality — a paired A/B at
        # matched gain could not separate the two capsules (13/13 wake
        # detections each, identical transcripts), and a 27-trial wake_test on
        # this one scored 27/27 against the TI's 25/27, which is 2 trials and
        # not a distinguishable difference.
        #
        # Two things to watch, neither yet measured in anger. It has a DC
        # offset of ~305 LSB (0.93% of full scale) and sub-100 Hz content, and
        # there is no high-pass anywhere in this pipeline — only the anti-alias
        # lowpass — so that reaches webrtcvad and Whisper, and inflates any
        # threshold compared against wideband RMS. And it advertises internal
        # noise reduction, which is level-dependent processing of the same
        # family as the AGC that caused the saturation incident.
        #
        # **12.00 dB was too hot and the endpointer paid for it.** Over 78 real
        # captures at gain 16: 28% ran to the 10 s cap, 31% peaked at or above
        # −1 dBFS, and 28% contained clipped samples. Near full scale webrtcvad
        # calls every frame speech, `silence_ms` is never satisfied, and the
        # utterance can only end at `max_utterance_ms` — so more than a quarter
        # of everything said to this assistant took ten seconds to be heard,
        # whatever it was. Worst outdoors, where people speak closer and the
        # room is louder, but present everywhere: the capped captures span the
        # whole day, not one session.
        #
        # This scale is 0-30 spanning −12.00 to +33.00 dB, so 1.5 dB a step.
        # Step 8 is 0.00 dB, which is 12 dB of headroom against a distribution
        # whose p90 peak was −0.0 dBFS. Chosen to move the loud tail clear of
        # the rail rather than to centre the quiet one, because saturation
        # costs a whole turn while a quiet capture costs nothing measurable —
        # the speech-band SNR here is 45 dB and SenseVoice normalises.
        #
        # The thing to watch is the other end: the p10 peak was −19.4 dBFS and
        # is now around −31, and no distance test has ever been run on this
        # capsule. If wake detection falls off across the room, that is where
        # it will show, and step 12 (6.00 dB) is the halfway house.
        MicProfile(match="USB Microphone", gain=8, agc=False,
                   note="Generalplus, omnidirectional (8/30 = 0.00 dB)"),
        # TI PCM2902. Every recognition threshold in this project was tuned
        # against this capsule, so it stays as the fallback. 8/16 is the value
        # the audio review settled on; 16/16 is what caused the clipping.
        MicProfile(match="USB PnP Sound Device", gain=8, agc=False,
                   note="TI PCM2902 (8/16 = 11.90 dB)"),
    )

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
    # ...unless the capture holds this much voiced audio in total, in which case
    # the unbroken-run bar is waived.
    #
    # `min_run_ms` was measured against the population it was built to reject:
    # captures holding 300-660 ms of voiced audio whose longest run was 150 ms.
    # It was never meant to judge a capture holding *seconds* of speech, and on
    # 2026-08-07 it threw two of those away — 3780 ms and 3240 ms of voiced
    # audio, discarded whole because no single run reached the bar:
    #
    #   19:35:34  discarding utterance with 3780 ms of speech (390 ms unbroken)
    #   21:16:54  discarding utterance with 3240 ms of speech (390 ms unbroken)
    #
    # A cough is not four seconds long. Both were captures made over ducked
    # music, where the VAD toggles on the song and no run survives — but the
    # user's command is in there, and rejecting it outright guarantees the turn
    # is lost, where transcribing it at least gives the router something.
    #
    # 1500 ms sits above the top of the real-phrase range these recordings show
    # (1140-1290 ms voiced), so an ordinary command still has to clear the run
    # bar and only the long, fragmented captures are let through on total alone.
    ample_speech_ms: int = 1500
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
class SenseVoiceConfig:
    """SenseVoiceSmall INT8, run in-process through sherpa-onnx.

    The default recogniser. Fetch the model with `scripts/get_sensevoice.sh`;
    nothing here is downloaded at runtime and nothing here talks to a network
    service, which is the point — STT is offline, with no API key and no cloud
    fallback to quietly succeed when the model is missing.
    """

    # The official multilingual release: zh, en, ja, ko, yue. The archive
    # carries both precisions and this names the INT8 one explicitly, because
    # `model.onnx` beside it is the fp32 weights and picking it up by accident
    # would present as "SenseVoice is slower than advertised" rather than as a
    # wrong file.
    directory: Path = MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    model_name: str = "model.int8.onnx"
    tokens_name: str = "tokens.txt"

    # Start at 2 and measure 1/2/3/4 before moving it. The Pi 5 has four cores
    # and this is not the only thing on them — the wake recogniser runs
    # continuously, and Piper wants a core the moment the transcript lands, so
    # the fastest setting for STT in isolation is not necessarily the fastest
    # turn. `scripts/stt_test.py --threads 1,2,3,4` sweeps it.
    num_threads: int = 2

    # "auto", or one of the five the model knows. Leave it on "auto": naming a
    # language is what broke bilingual use under Whisper, and the whole reason
    # this household has an assistant that listens in three languages is that
    # nobody should have to select one. See aia/stt/sensevoice.py.
    language: str = "auto"

    # Inverse text normalisation — "二零二五" comes back as "2025", and
    # sentences arrive punctuated. Wanted: the router strips punctuation before
    # matching anyway, and digits are what the number-taking commands parse.
    use_itn: bool = True

    # onnxruntime execution provider. "cpu" is the only one that exists on this
    # Pi; the field is here so that trying anything else is a config change
    # rather than a code change.
    provider: str = "cpu"

    # What the model can be asked to recognise, as opposed to what AIA can
    # answer in. Those are different sets and conflating them is what this
    # field exists to prevent: Cantonese is recognised and then answered in
    # Mandarin, because Piper has no yue voice. See SttConfig.supported_languages.
    recognised_languages: tuple[str, ...] = ("zh", "en", "yue")

    @property
    def model(self) -> Path:
        return self.directory / self.model_name

    @property
    def tokens(self) -> Path:
        return self.directory / self.tokens_name

    @property
    def is_auto(self) -> bool:
        return self.language.strip().lower() in ("auto", "")


@dataclass(frozen=True)
class SttConfig:
    # Which recogniser runs. "sensevoice" is the default and "whisper" is kept
    # as the fallback — it is the only transcriber here with a measured
    # accuracy record on real captures from this room, and a comparison needs
    # both halves present.
    #
    # The environment variable is for running one recording through both
    # without editing the file the other is being measured from.
    backend: str = field(
        default_factory=lambda: os.environ.get("AIA_STT_BACKEND", "sensevoice"))

    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)

    # ── whisper.cpp, the fallback backend ────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8081
    model: Path = MODELS / "ggml-base.bin"

    # The single most valuable setting in this file *for the whisper backend*,
    # and meaningless for SenseVoice, which is not autoregressive and has no
    # padded window to cap — its cost tracks the real length of the audio.
    # `__post_init__` only enforces the capture-length invariant below when
    # whisper is the backend selected, for the same reason.
    #
    # Whisper runs its encoder
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

    # The languages AIA can *answer* in — one Piper voice each. This is NOT the
    # set the recogniser can hear: SenseVoice also returns `yue`, and AIA
    # answers Cantonese in Mandarin because there is no Cantonese voice to
    # answer it with. Adding `yue` here without adding a voice would route a
    # Cantonese turn to `Speaker.say("yue")`, which falls back to whichever
    # voice happens to be first in the dict — the silent kind of wrong.
    # `SenseVoiceConfig.recognised_languages` is the other set.
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
class RetentionConfig:
    """How long AIA keeps what it recorded and what it heard.

    One number, `hours`, governs both the conversation database and the saved
    audio, because they are the same promise to the person in the room: nothing
    said here outlives a day.

    The audio has a second rule the text does not. Recordings cannot be remade —
    they are a particular speaker, room and microphone on a particular day — and
    a test that pruned the live directory to 3 files once destroyed 101 of them.
    So `keep_recordings` protects the newest N regardless of age, and only files
    *beyond* that set are subject to the clock. Conversation text carries no such
    exception: it expires at `hours` and that is all.
    """

    hours: float = 24.0

    # Where AIA_SAVE_AUDIO writes captured utterances. Named here rather than
    # rebuilt from __file__ at the two places that touch it, because a writer
    # and a deleter that each work out their own path are one refactor away
    # from disagreeing — and the way that disagreement presents is a cleanup
    # pointed at the wrong directory.
    #
    # It must stay this specific. `.bench/` also holds `wake-trials-*`, which
    # are irreplaceable measurement corpora — 27 recordings captured through
    # the broken decimator, the only surviving "before" audio in the project.
    # Nothing here may ever be widened to `.bench` itself.
    recordings: Path = ROOT / ".bench" / "utterances"

    # The newest recordings, always kept, however old they are. A floor rather
    # than a target: on a quiet day the clock alone would empty the directory,
    # and the last hundred captures are what a misrecognition is diagnosed from.
    keep_recordings: int = 100

    database: Path = ROOT / "conversation_history.db"

    # How often the sweep runs. Expiry is not a deadline — a recording that
    # lives an extra quarter of an hour costs nothing — and a sweep that runs
    # constantly is a wakeup on a device whose whole job is to be listening.
    sweep_interval_s: float = 900.0

    @property
    def seconds(self) -> float:
        return self.hours * 3600.0


@dataclass(frozen=True)
class UiConfig:
    """The conversation and settings web UI.

    **Loopback by default, and think before changing it.** This serves a
    transcript of everything said in the room, which is a more sensitive thing
    than it first sounds, and there is no authentication in front of it. The
    same reasoning is why Kodama-Lite's control endpoint binds loopback under a
    random token rather than a fixed public port.
    """

    host: str = "127.0.0.1"
    port: int = 8090

    # How many messages the page loads on first paint. The display is 440 px
    # tall, so this is scrollback, not a screenful.
    backlog: int = 200

    # Set AIA_NO_WEB=1 to run without it — the voice loop does not depend on
    # this in any way, and a port already in use must not stop the assistant.
    enabled: bool = field(default_factory=lambda: os.environ.get("AIA_NO_WEB") != "1")


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
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    ui: UiConfig = field(default_factory=UiConfig)

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
        # Whisper only. SenseVoice reads whatever it is given — there is no
        # fixed encoder window to overrun — so applying this to it would be
        # enforcing an invariant that does not exist, and would make
        # `max_utterance_ms` look like it had been chosen for a reason it had
        # not. The number stays at 10000 for both because it is *also* about
        # how long the assistant may sit in a noisy room, which is unchanged.
        if self.stt.backend.strip().lower() == "whisper":
            readable = self.stt.readable_audio_ms
            if self.vad.max_utterance_ms > readable:
                raise ValueError(
                    f"vad.max_utterance_ms={self.vad.max_utterance_ms} exceeds what "
                    f"whisper will read at stt.audio_ctx={self.stt.audio_ctx} "
                    f"({readable} ms). Audio past that is discarded silently. "
                    f"Raise audio_ctx or lower max_utterance_ms."
                )

        # Both recognisers and the wake word are 16 kHz models. Nothing here
        # breaks loudly at another rate — sherpa-onnx would resample on the
        # voice path, and whisper would be handed a WAV header it believed —
        # so a mismatch presents as latency or as confident wrong text rather
        # than as an error. `AudioConfig.capture_rate` is the one to change if
        # the microphone cannot do 48 kHz; this is what the models want.
        if self.audio.target_rate != 16000:
            raise ValueError(
                f"audio.target_rate={self.audio.target_rate} but the STT and wake "
                f"models are all 16 kHz. Decimate to 16000 in capture instead."
            )

        # `ample_speech_ms` waives the unbroken-run bar, and it can only mean
        # something if it is *above* the bar for admitting a capture at all.
        # At or below `min_speech_ms` every capture that qualifies also waives,
        # which silently deletes `min_run_ms` instead of relaxing it — the
        # cough guard would be gone and nothing would say so.
        if self.vad.ample_speech_ms <= self.vad.min_speech_ms:
            raise ValueError(
                f"vad.ample_speech_ms={self.vad.ample_speech_ms} is at or below "
                f"vad.min_speech_ms={self.vad.min_speech_ms}, which waives "
                f"min_run_ms for every capture rather than for long ones. "
                f"Raise ample_speech_ms."
            )

        # Two rules delete recordings and they must not contradict each other.
        # `keep_recordings` is a promise that the newest N survive whatever the
        # clock says; `keep_utterances` is the count cap the writer applies on
        # every turn. A floor above the cap is a promise the cap immediately
        # breaks — the retention sweep would spare files that `save_utterance`
        # had already deleted, so the floor would silently mean the cap.
        if self.retention.keep_recordings > self.audio.keep_utterances:
            raise ValueError(
                f"retention.keep_recordings={self.retention.keep_recordings} is above "
                f"audio.keep_utterances={self.audio.keep_utterances}, so the count cap "
                f"would delete recordings the retention floor promises to keep. "
                f"Raise keep_utterances or lower keep_recordings."
            )


CONFIG = Config()
