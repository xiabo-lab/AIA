"""Speech synthesis via a resident Piper process, one per voice.

Piper is dominated by loading its ONNX voice — measured on this Pi at 452 ms
for the English voice and 1190 ms for the Mandarin one, against 304 ms / 224 ms
to actually synthesise a sentence. Spawning `piper` per utterance therefore
costs more in startup than in work, and misses the latency budget outright.
So each voice gets one long-lived process, started once and fed on stdin.

Framing the stream is the fiddly part. In `--output-raw` mode Piper emits
PCM with no delimiter, so there is no way to tell where one utterance ends and
the next begins. In `--output_dir` mode it prints the path of each finished wav
on stdout, one line per input line — which is exactly the delimiter we need.
The files go to /dev/shm so this costs no SD-card writes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from aia.core.config import TtsConfig

log = logging.getLogger(__name__)


class Voice:
    """One resident Piper process bound to one .onnx voice."""

    def __init__(self, binary: Path, model: Path, scratch: Path):
        self.model = model
        self.scratch = scratch
        scratch.mkdir(parents=True, exist_ok=True)

        cfg_path = model.with_suffix(model.suffix + ".json")
        try:
            self.sample_rate = json.loads(cfg_path.read_text())["audio"]["sample_rate"]
        except Exception:
            self.sample_rate = 22050
            log.warning("could not read sample rate from %s; assuming %d",
                        cfg_path, self.sample_rate)

        t0 = time.monotonic()
        self._proc = subprocess.Popen(
            [str(binary), "--model", str(model), "--output_dir", str(scratch)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Piper loads the voice lazily on the first line, so this only starts
        # the process. The first synth() call absorbs the load cost; main.py
        # warms each voice at boot so no user ever pays it.
        self._lock = threading.Lock()
        log.info("piper up for %s in %.0f ms", model.name, (time.monotonic() - t0) * 1000)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesise one line. Returns (int16 samples, sample rate)."""
        # Newlines delimit utterances on stdin, so they cannot appear inside one.
        line = " ".join(text.split())
        if not line:
            return np.zeros(0, dtype=np.int16), self.sample_rate

        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("piper process has exited")
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            path = self._proc.stdout.readline().strip()

        if not path:
            raise RuntimeError("piper produced no output path")
        wav_path = Path(path)
        with wave.open(str(wav_path), "rb") as w:
            rate = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        # tmpfs is RAM; leaving these around would slowly consume it.
        wav_path.unlink(missing_ok=True)
        return data, rate

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


class Speaker:
    """Picks a voice by language and plays the result."""

    def __init__(self, cfg: TtsConfig):
        self.cfg = cfg
        self._voices: dict[str, Voice] = {}
        for lang, model in cfg.voices.items():
            if not model.exists():
                log.warning("voice for %r missing at %s; that language will be silent",
                            lang, model)
                continue
            self._voices[lang] = Voice(cfg.binary, model, cfg.scratch)

    def warm(self) -> None:
        """Force each voice to load now, off the critical path.

        Without this the first command of the session pays up to 1.2 s of ONNX
        load on top of everything else.
        """
        for lang, voice in self._voices.items():
            t0 = time.monotonic()
            voice.synth("Ready." if lang == "en" else "准备好了。")
            log.info("warmed %s voice in %.0f ms", lang, (time.monotonic() - t0) * 1000)

    def say(self, text: str, language: str = "en", blocking: bool = True) -> float:
        """Speak `text`. Returns milliseconds to first audio."""
        voice = self._voices.get(language) or next(iter(self._voices.values()), None)
        if voice is None:
            log.error("no voices available; cannot speak %r", text)
            return 0.0

        t0 = time.monotonic()
        samples, rate = voice.synth(text)
        first_audio_ms = (time.monotonic() - t0) * 1000
        if samples.size:
            self._play(samples, rate, blocking)
        log.info("tts[%s] %.0f ms to audio: %r", language, first_audio_ms, text)
        return first_audio_ms

    @staticmethod
    def _play(samples: np.ndarray, rate: int, blocking: bool) -> None:
        """Play, re-querying the device list once if it has gone stale.

        PortAudio enumerates devices when it initialises and never looks
        again. The Pi's only sink is HDMI, which PipeWire tears down and
        recreates when the last client goes away — so a perfectly healthy
        system can leave this process holding a device list in which the
        default output is -1, and every reply then dies with
        "Error querying device -1" while the rest of the turn works fine.
        Re-initialising rebuilds the list from what is actually there now.
        """
        try:
            sd.play(samples, rate)
        except Exception as first:
            log.warning("audio output failed (%s); re-querying devices", first)
            try:
                sd._terminate()
                sd._initialize()
                sd.play(samples, rate)
            except Exception as second:
                # Losing the spoken reply is bad but survivable; the command
                # itself has already run. Never take the assistant down for it.
                log.error("audio output still unavailable: %s", second)
                return
        if blocking:
            try:
                sd.wait()
            except Exception:
                log.debug("sd.wait() failed after a device reset", exc_info=True)

    def wait(self) -> None:
        """Block until playback finishes.

        Separate from `say()` so a caller can timestamp the moment audio
        *starts* — which is what the user perceives as the response time — and
        only then wait for it to finish before listening again.
        """
        sd.wait()

    def close(self) -> None:
        for voice in self._voices.values():
            voice.close()
