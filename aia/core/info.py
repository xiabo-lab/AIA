"""What AIA is running, gathered from the objects that are actually running it.

The settings page asks a question that is easy to answer wrongly: *which*
microphone, *which* model, *which* voice. Config files say what was asked for.
This module reads the live objects instead — the open `Microphone`, the loaded
`Speaker` — because on this device those two answers have already diverged in
practice. Two known USB capsules, either of which may be plugged in, ALSA card
numbers that move on every re-plug, and a capture stream that can re-select a
different microphone mid-session when one is unplugged.

Where nothing live exists to ask, that is said plainly rather than dressed up:
the LLM is not built, and reporting a configured model name for it would be an
invention.

Nothing here is on the voice path. It runs on the HTTP thread when somebody
opens the settings page.
"""

from __future__ import annotations

import logging
import platform
import socket
import sys
import time
from pathlib import Path

from aia._version import source as version_source
from aia._version import version
from aia.core.config import Config

log = logging.getLogger(__name__)


def _file(path: Path) -> dict:
    """Name, presence and size of a model file, without raising on any of it."""
    out: dict = {"name": path.name, "path": str(path), "present": False}
    try:
        stat = path.stat()
        out["present"] = True
        out["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
    except OSError:
        pass
    return out


class SystemInfo:
    """Holds references to the live subsystems and reports on them.

    Everything is optional: the same snapshot has to be servable before the
    microphone is open and after a subsystem has failed, because that is
    precisely when somebody goes looking at it.
    """

    def __init__(self, cfg: Config, *, mic=None, speaker=None, started: float | None = None):
        self.cfg = cfg
        self.mic = mic
        self.speaker = speaker
        self.started = time.time() if started is None else started

    # ── sections ─────────────────────────────────────────────────────

    def aia(self) -> dict:
        return {
            "version": version(),
            "version_source": version_source(),
            "uptime_s": round(time.time() - self.started),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "host": socket.gethostname(),
            "executable": sys.executable,
        }

    def stt(self) -> dict:
        stt = self.cfg.stt
        return {
            "engine": "whisper.cpp (whisper-server)",
            "model": _file(stt.model),
            "url": stt.url,
            # The setting the whole latency budget rests on, and the one most
            # likely to be quietly different from what someone assumes.
            "audio_ctx": stt.audio_ctx,
            "readable_audio_ms": stt.readable_audio_ms,
            "language": ("detected per utterance" if stt.auto_detect
                         else stt.default_language),
            "supported_languages": list(stt.supported_languages),
            "timeout_s": stt.timeout_s,
        }

    def tts(self) -> dict:
        out: dict = {
            "engine": "Piper",
            "binary": _file(self.cfg.tts.binary),
        }
        if self.speaker is not None:
            # The voices that actually loaded, with the sample rate read from
            # each voice's own JSON rather than the config's fallback. A voice
            # whose file is missing is skipped at startup and this is where
            # that becomes visible.
            out["voices"] = self.speaker.describe()
        else:
            out["voices"] = [
                {"language": lang, "model": model.name, "loaded": False}
                for lang, model in self.cfg.tts.voices.items()
            ]
        return out

    def llm(self) -> dict:
        """There is no LLM, and this says so.

        `docs/PLAN.md` has it as M2 and there is no `aia/llm/`. Anything the
        router does not recognise is repeated back rather than answered. Naming
        the model the plan intends would read, on a settings page, as a model
        that is loaded.
        """
        return {
            "provider": None,
            "model": None,
            "status": "not built",
            "local": None,
            "note": ("The intent router answers known commands directly. "
                     "Open-ended conversation arrives with M2."),
        }

    def microphone(self) -> dict:
        if self.mic is None:
            return {"status": "not open"}
        try:
            info = dict(self.mic.describe())
        except Exception:
            log.exception("could not describe the microphone")
            return {"status": "unavailable"}
        info["status"] = "in use"
        return info

    def retention(self) -> dict:
        r = self.cfg.retention
        return {
            "hours": r.hours,
            "keep_recordings": r.keep_recordings,
            "recordings": str(r.recordings),
            "database": str(r.database),
            "sweep_interval_s": r.sweep_interval_s,
        }

    def snapshot(self) -> dict:
        return {
            "aia": self.aia(),
            "stt": self.stt(),
            "tts": self.tts(),
            "llm": self.llm(),
            "microphone": self.microphone(),
            "retention": self.retention(),
        }
