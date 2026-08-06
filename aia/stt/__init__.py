"""Speech to text.

`build(cfg, rate)` is the only thing the rest of AIA needs from this package.
It returns something with a `listen()` on it, and which one is a configuration
question — `stt.backend`, or `AIA_STT_BACKEND` for a single run.

    stt = build(CONFIG.stt, CONFIG.audio.target_rate)
    text = stt.listen(audio).text

The environment variable exists for one job in particular: running the same
recording through both backends without editing a file the other one is also
being measured from. See `scripts/stt_test.py`.

Importing a backend is deferred until it is chosen. `sherpa_onnx` is a large
native wheel and `requests` reaches out to a server; neither should be a cost
paid by whoever picked the other one, and on a developer machine with no ARM
wheels the unused import would be a hard failure rather than an unused import.
"""

from __future__ import annotations

import logging

from aia.core.config import SttConfig
from aia.stt.base import (
    SttBackend,
    SttUnavailable,
    Transcript,
    as_float32,
    detect_script,
    parse_lang_tag,
    strip_meta,
)

log = logging.getLogger(__name__)

__all__ = [
    "SttBackend",
    "SttUnavailable",
    "Transcript",
    "as_float32",
    "build",
    "detect_script",
    "parse_lang_tag",
    "strip_meta",
]


def build(cfg: SttConfig, rate: int) -> SttBackend:
    """The configured speech recogniser, not yet loaded.

    `rate` is the sample rate of the audio that will be handed to `listen()`,
    and is required rather than defaulted. Both backends need it and both have
    a way of being silently wrong when it is not what they were told — see
    either one's `__init__`.

    An unknown backend name raises here, at startup, rather than falling back
    to a default. A typo in `stt.backend` that silently ran Whisper would
    present as "the new recogniser is no better", which is the most expensive
    possible way to find out.
    """
    backend = cfg.backend.strip().lower()

    if backend == "sensevoice":
        from aia.stt.sensevoice import SenseVoiceSTT
        return SenseVoiceSTT(cfg, rate)

    if backend == "whisper":
        from aia.stt.whisper import WhisperSTT
        return WhisperSTT(cfg, rate)

    raise SttUnavailable(
        f"unknown stt.backend {cfg.backend!r} — expected 'sensevoice' or 'whisper'"
    )
