"""Which language to answer in, and with which voice.

Deliberately the only place that maps a language code to a model. Cantonese is
out of scope for this stage — Piper ships no `yue` voice at all (35 languages,
`zh_CN` only) and stock Whisper is unusable on Cantonese at ~49.5% CER — but
the whole point of routing every such decision through here is that adding it
later is a table entry and a model file, not a refactor. If you find yourself
branching on language anywhere else, move it here instead.
"""

from __future__ import annotations

import logging

from aia.stt.engine import detect_script

log = logging.getLogger(__name__)

# Reply language for a given detected input language. Identity today; the row
# that will change when Cantonese lands is `yue`, which — until a yue voice
# exists — would have to answer in Mandarin and should be an explicit,
# reviewable decision rather than a silent fallback.
REPLY_LANGUAGE = {
    "en": "en",
    "zh": "zh",
}

DEFAULT = "en"


def reply_language(transcript_text: str, fallback: str = DEFAULT) -> str:
    """Language to answer in, inferred from what the user just said.

    Uses the script of the transcript rather than the language tag Whisper was
    asked for, because the sticky-language scheme in stt/engine.py can ask for
    the wrong one — and when it does, the script of the returned text is the
    honest signal about what was actually spoken.
    """
    detected = detect_script(transcript_text)
    if detected is None:
        return fallback
    resolved = REPLY_LANGUAGE.get(detected)
    if resolved is None:
        log.warning("no reply voice for %r; falling back to %s", detected, fallback)
        return fallback
    return resolved
