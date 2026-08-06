"""Which language to answer in, and with which voice.

Deliberately the only place that maps a language code to a model.

**Half of the Cantonese problem is now solved and this is the half that is
not.** SenseVoice recognises Cantonese as Cantonese, so the old reason to keep
it out of scope — stock Whisper at ~49.5% CER — is gone. Piper's side has not
moved: still 35 languages, still `zh_CN` only, still no `yue` voice. So a
Cantonese command is understood correctly and answered in Mandarin.

That decision is made in `aia/stt/sensevoice.py`'s `_REPLY_IN`, where the
language the recogniser reports is folded onto a voice AIA owns, because that
is the only place a `yue` tag exists. By the time text reaches this module it
is Han and indistinguishable from Mandarin — see `detect_script`. When a yue
voice lands, that table and `TtsConfig.voices` are the whole change.

If you find yourself branching on language anywhere else, move it here instead.
"""

from __future__ import annotations

import logging

from aia.stt import detect_script

log = logging.getLogger(__name__)

# Reply language for a given *script*, which is all `detect_script` can report.
# Identity, and it stays identity: Cantonese never reaches here as `yue`,
# because written Cantonese is Han. The yue row people expect to find here is
# `_REPLY_IN` in aia/stt/sensevoice.py, keyed on what the recogniser heard
# rather than on what the text looks like.
REPLY_LANGUAGE = {
    "en": "en",
    "zh": "zh",
}

DEFAULT = "en"


def reply_language(transcript_text: str, fallback: str = DEFAULT) -> str:
    """Language to answer in, inferred from what the user just said.

    Uses the script of the transcript rather than the language tag the
    recogniser was asked for. Those differ in the case that matters: a caller
    that named a language outright — the confirmation does — named the language
    of the *question*, which is not necessarily the one the answer came back
    in. The script of the returned text is the honest signal there.

    `fallback` is normally `Transcript.language`, which under SenseVoice has
    already had the recogniser's own verdict folded into it. That makes this a
    refinement of a good answer rather than the only answer, which is what it
    was under Whisper's fast `json` mode where nothing was reported at all.
    """
    detected = detect_script(transcript_text)
    if detected is None:
        return fallback
    resolved = REPLY_LANGUAGE.get(detected)
    if resolved is None:
        log.warning("no reply voice for %r; falling back to %s", detected, fallback)
        return fallback
    return resolved
