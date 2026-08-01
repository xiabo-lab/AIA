"""AIA — M1: the voice loop.

Wake word → capture → transcribe → speak. No LLM and no app control yet; this
milestone exists to prove the audio path works end to end on real hardware and
lands inside the latency budget, because everything after it is built on top.

Run it with the services up:

    ./scripts/run_services.sh start
    .venv/bin/python -m aia.main

Useful environment variables:

    AIA_NO_WAKE=1     skip wake-word gating; any speech is a command
    AIA_DEBUG=1       debug-level logging
    AIA_SAVE_AUDIO=1  keep each captured utterance under .bench/utterances
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import wave
from pathlib import Path

from aia.audio import wake as wake_mod
from aia.audio.capture import Microphone
from aia.audio.vad import Endpointer
from aia.core.config import CONFIG
from aia.core.state import Machine, State
from aia.plugins.base import Registry, Result
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System
from aia.router.fast import FastRouter
from aia.stt.engine import SpeechToText
from aia.tts.language import reply_language
from aia.tts.piper import Speaker
from aia.ui.panel import Panel

log = logging.getLogger("aia")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("AIA_DEBUG") else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
    # openWakeWord and its ONNX runtime are chatty about GPUs that a Pi
    # does not have.
    for noisy in ("openwakeword", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


CONFIRM_PROMPT = {
    "en": "That will {what}. Are you sure?",
    "zh": "这会{what}。确定吗？",
}

# Shown the moment the wake word fires, before there is any transcript.
LISTENING_TEXT = {"en": "Listening…", "zh": "我在听…"}

_YES = ("yes", "yeah", "yep", "sure", "confirm", "do it", "go ahead", "ok", "okay",
        "是", "是的", "对", "确定", "确认", "好", "好的", "可以", "没错")
_NO = ("no", "nope", "cancel", "stop", "don't", "never mind", "abort",
       "不", "不要", "取消", "别", "算了", "不用")


def is_affirmative(text: str) -> bool | None:
    """Yes, no, or neither.

    Returns None rather than guessing when the answer is ambiguous — for an
    irreversible action, "I could not tell" has to mean "do not do it".
    """
    stripped = text.strip().lower().rstrip(".!?。！？")
    if any(stripped == w or stripped.startswith(w) for w in _NO):
        return False
    if any(stripped == w or stripped.startswith(w) for w in _YES):
        return True
    return None


def save_utterance(audio, rate: int) -> None:
    """Write the captured utterance to .bench/utterances for offline replay.

    Every accuracy number so far came from synthesised speech, which is clean,
    close-miked and free of room reverb — it flatters every engine. Real
    captures are the only honest input for comparing recognisers, so keeping
    them costs a few KB and saves having to reproduce a misrecognition live.
    """
    directory = Path(__file__).resolve().parents[1] / ".bench" / "utterances"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
    log.info("saved utterance to %s", path)


def main() -> int:
    setup_logging()
    cfg = CONFIG
    save_audio = os.environ.get("AIA_SAVE_AUDIO") == "1"

    stt = SpeechToText(cfg.stt)
    log.info("waiting for whisper-server at %s", cfg.stt.url)
    if not stt.wait_ready():
        log.error("whisper-server never answered. Start it: ./scripts/run_services.sh start")
        return 1

    speaker = Speaker(cfg.tts)
    speaker.warm()

    panel = Panel(enabled=os.environ.get("AIA_NO_PANEL") != "1")

    registry = Registry([KodamaLite(), System()])
    router = FastRouter(registry)
    # An irreversible command waiting on a yes/no from the next turn.
    pending = None
    # Language of the last thing heard, so "Listening…" appears in whichever
    # language the user is already speaking rather than defaulting to English.
    stt_language = cfg.stt.default_language
    for plugin in registry.plugins:
        log.info("plugin %s: %s", plugin.name,
                 "available" if plugin.available() else "NOT reachable right now")

    detector = wake_mod.build(cfg.wake)
    endpointer = Endpointer(cfg.audio, cfg.vad)
    machine = Machine(cfg.target_latency_ms)
    machine.subscribe(lambda old, new: log.debug("[%s]", new.value))

    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        log.info("shutting down")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    with Microphone(cfg.audio) as mic:
        frames = mic.frames()
        log.info("AIA ready — say the wake word")

        while not stopping:
            frame = next(frames)
            if not detector.detect(frame):
                continue

            turn = machine.begin_turn()
            machine.to(State.LISTENING)
            # Shown before anything is transcribed, because the question the
            # user actually has at this moment is "did it hear me at all?".
            panel.status(LISTENING_TEXT.get(stt_language, LISTENING_TEXT["en"]))

            audio = endpointer.collect(frames)
            turn.mark("captured")
            if audio is not None and save_audio:
                save_utterance(audio, cfg.audio.target_rate)
            if audio is None:
                # Nothing was said. Take the "Listening…" line away rather
                # than leaving it up for five seconds implying otherwise.
                panel.hide()
                detector.reset()
                machine.end_turn()
                continue

            machine.to(State.THINKING)
            try:
                # Sticky language, with a confidence-triggered retry in the
                # other one. See stt/engine.py for why confidence and not script.
                result = stt.listen(audio)
                turn.mark("stt")

                # Whisper emits this marker for a clip with no speech in it —
                # it is not a transcript, and echoing it back is nonsense.
                text = result.text.strip()
                if text.upper().strip("[]") == "BLANK_AUDIO":
                    text = ""

                if not text:
                    machine.to(State.SPEAKING)
                    apology = ("I'm sorry, could you repeat that?"
                               if result.language == "en" else "抱歉，请再说一遍。")
                    panel.aia(apology)
                    speaker.say(apology, result.language)
                    machine.end_turn()
                    detector.reset()
                    mic.drain()
                    continue

                lang = reply_language(text, fallback=result.language)
                stt_language = result.language
                # What it heard, before it acts on it — so a misrecognition is
                # visible rather than something to infer from a wrong action.
                panel.user(text)

                # An irreversible command asked about on the previous turn is
                # waiting on a yes or no. This is checked before routing so
                # that "关机" -> "are you sure?" -> "关机" reads as a *reply*
                # rather than as a fresh request that asks again forever.
                if pending is not None:
                    answer, pending_intent = is_affirmative(text), pending
                    pending = None
                    if answer is True:
                        machine.to(State.ACTING)
                        reply = pending_intent.command.handler(
                            **pending_intent.arguments).say(lang)
                        turn.mark("acted")
                    elif answer is False:
                        reply = "Cancelled." if lang == "en" else "已取消。"
                    else:
                        # Neither yes nor no: treat it as a change of subject
                        # and drop the pending action rather than guessing.
                        reply = "Cancelled." if lang == "en" else "已取消。"
                    machine.to(State.SPEAKING)
                    panel.aia(reply)
                    speaker.say(reply, lang, blocking=False)
                    turn.mark("audio_out")
                    speaker.wait()
                    machine.end_turn()
                    detector.reset()
                    mic.drain()
                    continue

                intent = router.match(text, lang)
                turn.mark("routed")

                if intent is not None and intent.command.confirm:
                    # Destructive: ask first, act next turn. The spec requires
                    # confirmation for these, and a misrecognition powering the
                    # device off is exactly the failure that rule prevents.
                    pending = intent
                    question = CONFIRM_PROMPT.get(lang, CONFIRM_PROMPT["en"])
                    reply = question.format(what=intent.command.description)
                    log.info("awaiting confirmation for %s.%s",
                             intent.plugin.name, intent.command.name)
                elif intent is not None:
                    machine.to(State.ACTING)
                    if not intent.plugin.available():
                        reply = Result.failed(
                            f"{intent.plugin.description} is not currently running.",
                            f"{intent.plugin.description} 没有在运行。",
                        ).say(lang)
                    else:
                        outcome = intent.command.handler(**intent.arguments)
                        reply = outcome.say(lang)
                    turn.mark("acted")
                else:
                    # No known command. M2 puts the LLM here; until then, say
                    # what was heard so misrecognitions are still diagnosable
                    # by ear rather than only from the journal.
                    reply = f"You said: {text}" if lang == "en" else f"你说：{text}"

                machine.to(State.SPEAKING)
                # On screen first, then spoken. Writing text costs under a
                # millisecond while synthesis costs a few hundred, so showing
                # it first means the reply appears as the voice starts rather
                # than after it.
                panel.aia(reply)
                # Mark when audio STARTS, not when it finishes. Timing a
                # blocking play charges the whole spoken reply to the turn —
                # a two-second sentence looked like two seconds of latency,
                # which made a responsive turn read as 4 s over budget. What
                # the user experiences is the wait before hearing anything.
                speaker.say(reply, lang, blocking=False)
                turn.mark("audio_out")
                speaker.wait()

            except Exception:
                log.exception("turn failed")
                machine.to(State.ERROR)
            finally:
                machine.end_turn()
                detector.reset()
                # Drop anything captured while we were talking, so the
                # assistant does not hear itself.
                mic.drain()

    speaker.close()
    panel.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
