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
from dataclasses import replace
import time
import wave
from pathlib import Path

from aia.audio import wake as wake_mod
from aia.audio.capture import Microphone
from aia.audio.ducking import Ducker
from aia.audio.vad import Endpointer
from aia.core.config import CONFIG
from aia.core.state import Machine, State
from aia.plugins.base import Registry, Result
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System
from aia.router.fast import FastRouter, normalise
from aia.stt.engine import SpeechToText, detect_script
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

# Shown while waiting for a yes/no, so it is obvious the assistant is still
# holding the floor and no wake word is needed.
CONFIRM_LISTEN = {"en": "Say yes or no…", "zh": "请回答“确定”或“取消”…"}

# After a turn that captured nothing, ignore the wake word for this long.
# A wake word that fires on silence — a false positive, or AIA_NO_WAKE=1 where
# anything counts — otherwise loops: open a turn, wait out `max_wait_ms`, close
# it, open another. Each pass ducks and un-ducks whatever is playing and spawns
# playerctl processes to do it, so the failure is audible and unbounded rather
# than merely wasteful. Short enough that a genuine second attempt straight
# after a false one is not swallowed.
EMPTY_TURN_REFRACTORY_S = 1.0

_YES = ("yes", "yeah", "yep", "sure", "confirm", "do it", "go ahead", "ok", "okay",
        "是", "是的", "对", "确定", "确认", "好", "好的", "可以", "没错")
_NO = ("no", "nope", "cancel", "stop", "don't", "never mind", "abort",
       "不", "不要", "取消", "别", "算了", "不用")


def _sounds(words: tuple[str, ...]) -> tuple[str, ...]:
    """The Chinese entries of a word list, as pinyin.

    Only the Chinese ones. An English word left as itself would be compared
    against pinyin, where it matches things it has no business matching — "no"
    is inside 弄 (`nong`), "ok" is inside 多克 (`duoke`).
    """
    return tuple(dict.fromkeys(
        s for s in (normalise(w) for w in words if detect_script(w) == "zh") if s))


_YES_SOUNDS = _sounds(_YES)
_NO_SOUNDS = _sounds(_NO)


def is_affirmative(text: str) -> bool | None:
    """Yes, no, or neither.

    Returns None rather than guessing when the answer is ambiguous — for an
    irreversible action, "I could not tell" has to mean "do not do it".

    Matches anywhere in the sentence, not just at the start: an answer often
    arrives with something in front of it ("嗯确定", "好的没问题"), and the
    wake word can be transcribed into it too. Negatives are tested first
    because they contain affirmatives — 不确定 is a refusal, and checking
    yes-words first would read it as agreement.

    **Chinese is compared by sound, not by character.** Whisper's Chinese
    output is not stable in script: the same speaker saying the same 确定 into
    the same microphone came back simplified three times and traditional
    (確定) three times, in one evening. Character matching therefore answered
    correctly about half the time, and since anything it fails to recognise is
    treated as a refusal, the visible symptom was the assistant saying 已取消
    to somebody who had just clearly said 确定 — twice in a row, to a shutdown
    it had itself asked about.

    Simplified and traditional are the same word to a listener and the same
    pinyin to `normalise`, which is exactly why the wake word and the intent
    router already compare this way. This was the one place still reading
    characters, and the one place where getting it wrong silently drops an
    action the user explicitly authorised.
    """
    stripped = text.strip().lower().rstrip(".!?。！？ ")
    sound = normalise(stripped) if detect_script(stripped) == "zh" else ""

    def said(words: tuple[str, ...], sounds: tuple[str, ...]) -> bool:
        return (any(word in stripped for word in words)
                or (bool(sound) and any(s in sound for s in sounds)))

    # Every negative, in either form, before any affirmative — 不確定 must not
    # be read as agreement whichever script it arrives in.
    if said(_NO, _NO_SOUNDS):
        return False
    if said(_YES, _YES_SOUNDS):
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

    # The rate is passed, not defaulted. `SpeechToText` writes it into the WAV
    # header, and a header that disagrees with the samples does not fail — it
    # transcribes a pitch-shifted signal and returns confident nonsense. Same
    # trap as the wake recogniser built at a hardcoded 16 kHz.
    stt = SpeechToText(cfg.stt, cfg.audio.target_rate)
    log.info("waiting for whisper-server at %s", cfg.stt.url)
    if not stt.wait_ready():
        log.error("whisper-server never answered. Start it: ./scripts/run_services.sh start")
        return 1

    speaker = Speaker(cfg.tts)
    speaker.warm()

    panel = Panel(enabled=os.environ.get("AIA_NO_PANEL") != "1")

    registry = Registry([KodamaLite(), System()])
    router = FastRouter(registry)
    # Language of the last thing heard, so "Listening…" appears in whichever
    # language the user is already speaking rather than defaulting to English.
    stt_language = cfg.stt.default_language
    for plugin in registry.plugins:
        log.info("plugin %s: %s", plugin.name,
                 "available" if plugin.available() else "NOT reachable right now")

    detector = wake_mod.build(cfg.wake, cfg.audio)
    endpointer = Endpointer(cfg.audio, cfg.vad)
    # Answering a question takes longer than starting a command — the user has
    # to hear it, understand it, and decide. The normal 4 s window would time
    # out on anyone who paused to think, and silence cancels.
    confirm_endpointer = Endpointer(
        cfg.audio, replace(cfg.vad, max_wait_ms=cfg.vad.confirm_wait_ms))
    ducker = Ducker()
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
        last_empty_turn = 0.0

        while not stopping:
            frame = next(frames)
            if not detector.detect(frame):
                continue
            if time.monotonic() - last_empty_turn < EMPTY_TURN_REFRACTORY_S:
                # The last turn heard nothing, so this is almost certainly the
                # same non-event firing again. See EMPTY_TURN_REFRACTORY_S.
                continue

            turn = machine.begin_turn()
            machine.to(State.LISTENING)

            # Silence the music FIRST — before the panel, before anything.
            # The microphone and the speakers share a room, so a command given
            # over music is captured as the command plus the song, and Whisper
            # transcribes the mixture. This is the difference between the
            # assistant working while music plays and not working at all.
            # Only drop buffered audio if there actually was music in it.
            # Those buffered frames are the pre-roll the endpointer uses, and
            # music in them reads as speech to the VAD — it would start the
            # utterance immediately, then endpoint on the silence that follows
            # the pause. But draining unconditionally would clip the opening
            # syllable of "小艾同学播放五月天" said in one breath, which works
            # today, so it is done only when it buys something.
            if ducker.duck():
                mic.drain()

            # Shown before anything is transcribed, because the question the
            # user actually has at this moment is "did it hear me at all?".
            panel.status(LISTENING_TEXT.get(stt_language, LISTENING_TEXT["en"]))

            audio = endpointer.collect(frames)
            turn.mark("captured")
            if audio is not None and save_audio:
                save_utterance(audio, cfg.audio.target_rate)
            if audio is None:
                # Nothing was said — a false wake, or the user changed their
                # mind. Put the music straight back and take the "Listening…"
                # line away rather than leaving it up implying otherwise.
                ducker.restore()
                panel.hide()
                detector.reset()
                machine.end_turn()
                last_empty_turn = time.monotonic()
                continue

            machine.to(State.THINKING)
            # Bound before the try because `finally` reads it, and `finally`
            # runs on every exit path — including the early `continue` for a
            # confirmation reply, and an exception raised before routing.
            intent = None
            try:
                # Detected fresh every turn, so the user can switch language
                # between one command and the next. See stt/engine.py.
                result = stt.listen(audio)
                turn.mark("stt")

                # The blank-audio marker is cleared in stt/engine.py, which is
                # the only place that knows it is a whisper.cpp artefact.
                text = result.text.strip()

                if not text:
                    # Apologise in the language of the last turn that *worked*,
                    # not in `result.language`. Nothing was understood, so this
                    # transcript has no language to report — it falls back to
                    # the configured default, which is English, and a Mandarin
                    # speaker who was misheard would be told "I'm sorry, could
                    # you repeat that?". The last language they were understood
                    # in is the best guess available, and this only picks a
                    # voice; it does not affect how anything is transcribed.
                    machine.to(State.SPEAKING)
                    apology = ("I'm sorry, could you repeat that?"
                               if stt_language == "en" else "抱歉，请再说一遍。")
                    panel.aia(apology)
                    speaker.say(apology, stt_language)
                    machine.end_turn()
                    detector.reset()
                    mic.drain()
                    continue

                lang = reply_language(text, fallback=result.language)
                stt_language = result.language
                # What it heard, before it acts on it — so a misrecognition is
                # visible rather than something to infer from a wrong action.
                panel.user(text)

                intent = router.match(text, lang)
                turn.mark("routed")

                if intent is not None and intent.command.confirm:
                    # Destructive: ask, then KEEP THE FLOOR and listen for the
                    # answer in this same turn. Asking and then returning to
                    # idle made the reply a separate request that needed the
                    # wake word again — so "确定" arrived as "小艾同学，确定",
                    # was not recognised as an answer, and the shutdown was
                    # silently dropped while the assistant replied to
                    # something else entirely. A question you have to be
                    # re-summoned to answer is not a question.
                    question = CONFIRM_PROMPT.get(lang, CONFIRM_PROMPT["en"]).format(
                        what=intent.command.describe(lang))
                    log.info("asking to confirm %s.%s",
                             intent.plugin.name, intent.command.name)
                    machine.to(State.SPEAKING)
                    panel.aia(question)
                    # Blocking: the answer must not be recorded over our own
                    # question, and the microphone is drained straight after.
                    speaker.say(question, lang, blocking=True)
                    mic.drain()

                    machine.to(State.LISTENING)
                    panel.status(CONFIRM_LISTEN.get(lang, CONFIRM_LISTEN["en"]))
                    answer_audio = confirm_endpointer.collect(frames)
                    decision = None
                    if answer_audio is not None:
                        # In the language the question was asked in: we are
                        # holding the floor, so there is nothing to detect.
                        answer = stt.listen(answer_audio, language=lang)
                        if answer.text.strip():
                            panel.user(answer.text)
                            decision = is_affirmative(answer.text)
                        log.info("confirmation answer %r -> %s", answer.text, decision)
                    else:
                        log.info("no answer to the confirmation")

                    if decision is True:
                        machine.to(State.ACTING)
                        reply = intent.command.handler(**intent.arguments).say(lang)
                        turn.mark("acted")
                    else:
                        # Silence, "no", or anything unclear all cancel. For an
                        # irreversible action "I could not tell" must mean no.
                        intent = None      # so the music comes back
                        reply = "Cancelled." if lang == "en" else "已取消。"
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
                # Bring the music back, but only once the reply has finished
                # speaking — resuming first would talk over the answer.
                #
                # Commands that meant to leave audio stopped opt out: resuming
                # after "暂停" would undo exactly what was asked for. Anything
                # that started playback itself is already handled, because
                # `restore` only resumes players it finds still paused.
                if intent is not None and intent.command.stops_playback:
                    ducker.forget()
                else:
                    ducker.restore()
                machine.end_turn()
                detector.reset()
                # Drop anything captured while we were talking, so the
                # assistant does not hear itself.
                mic.drain()

    detector.close()
    speaker.close()
    panel.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
