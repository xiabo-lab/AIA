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
    AIA_NO_PANEL=1    no on-screen overlay
    AIA_NO_WEB=1      no web UI (it is otherwise on http://127.0.0.1:8090)

The conversation and the saved recordings are both kept for 24 hours and then
deleted; see aia/ui/retention.py.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import replace
import time
import wave

from aia.audio import wake as wake_mod
from aia.audio.capture import Microphone
from aia.audio.ducking import Ducker
from aia.audio.vad import Endpointer
from aia.core.config import CONFIG, RetentionConfig
from aia.core.info import SystemInfo
from aia.core.state import Machine, State
from aia.plugins.base import Registry, Result
from aia.plugins.kodama import KodamaLite
from aia.plugins.system import System
from aia.router.fast import FastRouter, normalise
from aia.stt import build as build_stt
from aia.stt import detect_script
from aia.tts.language import reply_language
from aia.tts.piper import Speaker
from aia.ui.history import ConversationLog
from aia.ui.panel import Panel
from aia.ui.retention import Retention
from aia.ui.server import WebUI

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


def save_utterance(audio, rate: int, keep: int, retention: RetentionConfig,
                   suffix: str = "") -> None:
    """Write the captured utterance to .bench/utterances for offline replay.

    Every accuracy number so far came from synthesised speech, which is clean,
    close-miked and free of room reverb — it flatters every engine. Real
    captures are the only honest input for comparing recognisers, so keeping
    them saves having to reproduce a misrecognition live.

    Bounded, because this runs on every turn for the life of the machine and
    the destination is an SD card. Measured after four days: 101 files, 15 MB,
    an average of **148 KB** each — the comment this replaces claimed "a few
    KB", which was out by a factor of thirty, and nothing pruned them. Left
    alone that is about 1.4 GB and 9,000 files a year, written to the same
    flash that tts/piper.py deliberately routes around because a
    write-per-utterance is "unkind to the card".

    Oldest go first. A misrecognition is diagnosed within days or not at all,
    and the newest are the ones anybody asks about.

    This is the count cap only. Recordings also expire on the clock, which is
    the rule that actually bounds how long anything said in this room is kept —
    see `aia/ui/retention.py`. The two agree on a directory because both read
    it from `RetentionConfig` rather than each working one out.

    `suffix` marks a capture the endpointer *rejected*, which is saved here and
    not somewhere separate on purpose. Both prunes glob `*.wav` in this one
    directory, so a reject inherits the 24-hour expiry and the count cap
    without a second set of rules — a failed turn is still someone's voice in
    their kitchen, and it must not outlive the ones that worked. The timestamp
    stays at the front of the name because both prunes sort by name and mean
    age when they do.
    """
    directory = retention.recordings
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    path = directory / f"{stamp}{'-' + suffix if suffix else ''}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
    log.info("saved utterance to %s", path)

    # A floor under the floor. Pruning deletes real recordings that cannot be
    # made again — the same speaker, room and microphone on a particular day —
    # so a small `keep` is far more likely to be a mistake than an intention.
    # This is not hypothetical: a test called this with keep=3 against the
    # live directory and took 101 captures down to 3, and the numbers derived
    # from them survive only because they were written into commit messages.
    #
    # The floor is `retention.keep_recordings`, the same number the expiry
    # sweep refuses to delete below, so there is one answer to "how many
    # captures are guaranteed to survive" rather than two that can drift.
    floor = retention.keep_recordings
    if keep < floor:
        log.warning("keep=%d is below the %d-file floor; keeping %d instead",
                    keep, floor, floor)
        keep = floor

    # Names are timestamps, so sorting by name is sorting by age.
    existing = sorted(directory.glob("*.wav"))
    for stale in existing[:max(0, len(existing) - keep)]:
        try:
            stale.unlink()
        except OSError as exc:  # a file someone is reading, or a read-only card
            log.debug("could not prune %s: %s", stale, exc)


def main() -> int:
    setup_logging()
    cfg = CONFIG
    started = time.time()
    save_audio = os.environ.get("AIA_SAVE_AUDIO") == "1"

    # Started before anything that can be slow, so a restart also expires
    # whatever the last session left behind — a machine that has been off for a
    # week comes back with an empty transcript rather than a week-old one, and
    # the sweep has happened before the first turn can add to it.
    history = ConversationLog(cfg.retention.database)
    retention = Retention(cfg.retention, history)
    retention.start()

    # Which recogniser this is depends on `stt.backend`; nothing below this
    # line knows or cares. The rate is passed, not defaulted — both backends
    # need it and both have a way of being silently wrong without it, see
    # aia/stt/__init__.py.
    stt = build_stt(cfg.stt, cfg.audio.target_rate)
    log.info("starting stt backend %r", cfg.stt.backend)
    if not stt.wait_ready():
        # SenseVoice loads in-process, so this is a missing model or a missing
        # wheel; whisper is a separate service, so it is one that never came
        # up. Both have already logged the specific reason.
        log.error("stt backend %r is not usable — see the error above",
                  cfg.stt.backend)
        return 1
    log.info("stt ready: %s", stt.name)

    speaker = Speaker(cfg.tts)
    speaker.warm()

    panel = Panel(enabled=os.environ.get("AIA_NO_PANEL") != "1", history=history)

    registry = Registry([KodamaLite(), System()])
    # The wake forms are handed to the router as sentence boundaries, not for
    # detection — a transcript containing one in the middle is two requests.
    router = FastRouter(registry, wake_words=cfg.wake.variants)
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

    web = WebUI(
        cfg.ui,
        history=history,
        # Built here but handed the microphone below, once there is one: the
        # settings page has to report the capsule the open stream is actually
        # reading from, and before `Microphone` exists the honest answer is
        # "not open".
        info=SystemInfo(cfg, speaker=speaker, stt=stt, started=started),
        state=lambda: machine.state.value,
        retention_hours=cfg.retention.hours,
    )

    # Cleanup belongs on every exit path, not just the tidy one. These own
    # real resources — the Vosk recogniser, both resident Piper processes, a
    # listening socket and a database connection — and an exception escaping
    # the loop used to skip all of them. systemd papers over it by killing the
    # whole cgroup; `python -m aia.main` by hand, which the README documents,
    # does not.
    try:
        with Microphone(cfg.audio) as mic:
            web.info.mic = mic
            web.start()
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
                    # ...but keep the newest `preroll_ms`, which is the part
                    # that is not music: it is the syllable the user is saying
                    # right now. The music is in the older audio, and that is
                    # what this is here to throw away.
                    mic.drain(keep_ms=cfg.vad.preroll_ms)

                # Shown before anything is transcribed, because the question the
                # user actually has at this moment is "did it hear me at all?".
                panel.status(LISTENING_TEXT.get(stt_language, LISTENING_TEXT["en"]))

                audio = endpointer.collect(frames)
                turn.mark("captured")
                if audio is not None and save_audio:
                    save_utterance(audio, cfg.audio.target_rate,
                                   cfg.audio.keep_utterances, cfg.retention)
                elif audio is None and save_audio and endpointer.rejected is not None:
                    # The turn the user is complaining about. Saved under the
                    # reason it failed, so "it did not hear me" can be answered
                    # by listening to what it did hear.
                    save_utterance(endpointer.rejected, cfg.audio.target_rate,
                                   cfg.audio.keep_utterances, cfg.retention,
                                   suffix=endpointer.reject_reason or "rejected")
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

                    # Empty here means "nothing usable was said" and nothing
                    # else. Every backend-specific marker for that —
                    # whisper.cpp's "[BLANK_AUDIO]", a clip too short to have
                    # a phoneme in it — is cleared inside the backend, which is
                    # the only place that knows what its own artefacts look
                    # like. Callers used to strip them here, which was too late:
                    # "[BLANK_AUDIO]" is ten Latin letters, so a silent
                    # Mandarin turn got answered in English.
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
                        panel.aia(apology, stt_language)
                        speaker.say(apology, stt_language)
                        machine.end_turn()
                        detector.reset()
                        mic.drain()
                        continue

                    lang = reply_language(text, fallback=result.language)
                    stt_language = result.language
                    # What it heard, before it acts on it — so a misrecognition is
                    # visible rather than something to infer from a wrong action.
                    panel.user(text, lang)

                    # One utterance can hold more than one command. `intent`
                    # stays the *last* of them, because everything downstream
                    # — the reply, whether it is spoken, whether the music is
                    # restored — is a question about how the turn ended.
                    chain = router.match_sequence(text)
                    intent = chain[-1] if chain else None
                    turn.mark("routed")

                    # Silence is the default; each branch below opts in. See
                    # `CommandSpec.speaks` for why round it that way.
                    speak = False

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
                        panel.aia(question, lang)
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
                                panel.user(answer.text, lang)
                                decision = is_affirmative(answer.text)
                            log.info("confirmation answer %r -> %s", answer.text, decision)
                        else:
                            log.info("no answer to the confirmation")

                        # Anything that got as far as being asked about out loud
                        # is answered out loud, whichever way it went. The
                        # question held the floor and the room is waiting on it;
                        # going silent after "关闭树莓派。确定吗？" would leave
                        # the most consequential moment in the whole interaction
                        # as the only one with no audible outcome.
                        speak = True
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
                        # Asked of every command in the chain before any of
                        # them runs, so a two-command utterance cannot half
                        # succeed against an app that is closed.
                        missing = next(
                            (s for s in chain if not s.plugin.available()), None)
                        if missing is not None:
                            reply = Result.failed(
                                f"{missing.plugin.description} is not currently running.",
                                f"{missing.plugin.description} 没有在运行。",
                            ).say(lang)
                            # A command that did not run is not a command whose
                            # result can be seen. Nothing changed on screen, so
                            # this is spoken even when the command itself is a
                            # quiet one — otherwise asking a closed app to skip
                            # a track is indistinguishable from being ignored.
                            speak = True
                        else:
                            # Every command in the utterance, in order. Only
                            # the last one's reply is kept and only its flag
                            # decides speech — a chain that announced each
                            # step would be slower and more talkative than
                            # the two separate turns it replaced.
                            for step in chain:
                                outcome = step.command.handler(**step.arguments)
                                reply = outcome.say(lang)
                            speak = intent.command.speaks
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
                    # than after it. The panel gets every reply, spoken or not —
                    # it is the channel that never goes quiet.
                    panel.aia(reply, lang)
                    # Mark when audio STARTS, not when it finishes. Timing a
                    # blocking play charges the whole spoken reply to the turn —
                    # a two-second sentence looked like two seconds of latency,
                    # which made a responsive turn read as 4 s over budget. What
                    # the user experiences is the wait before hearing anything.
                    #
                    # Marked either way, so a silent turn and a spoken one stay
                    # comparable in the journal rather than the field simply
                    # vanishing from half of them.
                    if speak:
                        speaker.say(reply, lang, blocking=False)
                        turn.mark("audio_out")
                        speaker.wait()
                    else:
                        turn.mark("audio_out")
                        log.info("reply not spoken (%s): %r",
                                 f"{intent.plugin.name}.{intent.command.name}"
                                 if intent is not None else "no command", reply)

                except Exception:
                    log.exception("turn failed")
                    machine.to(State.ERROR)
                    # Say so. A turn that dies here used to end in silence:
                    # the wake word was acknowledged, the music ducked, "Listening…"
                    # stayed on screen, and then nothing ever came back. From the
                    # outside that is indistinguishable from the assistant having
                    # ignored you, and the only record was in the journal.
                    #
                    # Guarded, because this runs on the path where something has
                    # already gone wrong: if speaking fails too, the original
                    # exception is what matters and it has already been logged.
                    try:
                        trouble = ("Sorry, something went wrong." if stt_language == "en"
                                   else "抱歉，出错了。")
                        panel.aia(trouble, stt_language)
                        speaker.say(trouble, stt_language)
                    except Exception:
                        log.exception("could not announce the failed turn")
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

    finally:
        detector.close()
        # Holds an ONNX graph on the SenseVoice backend, and an HTTP session
        # on the whisper one. Neither survives the process, but this list is
        # the project's record of what owns a real resource and it should stay
        # complete.
        stt.close()
        speaker.close()
        panel.close()
        web.stop()
        retention.stop()
        # Last, so anything the final turn queued is written before the
        # database connection goes.
        history.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
