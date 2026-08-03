"""Measure the wake word against your own voice, and tune it to match.

The wake word is the one component with no second chance: if it misses, the
user gets nothing at all and no explanation. This runs a fixed number of
trials, reports what the recogniser actually heard each time, and prints the
setting that would have caught whatever was missed.

    .venv/bin/python scripts/wake_test.py            # 20 trials
    .venv/bin/python scripts/wake_test.py -n 30

Stop AIA first; the microphone allows one reader:

    pkill -f aia.main

Every attempt is saved under .bench/wake-trials, so a disagreement can be
re-examined without having to say it again.

## Why there is a second phase

For a long time this measured only *misses*, and then offered to lower the
threshold whenever it saw one. That is half a measurement, and the missing
half was the expensive one: a wake word has two failure modes and they pull in
opposite directions, so a tool that can only see one of them will walk the
setting straight into the other. It did. Every trial was a deliberate
utterance of the phrase, so the corpus contained no negatives, and 同学 —
which scored 0.875 against the 爱同学 variant and fires on ordinary
conversation — went unnoticed across forty measured attempts.

So the second phase reads ordinary sentences that sound *near* the phrase
without being it, and the threshold advice is bounded by what they score. A
suggestion that would accept 同学 is not a suggestion worth printing.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aia.audio import wake as wake_mod  # noqa: E402
from aia.core.config import CONFIG  # noqa: E402

TRIALS_DIR = Path(__file__).resolve().parents[1] / ".bench" / "wake-trials"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# Sentences that are not the wake word but live next to it in sound. The first
# four all contain 同学, which is two of the three syllables of the 爱同学
# variant and one of the commonest words in the language; the rest are ordinary
# speech, there to show what the floor looks like. Read them aloud normally —
# the number that matters is the highest score any of them reaches, because
# that is the real lower bound on a safe threshold.
NEGATIVES = (
    "这位同学你好",
    "我的同学来了",
    "同学们早上好",
    "他是我的老同学",
    "今天天气怎么样",
    "帮我放一首歌",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--trials", type=int, default=20)
    ap.add_argument("--skip-negatives", action="store_true",
                    help="skip the false-alarm phase (the threshold advice is "
                         "then unsafe and will say so)")
    args = ap.parse_args()

    from vosk import KaldiRecognizer, SetLogLevel

    from aia.audio.capture import Microphone
    from aia.audio.vad import Endpointer

    SetLogLevel(-1)
    cfg = CONFIG
    threshold = cfg.wake.similarity

    print(f"\n  phrase    : {cfg.wake.phrase}")
    print(f"  threshold : {threshold:.2f} (pinyin similarity)")
    print(f"  trials    : {args.trials}\n")
    print("  loading the recogniser…", end="", flush=True)

    # Built ONCE. An earlier version constructed this inside the loop, which
    # reloaded the 66 MB model on every trial — the best part of a second of
    # dead time during which anything spoken was captured and then thrown
    # away by the next drain. It presented as the wake word needing to be
    # said two or three times, which is exactly the bug this tool exists to
    # find, so it is worth naming: the test must not be slower than the thing
    # it measures.
    detector = wake_mod.VoskWakeWord(cfg.wake, cfg.audio.target_rate)
    model = detector._model  # reuse; a second Model() would load it all again
    endpointer = Endpointer(cfg.audio, cfg.vad)
    print(" ready\n")

    print("  Say the wake word each time it says SPEAK. Say it as you normally")
    print("  would — the point is to match how you actually talk to it.")
    print(f"\n  While you speak you will see: {GREEN}●{RESET}/{DIM}○{RESET} voice detected,"
          f" a level meter, and the live transcript.")
    print(f"  {DIM}If the level bar barely moves, move closer or speak up. If the"
          f" transcript\n  drops the first syllable, start a moment after the"
          f" prompt.{RESET}\n")

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[bool, float, str, str]] = []
    negatives: list[tuple[bool, float, str, str]] = []
    heard_counter: collections.Counter[str] = collections.Counter()

    def attempt(prefix):
        """Record one utterance and put it through the real detector.

        Returns (hit, score, text, window, audio), or None if nothing was said.

        Do NOT drain before this. Draining between trials was the cause of
        every miss in two consecutive runs: a person settles into a rhythm and
        starts the next attempt while the previous result is still being
        written, so the discarded audio is the *beginning* of the phrase.
        小爱同学 then arrives as 哎同学, 同学, or a fragment like 跑 —
        front-truncated, 1.55 s against 2.41 s for a clean capture. Moving the
        drain before the prompt narrowed the window but did not close it,
        because the speech starts before the prompt exists.

        Keeping the buffer means anything said early is still there when
        collect() runs. The endpointer waits for speech onset regardless, and
        the residue from the previous turn is the trailing silence it already
        consumed.
        """
        print(prefix + "…", end="", flush=True)

        # Live view of what the microphone is picking up, so speaking can be
        # corrected in the moment rather than inferred from a verdict
        # afterwards. A separate recogniser from the detector's: this one is
        # transcribing for the human, and must not disturb the state the
        # detector is being judged on.
        live = KaldiRecognizer(model, cfg.audio.target_rate)

        def show(frame, speech, started, _prefix=prefix, _rec=live):
            _rec.AcceptWaveform(frame.tobytes())
            partial = json.loads(_rec.PartialResult()).get("partial", "")
            level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            bars = min(12, int(level / 120))
            meter = ("█" * bars).ljust(12, "·")
            mark = GREEN + "●" + RESET if speech else DIM + "○" + RESET
            # \033[K clears the rest of the line so a shorter partial does
            # not leave the tail of a longer one behind.
            sys.stdout.write(f"\r{_prefix}{mark} {DIM}{meter}{RESET} "
                             f"{partial or '…'}\033[K")
            sys.stdout.flush()

        audio = endpointer.collect(frames, on_frame=show)
        sys.stdout.write(f"\r{prefix}\033[K")
        if audio is None:
            return None

        # Replay through the real detector, frame by frame, exactly as the
        # live loop feeds it. Reimplementing the match here would let the
        # test and the assistant disagree, which is the one thing a
        # wake-word test must not do.
        detector.reset()
        n = cfg.audio.frame_samples
        hit = False
        for i in range(0, len(audio) - n + 1, n):
            if detector.detect(audio[i:i + n]):
                hit = True
                break

        rec = KaldiRecognizer(model, cfg.audio.target_rate)
        rec.AcceptWaveform(audio.tobytes())
        text = json.loads(rec.FinalResult()).get("text", "").replace(" ", "")
        # Reset before scoring: the detector caches its last verdict per
        # scanned text, and this must be a fresh judgement of the whole
        # utterance rather than whatever the replay above left behind.
        detector.reset()
        score, window = detector._matches(text)[1:]
        return hit, score, text, window, audio

    def save(name, audio):
        with wave.open(str(TRIALS_DIR / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(cfg.audio.target_rate)
            w.writeframes(audio.tobytes())

    with Microphone(cfg.audio) as mic:
        frames = mic.frames()
        trial = 0
        while trial < args.trials:
            trial += 1
            outcome = attempt(f"  [{trial:2}/{args.trials}] {GREEN}SPEAK{RESET} ")
            if outcome is None:
                print(f"{YELLOW}nothing heard — trying again{RESET}")
                trial -= 1  # a silent timeout is not an attempt
                continue
            hit, score, text, window, audio = outcome

            results.append((hit, score, text, window))
            if window:
                heard_counter[window] += 1

            colour = GREEN if hit else RED
            print(f"{colour}{'HIT ' if hit else 'MISS'}{RESET} {score:.2f}  "
                  f"heard {text!r}")
            save(f"{trial:02d}-{'hit' if hit else 'miss'}.wav", audio)

        # ── false alarms ──────────────────────────────────────────────
        # The half of the measurement that was missing. See the module
        # docstring: without it the report below is free to recommend a
        # threshold that fires on ordinary conversation, which is what it used
        # to do.
        if not args.skip_negatives:
            print(f"\n  {'=' * 62}")
            print("  Now the false-alarm phase. These are NOT the wake word —")
            print("  read each one aloud, normally. Nothing should match.\n")
            for index, phrase in enumerate(NEGATIVES, 1):
                prefix = (f"  [{index:2}/{len(NEGATIVES)}] {YELLOW}SAY{RESET} "
                          f"{phrase}  ")
                outcome = attempt(prefix)
                if outcome is None:
                    print(f"{DIM}nothing heard — skipped{RESET}")
                    continue
                fired, score, text, window, audio = outcome
                negatives.append((fired, score, phrase, text))
                colour = RED if fired else GREEN
                verdict = "FALSE ALARM" if fired else "ok         "
                print(f"{colour}{verdict}{RESET} {score:.2f}  heard {text!r}")
                save(f"neg-{index:02d}-{'fired' if fired else 'ok'}.wav", audio)

    # ── report ────────────────────────────────────────────────────────
    print(f"\n  {'=' * 62}")
    if not results:
        print("  Nothing was recorded — is the microphone free? (pkill -f aia.main)")
        return 1

    hits = [r for r in results if r[0]]
    rate = len(hits) / len(results) * 100
    colour = GREEN if rate >= 90 else YELLOW if rate >= 70 else RED
    print(f"  detection rate: {colour}{len(hits)}/{len(results)}  ({rate:.0f}%){RESET}"
          f"   at threshold {threshold:.2f}")

    scores = sorted(r[1] for r in results)
    print(f"  scores        : min {scores[0]:.2f}  median "
          f"{scores[len(scores) // 2]:.2f}  max {scores[-1]:.2f}")

    print("\n  what the recogniser heard:")
    for form, count in heard_counter.most_common(8):
        missed = any(r[3] == form and not r[0] for r in results)
        print(f"    {count:2}x  {form}{f'  {RED}(missed){RESET}' if missed else ''}")

    # The ceiling on any threshold worth suggesting. Anything at or below the
    # loudest false alarm is a setting that fires on ordinary speech.
    floor = 0.0
    if negatives:
        fired = [n for n in negatives if n[0]]
        worst = max(n[1] for n in negatives)
        floor = worst
        colour = RED if fired else GREEN
        print(f"\n  false alarms   : {colour}{len(fired)}/{len(negatives)}{RESET}"
              f"   worst ordinary phrase scored {worst:.2f}")
        # A score of 0.00 here does not mean the phrase sounded nothing like
        # the wake word — it usually means it sounded a great deal like it and
        # was refused anyway, because the first syllable was never heard. That
        # is a rejection no threshold can undo, which is the whole point of
        # reporting the gated score rather than the raw similarity.
        for hit, score, phrase, text in sorted(negatives, key=lambda n: -n[1])[:4]:
            mark = f"{RED}fired{RESET}" if hit else f"{DIM}ok   {RESET}"
            print(f"    {score:.2f}  {mark}  {phrase}  {DIM}heard {text!r}{RESET}")
        if fired:
            print(f"\n  {RED}The wake word fires on ordinary speech.{RESET} Raising the")
            print(f"  threshold above {worst:.2f} is the minimum; if that costs too many")
            print("  hits, the variants in WakeConfig are too short to be safe —")
            print("  a three-syllable variant cannot tolerate a dropped syllable")
            print("  without also accepting its own two-syllable tail.")

    misses = [r for r in results if not r[0]]
    print(f"\n  {'=' * 62}")
    if not misses:
        print(f"  {GREEN}Every attempt matched.{RESET} Nothing to change.")
        return 0

    needed = max(r[1] for r in misses)
    print(f"  {len(misses)} missed. The closest missed attempt scored {needed:.2f}.")

    suggested = round(needed - 0.02, 2)
    if needed < 0.60:
        print("\n  Too far off to reach by threshold alone — the recogniser is")
        print("  hearing something genuinely different:")
        for form in dict.fromkeys(r[3] for r in misses if r[3]):
            print(f"    {form}")
        print(f"\n  Add the ones that are really you to {GREEN}WakeConfig.variants"
              f"{RESET} in aia/core/config.py.")
    elif not negatives:
        # No negatives measured, so there is no evidence about what lowering
        # the threshold would let in. Refusing to name a number is the honest
        # answer; this tool has recommended one on no evidence before.
        print(f"\n  Lowering the threshold to {suggested} would catch them, but this")
        print(f"  run measured no false alarms, so {YELLOW}there is nothing to say"
              f" whether{RESET}")
        print(f"  {YELLOW}that is safe{RESET}. Re-run without --skip-negatives.")
    elif suggested <= floor:
        print(f"\n  {RED}Not safe to lower.{RESET} Catching them needs {suggested},"
              f" but ordinary")
        print(f"  speech already scores {floor:.2f} — that setting would wake the")
        print("  assistant on conversation. These misses need a new variant in")
        print(f"  {GREEN}WakeConfig.variants{RESET}, or a better microphone position:")
        for form in dict.fromkeys(r[3] for r in misses if r[3]):
            print(f"    {form}")
    else:
        print(f"\n  Lowering the threshold to {GREEN}{suggested}{RESET} catches all of them:")
        print(f"    aia/core/config.py -> WakeConfig.similarity = {suggested}")
        print(f"  {DIM}Measured headroom: the nearest ordinary phrase scored"
              f" {floor:.2f}.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
