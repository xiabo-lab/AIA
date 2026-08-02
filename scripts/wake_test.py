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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--trials", type=int, default=20)
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
    heard_counter: collections.Counter[str] = collections.Counter()

    with Microphone(cfg.audio) as mic:
        frames = mic.frames()
        trial = 0
        while trial < args.trials:
            trial += 1
            # Do NOT drain here. Draining between trials was the cause of
            # every miss in two consecutive runs: a person settles into a
            # rhythm and starts the next attempt while the previous result is
            # still being written, so the discarded audio is the *beginning*
            # of the phrase. 小爱同学 then arrives as 哎同学, 同学, or a
            # fragment like 跑 — front-truncated, 1.55 s against 2.41 s for a
            # clean capture. Moving the drain before the prompt narrowed the
            # window but did not close it, because the speech starts before
            # the prompt exists.
            #
            # Keeping the buffer means anything said early is still there when
            # collect() runs. The endpointer waits for speech onset regardless,
            # and the residue from the previous turn is the trailing silence it
            # already consumed.
            prefix = f"  [{trial:2}/{args.trials}] {GREEN}SPEAK{RESET} "
            print(prefix + "…", end="", flush=True)

            # Live view of what the microphone is picking up, so speaking can
            # be corrected in the moment rather than inferred from a verdict
            # afterwards. A separate recogniser from the detector's: this one
            # is transcribing for the human, and must not disturb the state
            # the detector is being judged on.
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
                print(f"{YELLOW}nothing heard — trying again{RESET}")
                trial -= 1  # a silent timeout is not an attempt
                continue

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
            score, window = detector._matches(text)[1:]

            results.append((hit, score, text, window))
            if window:
                heard_counter[window] += 1

            colour = GREEN if hit else RED
            print(f"{colour}{'HIT ' if hit else 'MISS'}{RESET} {score:.2f}  "
                  f"heard {text!r}")

            path = TRIALS_DIR / f"{trial:02d}-{'hit' if hit else 'miss'}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(cfg.audio.target_rate)
                w.writeframes(audio.tobytes())

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

    misses = [r for r in results if not r[0]]
    print(f"\n  {'=' * 62}")
    if not misses:
        print(f"  {GREEN}Every attempt matched.{RESET} Nothing to change.")
        return 0

    needed = max(r[1] for r in misses)
    print(f"  {len(misses)} missed. The closest missed attempt scored {needed:.2f}.")
    if needed >= 0.60:
        suggested = round(needed - 0.02, 2)
        print(f"\n  Lowering the threshold to {GREEN}{suggested}{RESET} catches all of them:")
        print(f"    aia/core/config.py -> WakeConfig.similarity = {suggested}")
        print(f"  {DIM}Weigh the false-alarm cost: the nearest ordinary phrase"
              f" measured 0.55.{RESET}")
    else:
        print("\n  Too far off to reach by threshold alone — the recogniser is")
        print("  hearing something genuinely different:")
        for form in dict.fromkeys(r[3] for r in misses if r[3]):
            print(f"    {form}")
        print(f"\n  Add the ones that are really you to {GREEN}WakeConfig.variants"
              f"{RESET} in aia/core/config.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
