"""Run a wav file through the full pipeline and report the latency breakdown.

The live loop needs somebody to speak into the microphone, which makes it
useless as a regression test. This drives every stage after capture —
endpointing, transcription, language selection, synthesis — from a recording
instead, so the fast-path budget can be checked in CI or over SSH.

    .venv/bin/python scripts/replay.py .bench/en2s.wav [--play] [--repeat N]

Synthesis is measured but NOT played unless --play is given, so this is safe to
run on a device somebody is sitting next to.

What it does not cover: the microphone itself and the wake word. Both are
measured separately — see scripts/bench_m0.sh and the notes in docs/PLAN.md.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aia.core.config import CONFIG  # noqa: E402
from aia.stt import build as build_stt  # noqa: E402
from aia.tts.language import reply_language  # noqa: E402
from aia.tts.piper import Speaker  # noqa: E402

# Stages that happen before this harness can see them, taken from the design
# constants in docs/PLAN.md so the reported total is comparable to the budget.
WAKE_MS = 150
VAD_MS = CONFIG.vad.silence_ms


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(
                f"{path} is {w.getframerate()} Hz / {w.getnchannels()} ch; "
                "need 16 kHz mono (convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+", type=Path)
    ap.add_argument("--play", action="store_true", help="actually play the reply")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)-7s %(name)-18s %(message)s",
    )

    stt = build_stt(CONFIG.stt, CONFIG.audio.target_rate)
    if not stt.wait_ready(timeout=30):
        print(f"stt backend {CONFIG.stt.backend!r} is not usable — see the error above.")
        print("  sensevoice: ./scripts/get_sensevoice.sh")
        print("  whisper:    ./scripts/run_services.sh start")
        return 1

    speaker = Speaker(CONFIG.tts)
    speaker.warm()

    overall_ok = True
    for path in args.wav:
        audio = load_wav(path)
        print(f"\n=== {path.name}  ({len(audio) / 16000:.2f}s of audio) ===")

        totals: list[float] = []
        for i in range(args.repeat):
            t0 = time.monotonic()
            result = stt.listen(audio)
            t_stt = (time.monotonic() - t0) * 1000

            lang = reply_language(result.text, fallback=CONFIG.stt.default_language)
            reply = (f"You said: {result.text}" if lang == "en" else f"你说：{result.text}")

            t1 = time.monotonic()
            speaker.say(reply, lang, blocking=args.play) if args.play else \
                speaker._voices[lang].synth(reply)
            t_tts = (time.monotonic() - t1) * 1000

            total = WAKE_MS + VAD_MS + t_stt + t_tts
            totals.append(total)
            if i == 0:
                print(f"  heard [{result.language}]: {result.text!r}")
                print(f"  reply [{lang}]: {reply!r}")

        median = statistics.median(totals)
        budget = CONFIG.target_latency_ms
        ok = median <= budget
        overall_ok &= ok
        print(f"  wake {WAKE_MS} + vad {VAD_MS} + stt {statistics.median(totals) - WAKE_MS - VAD_MS - t_tts:.0f}"
              f" + tts {t_tts:.0f}")
        print(f"  TOTAL median {median:.0f} ms of {budget} ms budget"
              f"  {'PASS' if ok else 'FAIL'}")

    speaker.close()
    stt.close()
    print("\n" + ("All within budget." if overall_ok else "Over budget — see above."))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
