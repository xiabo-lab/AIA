"""Measure a speech recogniser on this device, against real speech.

Three things, in the order you need them, all with `aia` stopped:

    .venv/bin/python scripts/stt_test.py record                # build the corpus
    .venv/bin/python scripts/stt_test.py run --backend both    # score both engines
    .venv/bin/python scripts/stt_test.py run --threads 1,2,3,4 # sweep threads

**`record` first, and it needs a person.** There is no Cantonese audio in this
project and none can be synthesised — a `yue` corpus is somebody sitting in
this room saying these phrases into this microphone. `record` walks the prompt
list, captures each one through the real `Endpointer`, and writes a manifest.
Everything after that is repeatable without a person present, which is the
whole point of doing it this way rather than measuring live turns.

**`run` reports what the spec asks for**: per-language accuracy, mean and p95
latency, RTF, CPU and RAM — for one backend or for both side by side, on the
same audio, in one process. Comparing two engines across two runs on two
recordings is how you get a number nobody can act on.

## What "accuracy" means here, and why there are two of them

**CER** is the character error rate against what the person was asked to say.
It is the honest ASR metric and it is the one to quote.

**Routed** is whether the transcript reaches a command in `FastRouter`. That is
the metric this project has always used, because it is the one the user
experiences: a transcript that is 90% right and does not route is a failed
turn, and one that is 80% right and routes is a working one. Whisper's Mandarin
errors are overwhelmingly homophone errors and the router matches by pinyin, so
the two numbers genuinely come apart. Quote both.

Routing is not this module's business to fix. A phrase that transcribes
perfectly and does not route is a finding about `router/fast.py`, and it is
reported as such rather than folded into the ASR score.

## CPU and RAM

Sampled from `/proc`, for the process that does the work — which is not the
same process in the two cases. SenseVoice runs in here; whisper.cpp runs in
whisper-server and this would otherwise measure an HTTP client sitting idle.
CPU is a percentage of ONE core, so 400% is the whole Pi 5.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aia.core.config import CONFIG  # noqa: E402
from aia.plugins.kodama import KodamaLite  # noqa: E402
from aia.plugins.base import Registry  # noqa: E402
from aia.plugins.system import System  # noqa: E402
from aia.router.fast import FastRouter  # noqa: E402
from aia.stt import build as build_stt  # noqa: E402

CORPUS = CONFIG.retention.recordings.parent / "stt-set"
MANIFEST = CORPUS / "manifest.json"


# The test set from the spec. `expect` is the command the phrase should reach
# when it is transcribed correctly; None means "transcribe it and report where
# it lands", which is the honest entry for a phrase whose routing has never
# been agreed on.
#
# Cantonese is written as it is *said*, not as the Mandarin equivalent — 搵 and
# 嘅 are the whole point. A recogniser that returns the Mandarin phrasing for
# these has not understood Cantonese, and scoring against a Mandarin reference
# would hide exactly that.
PROMPTS: tuple[tuple[str, str, str, str | None], ...] = (
    # (language, id, phrase, expected command)
    ("zh", "zh-search-lyrics", "搜索歌词", "search_lyrics"),
    # `search_song` takes a query, and a bare trigger with no song named is
    # deliberately not a complete command in `router/fast.py`. Verified against
    # a *perfect* transcript of this phrase, in both languages: it routes to
    # None by design. So it is carried here as an ASR item — the spec names it
    # and it must transcribe correctly — with no routing expectation, rather
    # than as a standing failure in the routed column that is not about STT.
    ("zh", "zh-search-song", "搜索歌曲", None),
    ("zh", "zh-play-jay", "播放周杰伦的歌", "play"),
    ("zh", "zh-pause", "暂停音乐", "pause"),
    ("zh", "zh-save-lyrics", "保存歌词", "save_lyrics"),

    ("yue", "yue-find-lyrics", "帮我搵下歌词", None),
    ("yue", "yue-play-eason", "播放陈奕迅嘅歌", "play"),
    ("yue", "yue-pause", "暂停音乐", "pause"),

    ("en", "en-search-lyrics", "search lyrics", "search_lyrics"),
    ("en", "en-search-song", "search for a song", None),   # see zh-search-song
    ("en", "en-play-music", "play some music", None),
    ("en", "en-pause", "pause the music", "pause"),
    ("en", "en-save-lyrics", "save the lyrics", "save_lyrics"),

    ("mixed", "mix-taylor", "搜索 Taylor Swift 的歌词", None),
    ("mixed", "mix-beyond", "播放 Beyond 的歌", "play"),
    ("mixed", "mix-jay", "search Jay Chou 的歌曲", None),
    ("mixed", "mix-dashboard", "open dashboard", None),
)

# Stripped before comparing. A recogniser with ITN on punctuates and one
# without it does not, and that difference is not an error anybody cares about
# — the router removes it too, for the same reason.
_PUNCT = re.compile(r"[\s,.!?;:'\"()\[\]，。！？；：、…—「」『』《》〈〉·]+")


def normalise(text: str) -> str:
    return _PUNCT.sub("", text.strip().lower())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, 0.0 perfect. Levenshtein over normalised text.

    Capped at 1.0. An unbounded CER — a recogniser that hallucinates a
    paragraph over a two-character command scores 8.0 — averages into a
    per-language figure that says nothing, and the failure it represents is
    already fully described by "got it wrong".
    """
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + (r != h)))
        previous = current
    return min(1.0, previous[-1] / len(ref))


def p95(values: list[float]) -> float:
    """95th percentile by nearest-rank, which is defined for small n.

    `statistics.quantiles` interpolates and needs at least two points; this set
    is 17 utterances and a per-language slice of it is three. Nearest-rank
    returns an observation that actually happened, which is the right kind of
    answer for a latency figure read off a handful of trials.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(-(-95 * len(ordered) // 100)))
    return ordered[rank - 1]


# ── resource sampling ────────────────────────────────────────────────

class Resources:
    """CPU and peak RSS for the processes that actually do the work.

    Reads /proc directly rather than depending on psutil, which is not in
    requirements.txt and would be a new dependency for a diagnostic.
    """

    def __init__(self, pids: list[int], interval: float = 0.1):
        self.pids = pids
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._cpu_start = self._cpu_total()
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def find(name: str) -> list[int]:
        """Pids whose comm matches, without shelling out to pgrep.

        `pgrep -f` matches its own invoking shell and has produced wrong
        answers in this project before; /proc/*/comm is the exact-name match
        and cannot.
        """
        proc = Path("/proc")
        if not proc.is_dir():   # not Linux; the caller falls back to this process
            return []
        found = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if (entry / "comm").read_text().strip() == name:
                    found.append(int(entry.name))
            except OSError:
                continue
        return found

    def _cpu_total(self) -> float:
        """Combined user+system jiffies across the tracked pids."""
        total = 0.0
        for pid in self.pids:
            try:
                fields = (Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1]).split()
                total += float(fields[11]) + float(fields[12])  # utime, stime
            except (OSError, IndexError):
                continue
        return total / 100.0  # USER_HZ is 100 on this kernel

    def _rss_mb(self) -> float:
        total = 0.0
        for pid in self.pids:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1]) / 1024.0
            except OSError:
                continue
        return total

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())

    def __enter__(self) -> Resources:
        self.peak_rss_mb = self._rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cpu_s = self._cpu_total() - self._cpu_start
        self.wall_s = time.monotonic() - self._t0

    @property
    def cpu_pct(self) -> float:
        """Percent of one core. The Pi 5 has four, so the ceiling is 400."""
        return 100.0 * self.cpu_s / max(self.wall_s, 1e-6)


# ── corpus ───────────────────────────────────────────────────────────

def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != CONFIG.audio.target_rate or w.getnchannels() != 1:
            raise SystemExit(
                f"{path} is {w.getframerate()} Hz / {w.getnchannels()} ch; "
                f"need {CONFIG.audio.target_rate} Hz mono"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CONFIG.audio.target_rate)
        w.writeframes(audio.tobytes())


def record(only: str | None) -> int:
    """Capture each prompt through the real capture and endpointing path.

    Deliberately the same `Microphone` and `Endpointer` the assistant uses,
    with only the wake word skipped. A corpus recorded through a different
    path measures a signal the assistant never sees — this project has already
    paid for that lesson once, with 27 trials captured through a broken
    decimator.

    Stop `aia` first: the microphone allows exactly one reader, and the second
    opener fails with an error that looks nothing like the cause.
    """
    from aia.audio.capture import Microphone
    from aia.audio.vad import Endpointer

    prompts = [p for p in PROMPTS if only in (None, p[0])]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}

    print(f"Recording {len(prompts)} prompts into {CORPUS}")
    print("Say each phrase after the prompt. Ctrl-C to stop.\n")

    endpointer = Endpointer(CONFIG.audio, CONFIG.vad)
    with Microphone(CONFIG.audio) as mic:
        frames = mic.frames()
        for lang, ident, phrase, expect in prompts:
            input(f"  [{lang}] {phrase}\n      press Enter, then speak… ")
            mic.drain()
            audio = endpointer.collect(frames)
            if audio is None:
                print("      nothing captured — skipped\n")
                continue
            path = CORPUS / lang / f"{ident}.wav"
            write_wav(path, audio)
            manifest[ident] = {
                "language": lang,
                "phrase": phrase,
                "expect": expect,
                "path": str(path.relative_to(CORPUS)),
                "duration_s": round(len(audio) / CONFIG.audio.target_rate, 2),
            }
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"      {len(audio) / CONFIG.audio.target_rate:.2f}s -> {path.name}\n")

    print(f"Manifest: {MANIFEST}")
    return 0


def load_corpus() -> list[dict]:
    if not MANIFEST.is_file():
        raise SystemExit(
            f"no corpus at {MANIFEST} — record one first:\n"
            f"    systemctl --user stop aia\n"
            f"    .venv/bin/python scripts/stt_test.py record"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = []
    for ident, entry in manifest.items():
        path = CORPUS / entry["path"]
        if not path.is_file():
            print(f"  ! {ident}: {path} is in the manifest and not on disk — skipping")
            continue
        entry = dict(entry, id=ident, audio=read_wav(path))
        items.append(entry)
    return items


# ── scoring ──────────────────────────────────────────────────────────

def score(backend_name: str, cfg, items: list[dict], router: FastRouter,
          repeat: int) -> dict:
    """Run the corpus through one backend and return everything measured."""
    stt = build_stt(cfg, CONFIG.audio.target_rate)
    if not stt.wait_ready(timeout=60):
        raise SystemExit(f"{backend_name}: backend is not usable — see the error above")

    # The pids to charge the work to. SenseVoice runs in this process;
    # whisper.cpp runs in a server this process only talks to, and measuring
    # the client would report an idle HTTP session as the cost of Whisper.
    pids = [os.getpid()]
    if cfg.backend.strip().lower() == "whisper":
        pids = Resources.find("whisper-server") or pids

    rows = []
    with Resources(pids) as res:
        for item in items:
            audio = item["audio"]
            duration_s = len(audio) / CONFIG.audio.target_rate
            latencies = []
            result = None
            for _ in range(repeat):
                t0 = time.monotonic()
                result = stt.listen(audio)
                latencies.append((time.monotonic() - t0) * 1000)
            # The median of the repeats, so one cold outlier does not become
            # the reported latency of a phrase.
            ms = statistics.median(latencies)
            intent = router.match(result.text)
            rows.append({
                "id": item["id"],
                "language": item["language"],
                "phrase": item["phrase"],
                "expect": item["expect"],
                "text": result.text,
                "detected": result.detected,
                "reply_language": result.language,
                "duration_s": duration_s,
                "ms": ms,
                "rtf": ms / max(duration_s * 1000, 1e-6),
                "cer": cer(item["phrase"], result.text),
                "exact": normalise(item["phrase"]) == normalise(result.text),
                "routed": intent.command.name if intent else None,
            })

    stt.close()
    return {
        "backend": backend_name,
        "rows": rows,
        "cpu_pct": res.cpu_pct,
        "peak_rss_mb": res.peak_rss_mb,
    }


def report(run: dict) -> None:
    rows = run["rows"]
    print(f"\n{'=' * 78}\n{run['backend']}\n{'=' * 78}")
    print(f"{'id':<18} {'CER':>5} {'ms':>7} {'RTF':>5} {'lang':>5}  transcript")
    print("-" * 78)
    for r in rows:
        flag = "ok " if r["exact"] else ("~  " if r["cer"] < 0.5 else "XX ")
        print(f"{flag}{r['id']:<15} {r['cer']:>5.2f} {r['ms']:>7.0f} {r['rtf']:>5.2f} "
              f"{str(r['detected'] or '-'):>5}  {r['text']!r}")
        if r["expect"] and r["routed"] != r["expect"]:
            print(f"{'':<18} {'':>5} {'':>7} {'':>5} {'':>5}  "
                  f"-> routed {r['routed']!r}, expected {r['expect']!r}")

    print(f"\n{'language':<10} {'n':>3} {'CER':>6} {'exact':>7} {'routed':>8} "
          f"{'mean ms':>8} {'p95 ms':>7} {'RTF':>6}")
    print("-" * 78)
    for lang in ("zh", "yue", "en", "mixed"):
        group = [r for r in rows if r["language"] == lang]
        if not group:
            continue
        _summary_line(lang, group)
    print("-" * 78)
    _summary_line("ALL", rows)

    print(f"\nCPU {run['cpu_pct']:.0f}% of one core   peak RSS {run['peak_rss_mb']:.0f} MB")


def _summary_line(label: str, group: list[dict]) -> None:
    lat = [r["ms"] for r in group]
    # Routing is only scored where the prompt names a command. A phrase with
    # no agreed command cannot be a routing failure, and counting it as one
    # would make the English rows look worse than they are.
    routable = [r for r in group if r["expect"]]
    hit = sum(1 for r in routable if r["routed"] == r["expect"])
    routed = f"{hit}/{len(routable)}" if routable else "-"
    print(f"{label:<10} {len(group):>3} "
          f"{statistics.mean(r['cer'] for r in group):>6.3f} "
          f"{sum(r['exact'] for r in group):>4}/{len(group):<2} "
          f"{routed:>8} "
          f"{statistics.mean(lat):>8.0f} {p95(lat):>7.0f} "
          f"{statistics.mean(r['rtf'] for r in group):>6.2f}")


def compare(runs: list[dict]) -> None:
    """Side by side, on the same audio, in the same process."""
    print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
    print(f"{'metric':<24} " + "".join(f"{r['backend']:>26}" for r in runs))
    print("-" * 78)

    def line(label, fn, fmt="{:.2f}"):
        def cell(run):
            value = fn(run)
            # A language with no recordings is "-", not "nan". This table is
            # read to make a decision about which engine to ship, and a
            # Cantonese row reading nan looks like a measurement that failed
            # rather than one that was never recorded.
            return "-" if value is None else fmt.format(value)
        print(f"{label:<24} " + "".join(f"{cell(r):>26}" for r in runs))

    for lang, name in (("zh", "Mandarin"), ("yue", "Cantonese"),
                       ("en", "English"), ("mixed", "Mixed")):
        def acc(run, lang=lang):
            group = [r for r in run["rows"] if r["language"] == lang]
            return statistics.mean(r["cer"] for r in group) if group else None
        line(f"{name} CER", acc)

    line("mean latency ms", lambda r: statistics.mean(x["ms"] for x in r["rows"]), "{:.0f}")
    line("p95 latency ms", lambda r: p95([x["ms"] for x in r["rows"]]), "{:.0f}")
    line("RTF", lambda r: statistics.mean(x["rtf"] for x in r["rows"]))
    line("CPU % of one core", lambda r: r["cpu_pct"], "{:.0f}")
    line("peak RSS MB", lambda r: r["peak_rss_mb"], "{:.0f}")


def selftest() -> int:
    """Prove the backend loads and behaves on audio, without needing a corpus.

    What it checks is deliberately narrow: that the model is present and
    loadable, that silence comes back empty rather than as a hallucinated
    phrase, and that a clip too short to hold a phoneme is refused rather than
    crashing. None of that is accuracy — it is the set of things that would
    otherwise fail at 7am in front of a person, having looked fine in a test.
    """
    cfg = CONFIG.stt
    print(f"backend: {cfg.backend}")
    stt = build_stt(cfg, CONFIG.audio.target_rate)

    t0 = time.monotonic()
    if not stt.wait_ready(timeout=60):
        print("FAIL: backend did not become ready")
        return 1
    print(f"  loaded and warm in {(time.monotonic() - t0) * 1000:.0f} ms")

    rate = CONFIG.audio.target_rate
    checks = [
        ("one second of silence", np.zeros(rate, dtype=np.int16), ""),
        ("empty array", np.zeros(0, dtype=np.int16), ""),
        ("10 ms, far too short", np.zeros(rate // 100, dtype=np.int16), ""),
        # Noise, not silence: a recogniser that returns text for this is
        # hallucinating, which is the failure that puts a wrong command
        # through the router with nobody having said anything.
        ("half a second of noise",
         (np.random.default_rng(0).normal(0, 800, rate // 2)).astype(np.int16), ""),
    ]
    ok = True
    for label, audio, expected in checks:
        result = stt.listen(audio)
        good = normalise(result.text) == normalise(expected)
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label:<26} -> {result.text!r}")

    stt.close()
    print("\nselftest passed" if ok else "\nselftest FAILED")
    return 0 if ok else 1


def main() -> int:
    # Every transcript this prints is Mandarin or Cantonese, and the report is
    # printed *after* all the measuring is done. An ssh session that arrives
    # with LANG=C gives stdout an ASCII codec, and the run then dies on the
    # first Han character with a UnicodeEncodeError — throwing away a complete
    # set of measurements at the last step, including the recorded-once
    # Cantonese ones. `errors="replace"` prints tofu instead, which is a
    # legible complaint rather than a lost run.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="run", choices=("run", "record"))
    ap.add_argument("--backend", default=None,
                    help="sensevoice | whisper | both (default: what config says)")
    ap.add_argument("--threads", default=None,
                    help="comma-separated SenseVoice thread counts to sweep, e.g. 1,2,3,4")
    ap.add_argument("--language", default=None, help="record only this language")
    ap.add_argument("--repeat", type=int, default=3,
                    help="passes per utterance; the median is reported")
    ap.add_argument("--json", type=Path, help="also write the raw rows here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mode == "record":
        return record(args.language)

    items = load_corpus()
    print(f"{len(items)} utterances, "
          f"{sum(len(i['audio']) for i in items) / CONFIG.audio.target_rate:.1f}s of audio")

    # Built once and shared, so both backends are scored by exactly the same
    # matcher. The plugins are not contacted — `match()` is pure text.
    router = FastRouter(Registry([KodamaLite(), System()]))

    # Without pypinyin the router compares Mandarin by character instead of by
    # sound, and homophone errors — which are most of what a Mandarin
    # recogniser gets wrong — stop matching. The whole routed column would then
    # be measuring a missing package rather than either engine. The router
    # warns at import; this says what it means for these numbers.
    try:
        import pypinyin  # noqa: F401
    except ImportError:
        print("\n  !! pypinyin is not installed: the 'routed' column will be "
              "far too low for zh/yue/mixed and is not comparable to earlier "
              "runs. CER is unaffected. Install it before quoting routing.\n")

    runs = []
    if args.threads:
        for n in [int(x) for x in args.threads.split(",")]:
            cfg = replace(CONFIG.stt, backend="sensevoice",
                          sensevoice=replace(CONFIG.stt.sensevoice, num_threads=n))
            runs.append(score(f"sensevoice x{n} threads", cfg, items, router, args.repeat))
    else:
        wanted = (args.backend or CONFIG.stt.backend).strip().lower()
        names = ["sensevoice", "whisper"] if wanted == "both" else [wanted]
        for name in names:
            cfg = replace(CONFIG.stt, backend=name)
            runs.append(score(name, cfg, items, router, args.repeat))

    for run in runs:
        report(run)
    if len(runs) > 1:
        compare(runs)

    if args.json:
        args.json.write_text(json.dumps(runs, ensure_ascii=False, indent=2,
                                        default=str), encoding="utf-8")
        print(f"\nraw rows -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
