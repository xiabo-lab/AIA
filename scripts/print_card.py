"""Generate a printable command reference: commands.html.

Same source as commands.txt — `plugin.manifest()` — so the printed card cannot
disagree with what the router actually matches. Plain text lost the table
alignment once phrase lists got long; HTML keeps the columns and prints
properly from any browser.

    .venv/bin/python scripts/print_card.py      # -> commands.html, then Ctrl+P

Styled for paper rather than screen: black on white, no filled backgrounds to
drain a cartridge, and page-break rules so a command never splits across two
sheets.
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aia.core.config import CONFIG  # noqa: E402
from aia.plugins.base import Registry  # noqa: E402
from aia.plugins.kodama import KodamaLite  # noqa: E402
from aia.plugins.system import System  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "commands.html"

# Which commands belong in which printed section, in the order a person is
# likely to want them. Anything not listed falls into "More" rather than being
# dropped — a new command must never vanish from the card just because this
# table was not updated.
SECTIONS = [
    ("Playback", ["resume", "pause", "next", "previous", "stop", "toggle"]),
    ("Ask for something", ["play", "search", "search_song", "volume"]),
    ("Modes and information", ["now_playing", "shuffle", "repeat", "like",
                               "lyrics", "karaoke"]),
    # Their own section, and printed adjacent, because the card is also where
    # someone works out which of the two lyric commands they want. "搜索歌词"
    # and "搜索歌曲" differ by one syllable and do unrelated things; seeing
    # them apart on the page is worth more than tidy grouping.
    ("Lyrics on the karaoke screen", ["search_lyrics", "save_lyrics"]),
    ("Device — these ask before acting", ["quit", "shutdown", "reboot"]),
]

CSS = """
@page { size: A4 portrait; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body {
  font-family: "Noto Sans", "DejaVu Sans", "Segoe UI", system-ui, sans-serif;
  color: #000; background: #fff;
  font-size: 9.5pt; line-height: 1.35; margin: 0;
}
h1 { font-size: 17pt; margin: 0 0 2mm; letter-spacing: -.2pt; }
.wake {
  border: 1.2pt solid #000; border-radius: 2mm;
  padding: 2.5mm 4mm; margin: 0 0 4mm;
  display: flex; align-items: baseline; gap: 4mm;
}
.wake .label { font-size: 8pt; text-transform: uppercase; letter-spacing: .6pt; }
.wake .phrase { font-size: 20pt; font-weight: 700; }
.wake .hint { font-size: 8.5pt; color: #333; }
h2 {
  font-size: 10.5pt; margin: 4.5mm 0 1.5mm;
  border-bottom: 1pt solid #000; padding-bottom: 1mm;
}
table { width: 100%; border-collapse: collapse; }
tr { page-break-inside: avoid; }          /* never split a command across pages */
td { vertical-align: top; padding: 1.4mm 2mm 1.4mm 0; border-bottom: .3pt solid #bbb; }
td.does { width: 21%; font-weight: 600; }
td.en   { width: 40%; }
td.zh   { width: 39%; }
.alt { color: #444; }
.arg { font-style: italic; }
.note { font-size: 8.5pt; color: #222; margin: 1.5mm 0 0; }
.spk { font-weight: 400; font-size: 7.5pt; }
.foot {
  margin-top: 5mm; padding-top: 2.5mm; border-top: 1pt solid #000;
  font-size: 8.5pt; page-break-inside: avoid;
}
.foot b { font-family: inherit; }
.yesno { display: flex; gap: 8mm; margin: 1.5mm 0; }
@media screen {
  body { max-width: 200mm; margin: 8mm auto; padding: 0 6mm; }
}
"""


def phrases_html(command, code: str) -> str:
    """First phrase bold as the one to learn, the rest as alternatives."""
    items = list(command.phrases.get(code, ()))
    if not items:
        return "<span class='alt'>—</span>"

    def show(text: str) -> str:
        text = html.escape(text)
        # {query} -> the argument, rendered so it reads as a placeholder.
        return text.replace("{query}", "<span class='arg'>…</span>") \
                   .replace("{level}", "<span class='arg'>N</span>")

    head = f"<b>{show(items[0])}</b>"
    if len(items) == 1:
        return head
    rest = " · ".join(show(p) for p in items[1:])
    return f"{head}<br><span class='alt'>{rest}</span>"


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    registry = Registry([KodamaLite(), System()])
    by_name = {c.name: (p, c) for p in registry.plugins for c in p.commands()}

    placed = {name for _, names in SECTIONS for name in names}
    leftovers = [n for n in by_name if n not in placed]
    sections = SECTIONS + ([("More", leftovers)] if leftovers else [])

    rows: list[str] = []
    for title, names in sections:
        rows.append(f"<h2>{html.escape(title)}</h2><table>")
        for name in names:
            if name not in by_name:
                continue
            _, command = by_name[name]
            label = html.escape(command.description)
            if command.params:
                label += " <span class='arg'>…</span>"
            # Marked, because "did it hear me?" is the question a silent
            # assistant provokes, and the card is what somebody has in their
            # hand when they ask it.
            if command.speaks:
                label += "<sup class='spk'>†</sup>"
            rows.append(
                f"<tr><td class='does'>{label}</td>"
                f"<td class='en'>{phrases_html(command, 'en')}</td>"
                f"<td class='zh'>{phrases_html(command, 'zh')}</td></tr>"
            )
        rows.append("</table>")

    wake = html.escape(CONFIG.wake.phrase)
    total = len(by_name)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AIA voice commands</title><style>{CSS}</style></head><body>
<h1>AIA — voice commands</h1>
<div class="wake">
  <span class="label">Say</span>
  <span class="phrase">{wake}</span>
  <span class="hint">then the command — in one breath ({wake}播放五月天)
    or with a short pause after it.</span>
</div>
{''.join(rows)}
<p class="note"><b>…</b> stands for whatever you want — a song, an artist, a
search. It runs to the end of the sentence. <b>N</b> is a number 0–100;
“fifty percent”, “五十” and “百分之五十” all work.</p>
<p class="note">Phrases are matched by <b>sound, not spelling</b>, so they do
not have to be said exactly. Anything not recognised as a command is shown on
screen rather than read back.</p>
<p class="note">Two commands in one breath work if you join them — “下一首
<b>and</b> 现在播放什么”, “暂停<b>然后</b>下一首”. Anything that asks first has
to be said on its own.</p>
<p class="note"><b>†</b> AIA answers these out loud. Everything else it simply
does — you can see or hear the result already — so it acts without saying
anything. A command that asks first always speaks.</p>
<div class="foot">
  <b>Commands that ask first</b> say what they are about to do and keep
  listening — you do not need the wake word again to answer.
  <div class="yesno">
    <span><b>Yes</b> — 确定 · 是 · 好 · 好的 · 可以 · yes · sure · ok</span>
    <span><b>No</b> — 取消 · 不要 · 别 · 算了 · no · cancel</span>
  </div>
  Silence, or anything unclear, cancels.
  <div style="margin-top:2mm;color:#444">{total} commands ·
    generated by scripts/print_card.py · do not edit by hand</div>
</div>
</body></html>
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT}")
    print("open it in a browser and print (Ctrl+P). A4 portrait, no backgrounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
