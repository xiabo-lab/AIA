"""Where the version number comes from.

There is deliberately no version constant in this project to bump. Releases are
git tags (`v0.3.1`) with the explanation written into the tag annotation, and
`.github/workflows/release.yml` publishes those notes from the tag object — so a
hardcoded string here would be a second place to edit and the one nobody edits
is the one the UI would show.

The complication is that **the Pi has no `.git`**. It is deployed with
`git archive <tag> aia scripts | ssh …`, not by checking out, so `git describe`
cannot run there. What does work is `export-subst`: `git archive` rewrites the
placeholder below at archive time, so the deployed tree arrives carrying the tag
it was cut from. That requires the line in `.gitattributes`:

    aia/_version.py export-subst

In a working checkout the placeholder stays literal — nothing substitutes it —
which is exactly how `_archived()` tells the two situations apart, and there
`git describe` is available and more informative anyway (it reports commits
past the tag, and a dirty tree).
"""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Rewritten by `git archive` into something like "v0.3.1" or "v0.3.1-4-gabc1234".
# Left alone anywhere else. Do not "tidy" this into a normal string.
_SUBSTITUTED = "$Format:%(describe:tags)$"

UNKNOWN = "unknown"


def _archived() -> str | None:
    """The tag `git archive` stamped in, or None in a working checkout."""
    if _SUBSTITUTED.startswith("$Format:"):
        return None
    # `%(describe)` expands to an empty string when the commit has no tag in
    # its history — an archive of an untagged branch. Empty is not a version.
    return _SUBSTITUTED.strip() or None


def _described() -> str | None:
    """`git describe` against this checkout, or None if that is not possible."""
    try:
        proc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git describe failed: %s", exc)
        return None
    if proc.returncode != 0:
        log.debug("git describe exited %d: %s", proc.returncode, proc.stderr.strip())
        return None
    return proc.stdout.strip() or None


@functools.lru_cache(maxsize=1)
def version() -> str:
    """The version this code was deployed as.

    Cached: on the Pi this is a constant, and in a checkout it shells out to
    git, which is not something to do once per HTTP request.
    """
    return _archived() or _described() or UNKNOWN


@functools.lru_cache(maxsize=1)
def source() -> str:
    """How the version above was arrived at. Shown in the UI beside it.

    Worth surfacing, because "unknown" has two very different causes — a tree
    that was copied rather than archived, and an archive of an untagged commit
    — and they need different fixes.
    """
    if _archived():
        return "git archive"
    if _described():
        return "git describe"
    return "no tag and no git checkout"
