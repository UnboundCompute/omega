"""The third sense: a recording becomes a transcript becomes a review — DL-066.

**omega does not record.** The contract is a file appearing in a folder, and
that single choice is why this module is small. A meeting happens whether or not
the laptop is open, so capture cannot depend on omega running (DL-016 priced the
closing lid as initiative pausing, and capture may not pause). It also makes one
watcher the *union* of most transports: an Apple Shortcut, Voice Memos, a
hardware recorder, AirDrop and a future Telegram relay all terminate in the same
event, so which pipe the person uses stays theirs to change without a rewrite.
That is DL-005's channel-is-transport rule applied to a sense.

**Nothing here touches the store, and that is structural.** The log holds an
exclusive advisory lock for the life of the handle (DL-016's single-writer rule),
so a command that opens it cannot run while omega is running. Everything in this
file is therefore folder in, text out — testable with neither a log nor a key —
and the store-touching wrapper lives beside ``ingest()`` and ``digest()`` in the
executor.

**Transcription here is local and the model is pinned.** Local because the
premise is a private second brain, and a recording of a meeting is a recording
of other people: shipping the room's audio to a vendor is the one thing this
feature must not *quietly* do. It is no longer the only path: when the person
sets ``SARVAM_API_KEY``, the relay sends recordings to Sarvam first, because
they asked for it and it hears Hindi and English mixed (DL-078,
:mod:`omega.sarvam`), and this module becomes the fallback that runs whenever
Sarvam does not answer — so the local path still has to work on its own.
Pinned because the whisper CLI's own default is ``turbo``, which is **not**
cached — unpinned, the first meeting would trigger a silent multi-gigabyte
download at the moment of use. :func:`transcribe` refuses to run against an
uncached model rather than letting the network decide.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from omega import provider

__all__ = [
    "DEFAULT_FOLDER",
    "AUDIO_SUFFIXES",
    "WHISPER_MODEL",
    "SETTLE_SECONDS",
    "NotHeard",
    "NotReviewed",
    "Heard",
    "Review",
    "MAX_PER_PASS",
    "HEARD_DIRNAME",
    "not_ready",
    "file_away",
    "folder_path",
    "model_cache",
    "model_is_cached",
    "waiting",
    "whisper_binary",
    "transcribe",
    "review",
]

#: Where a recording is expected to land. In iCloud Drive because the *phone*
#: has to be able to write it, which is the whole of "not just mac" — and
#: iCloud specifically because it is the only sync root on this machine (no
#: Dropbox, no Drive, no OneDrive). Expanded late, never at import, so a test
#: can point at a tmpdir without touching the real one.
DEFAULT_FOLDER = "~/Library/Mobile Documents/com~apple~CloudDocs/omega-listen"

#: What counts as a recording. Deliberately a suffix allowlist and not a
#: content sniff: the folder is a place a person drops things, so a stray PDF or
#: a screenshot must be *ignored* rather than handed to ffmpeg to fail on. Wide
#: enough to cover what the plausible capture surfaces actually emit — Voice
#: Memos writes `.m4a`, Shortcuts can write `.m4a` or `.wav`, Telegram voice
#: notes are `.ogg`/`.opus`, cheap recorders write `.mp3` or `.wav`.
AUDIO_SUFFIXES = frozenset(
    {
        ".m4a",
        ".mp3",
        ".wav",
        ".aiff",
        ".aif",
        ".caf",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".mp4",
        ".m4b",
        ".webm",
    }
)

#: The whisper model, named rather than defaulted. See the module docstring:
#: the CLI's own default is `turbo` and it is not cached here.
WHISPER_MODEL = "base"

#: How long a file must sit unchanged before it is considered finished.
#:
#: Not paranoia. Two real races share one fix: a recorder still writing the
#: file, and iCloud still pulling it down. Transcribing a half-written file
#: produces a plausible transcript of the first half of a meeting, which is
#: worse than an error because nothing about it looks wrong.
SETTLE_SECONDS = 20.0

#: Recordings handled per idle pass. **One**, where a transcript pass takes
#: three and a usage pass two, and the difference is not timidity: those two
#: digests are mechanical and take milliseconds, while transcribing an hour of
#: audio takes tens of minutes on this machine. The executor is a single
#: consumer, so every recording in a pass is time omega cannot answer a
#: message — a cost that is bounded at one and would be unbounded at three.
MAX_PER_PASS = 1

#: Where a recording goes once it has been heard. A *subfolder*, and the file is
#: **moved** into it rather than deleted: the recording is the person's, the
#: only copy of a meeting may be sitting there, and nothing here is allowed to
#: be the thing that loses it. Moving is also what keeps the folder a queue —
#: :func:`waiting` lists one level, so a filed recording leaves the queue by
#: arriving somewhere else, and the person can always put it back.
HEARD_DIRNAME = "heard"

#: Transcription budget, as a multiple of the audio's own length plus a floor.
#: Measured on this machine: `base` on CPU runs ~1.8x realtime, so 8x leaves
#: room for a loaded machine and a bigger model without ever hanging forever.
TIMEOUT_FACTOR = 8.0
TIMEOUT_FLOOR = 120.0

#: Where whisper keeps its weights. Checked rather than assumed, because the
#: check is what turns a surprise download into a refusal.
_MODEL_CACHE = "~/.cache/whisper"

#: whisper installed as a user script rather than onto PATH is the normal case
#: on macOS, so `shutil.which` is tried first and these are the fallbacks.
_WHISPER_FALLBACKS = (
    "~/Library/Python/3.9/bin/whisper",
    "~/Library/Python/3.11/bin/whisper",
    "~/.local/bin/whisper",
)

#: Clipped before the review call. An hour of speech is ~60 KB of text, which
#: fits; a day-long recording does not, and a request that fails on length
#: would lose the whole meeting rather than most of it.
MAX_REVIEW_CHARS = 48_000


class NotHeard(RuntimeError):
    """Transcription did not produce text. The recording is kept regardless."""


class NotReviewed(RuntimeError):
    """The review model's answer could not be read as a review."""


@dataclass(frozen=True)
class Heard:
    """What a recording turned out to say."""

    text: str
    #: Seconds, when the transcript carried timestamps to derive it from.
    #: ``None`` rather than ``0.0`` for unknown: a duration of zero is a claim
    #: about a recording, and "I could not tell" is not that claim.
    duration: Optional[float] = None
    model: str = WHISPER_MODEL

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Review:
    """What the person reads afterwards (DL-066 #6).

    A *document*, not memory. Claims go through the learn path so omega
    remembers what it heard; this is the thing a person opens. Collapsing the
    two would file twenty sentences of one Tuesday as standing truths.
    """

    summary: str
    important: Sequence[str] = field(default_factory=tuple)
    highlights: Sequence[str] = field(default_factory=tuple)
    actions: Sequence[str] = field(default_factory=tuple)

    @property
    def empty(self) -> bool:
        """True when the model found nothing in any section.

        Exists so a caller can tell "reviewed, and there was nothing in it"
        from "not reviewed" — a distinction `CLAUDE.md` requires of any check
        that could otherwise pass on empty.
        """
        return not (
            self.summary.strip()
            or self.important
            or self.highlights
            or self.actions
        )

    def render(self, *, title: str = "") -> str:
        """The review as markdown, which is what gets stored and shown.

        Sections with nothing in them are **omitted**, not printed empty, for
        DL-041's reason: a heading that is usually blank teaches a reader to
        skip it, and by the time it matters they have learned to.
        """
        out: list[str] = []
        if title:
            out.append(f"# {title}")
        if self.summary.strip():
            out.append(self.summary.strip())
        for heading, items in (
            ("What matters", self.important),
            ("Highlights", self.highlights),
            ("To do", self.actions),
        ):
            if items:
                out.append(f"## {heading}")
                out.extend(f"- {item}" for item in items)
        return "\n\n".join(out).strip()


def folder_path(folder: Optional[str] = None) -> Path:
    """Resolve the watched folder. Expanded here, never at import."""
    return Path(os.path.expanduser(folder or DEFAULT_FOLDER))


def model_cache() -> Path:
    return Path(os.path.expanduser(_MODEL_CACHE))


def model_is_cached(model: str = WHISPER_MODEL) -> bool:
    """Whether the weights are already on disk.

    The whole point of this function is the refusal it enables. whisper
    downloads a missing model silently on first use, so without this check the
    first meeting a person records is the one that stalls on a multi-gigabyte
    fetch — and on a train, fails.
    """
    return (model_cache() / f"{model}.pt").is_file()


def whisper_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Find the whisper CLI, PATH first and then the usual user-script dirs."""
    if explicit:
        return explicit if Path(os.path.expanduser(explicit)).is_file() else None
    found = shutil.which("whisper")
    if found:
        return found
    for candidate in _WHISPER_FALLBACKS:
        path = Path(os.path.expanduser(candidate))
        if path.is_file():
            return str(path)
    return None


def not_ready(model: str = WHISPER_MODEL) -> Optional[str]:
    """Why listening cannot run at all right now, or ``None`` if it can.

    **The distinction this draws is load-bearing.** A receipt marks a recording
    heard forever, so writing one when the *environment* is not ready — whisper
    not installed, weights not fetched — would burn every recording in the
    folder on the first pass and transcribe none of them ever. That is not a
    hypothetical: it is precisely the state a new machine is in.

    So the caller checks this **once per pass** and does nothing at all when it
    answers, exactly as :meth:`Executor.ingest` writes no receipt when it fails
    to *look*: a failure to look is not a failure to learn from any particular
    thing. A failure that is genuinely about one recording — a corrupt file,
    audio whisper chokes on — still earns its receipt.
    """
    if whisper_binary() is None:
        return (
            "whisper is not installed, and it is the local transcriber "
            "(the only one without SARVAM_API_KEY)"
        )
    if not model_is_cached(model):
        return f"the whisper {model!r} model is not in {model_cache()}"
    return None


def file_away(path: Path, *, folder: Optional[str] = None) -> Optional[Path]:
    """Move a heard recording into the ``heard/`` subfolder. Never deletes.

    Returns where it went, or ``None`` if it could not be moved — which is not
    an error worth failing a pass over. The cursor, not the folder, is what
    stops a recording being heard twice (:attr:`Learned.heard`); this only
    stops the queue growing without bound, so the worst case of a failed move
    is that one file is hashed again on the next pass and skipped.

    A name collision is resolved by suffixing rather than by overwriting. Two
    recorders both producing ``audio.m4a`` is ordinary, and the one thing this
    must not do is quietly replace a meeting with a different meeting.
    """
    try:
        done = folder_path(folder) / HEARD_DIRNAME
        done.mkdir(parents=True, exist_ok=True)
        target = done / path.name
        if target.exists():
            stem, suffix = path.stem, path.suffix
            for n in range(2, 1000):
                candidate = done / f"{stem}-{n}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
            else:
                return None
        return Path(shutil.move(str(path), str(target)))
    except OSError:
        return None


def _is_placeholder(path: Path) -> bool:
    """An iCloud file that has not been downloaded yet.

    iCloud evicts a file's contents and leaves ``.name.ext.icloud`` — a small
    plist — in its place. Treating one as audio would hand whisper a few
    hundred bytes of XML; it is not a recording, it is a recording that is not
    here yet, and it will be a real file on a later pass.
    """
    return path.name.startswith(".") and path.name.endswith(".icloud")


def waiting(
    folder: Optional[str] = None,
    *,
    now: Optional[float] = None,
    settle: float = SETTLE_SECONDS,
) -> list[Path]:
    """Recordings in the folder that look finished, oldest first.

    Oldest first because a backlog should come out in the order it went in: a
    person who recorded three meetings expects the first one reviewed first.

    **Four things are skipped, each for its own reason.** A suffix that is not
    audio, because the folder is a place people drop things. A dotfile, because
    `.DS_Store` and friends live there. An iCloud placeholder, because the
    bytes have not arrived. And anything modified within ``settle`` seconds,
    because a file still being written or still syncing would transcribe as a
    confident account of half a meeting.
    """
    root = folder_path(folder)
    if not root.is_dir():
        return []
    clock = time.time() if now is None else now
    ready: list[tuple[float, Path]] = []
    for path in root.iterdir():
        if _is_placeholder(path):
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:
            # Vanished between listing and stat — a sync in flight, not an
            # error. It will be here or not on the next pass.
            continue
        if not path.is_file() or stat.st_size == 0:
            continue
        if clock - stat.st_mtime < settle:
            continue
        ready.append((stat.st_mtime, path))
    return [path for _, path in sorted(ready, key=lambda pair: (pair[0], pair[1].name))]


#: `[00:01:02.340 --> 00:01:05.120]` — whisper's own timestamp format.
_STAMP = re.compile(
    r"\[(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\]"
)


def _duration_from_stamps(text: str) -> Optional[float]:
    """The end of the last timestamped segment, in seconds, or ``None``.

    Read off the transcript rather than probed from the file, so a duration
    needs no second subprocess — and returns ``None`` rather than a guess when
    the format carried no stamps, because a wrong duration is a fact nobody
    checks.
    """
    last = None
    for match in _STAMP.finditer(text):
        hours, minutes, seconds, millis = match.group(5, 6, 7, 8)
        last = (
            int(hours or 0) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000.0
        )
    return last


def _strip_stamps(text: str) -> str:
    """Drop timestamps, keep the words, one line per segment."""
    lines = [
        _STAMP.sub("", line).strip() for line in text.splitlines() if line.strip()
    ]
    return "\n".join(line for line in lines if line)


def transcribe(
    path: Path,
    *,
    model: str = WHISPER_MODEL,
    binary: Optional[str] = None,
    language: Optional[str] = "en",
    run: Optional[Callable[..., Any]] = None,
) -> Heard:
    """Turn one recording into text with local whisper. Raises :class:`NotHeard`.

    ``run`` is the subprocess seam, so a test drives this without whisper
    installed and without waiting on a real transcription.

    The model is passed explicitly on every call. See the module docstring: the
    CLI's default is uncached and would download.
    """
    exe = whisper_binary(binary)
    if exe is None:
        raise NotHeard(
            "whisper is not installed, and it is the local transcriber "
            "(the only one without SARVAM_API_KEY)"
        )
    if not model_is_cached(model):
        # Refused rather than downloaded. A silent multi-gigabyte fetch at the
        # moment of use is the failure this check exists for.
        raise NotHeard(
            f"the whisper {model!r} model is not in {model_cache()}; refusing to "
            "download it mid-use — fetch it once deliberately instead"
        )
    runner = run or subprocess.run
    with tempfile.TemporaryDirectory(prefix="omega-listen-") as tmp:
        argv = [
            exe,
            str(path),
            "--model",
            model,
            "--output_format",
            "txt",
            "--output_dir",
            tmp,
        ]
        if language:
            # Named to skip detection, which costs a pass over the audio and
            # on a short or quiet recording guesses wrong.
            argv += ["--language", language]
        timeout = max(TIMEOUT_FLOOR, TIMEOUT_FACTOR * _probe_seconds(path))
        try:
            done = runner(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NotHeard(f"transcription timed out after {timeout:.0f}s") from exc
        except OSError as exc:
            raise NotHeard(f"could not run whisper: {exc}") from exc
        if getattr(done, "returncode", 1) != 0:
            tail = (getattr(done, "stderr", "") or "").strip().splitlines()
            raise NotHeard(
                "whisper failed: " + (tail[-1] if tail else "no error reported")
            )
        # whisper names the output after the input's stem, not after anything we
        # chose, so the file is found rather than constructed.
        produced = sorted(Path(tmp).glob("*.txt"))
        if not produced:
            raise NotHeard("whisper wrote no transcript")
        raw = produced[0].read_text(encoding="utf-8", errors="replace")

    # The stdout carries the timestamps; the txt file is already plain. Prefer
    # whichever actually has stamps, so a duration can be read off either.
    stamped = getattr(done, "stdout", "") or ""
    duration = _duration_from_stamps(stamped) or _duration_from_stamps(raw)
    text = _strip_stamps(raw).strip()
    if not text:
        raise NotHeard("the recording transcribed to nothing")
    return Heard(text=text, duration=duration, model=model)


def _probe_seconds(path: Path) -> float:
    """A cheap length estimate for the timeout, never for the record.

    Bytes over a pessimistic bitrate. Deliberately not ffprobe: this only sizes
    a timeout, and spawning a second process to protect the first one from
    hanging is a worse trade than overestimating.
    """
    try:
        return max(1.0, path.stat().st_size / 4_000.0)
    except OSError:
        return TIMEOUT_FLOOR


_REVIEW_SYSTEM = (
    "You are reading a transcript of something the person you work for "
    "recorded - usually a meeting, sometimes them thinking out loud. Produce "
    "the review they will read instead of the transcript.\n"
    "Answer with JSON only: {\"summary\": str, \"important\": [str], "
    "\"highlights\": [str], \"actions\": [str]}.\n"
    "`summary` is two or three sentences on what this was and what came of it. "
    "`important` is what actually matters - decisions reached, numbers, dates, "
    "names, anything that changes what happens next. `highlights` are the few "
    "moments worth re-reading, quoted or close to it. `actions` are things to "
    "do, each naming who owes it when the transcript says.\n"
    "This is a machine transcript of speech, so it has errors. Proper nouns are "
    "the least reliable part of it - do not silently 'correct' a name into a "
    "different one, and if a word that matters is clearly garbled say so rather "
    "than guessing confidently.\n"
    "Leave a list empty when the transcript does not support it. A meeting with "
    "no decisions and nothing to do is a real and common outcome, and padding "
    "the sections to look thorough is the one failure that makes the whole "
    "review untrustworthy."
)


def review(
    text: str,
    *,
    complete: Optional[Callable[..., provider.Response]] = None,
    title: str = "",
) -> Review:
    """Analyse a transcript into *what matters / highlights / to do*.

    Raises :class:`NotReviewed` on anything that is not a readable answer,
    rather than returning an empty review — an empty :class:`Review` is a real
    and meaningful result ("there was nothing in this meeting"), so a failure
    must never be able to present as one.
    """
    body = text.strip()
    if not body:
        raise NotReviewed("there is no transcript to review")
    call = complete or provider.complete
    head = f"Recording: {title}\n\n" if title else ""
    clipped = body[:MAX_REVIEW_CHARS]
    response = call(
        provider.LEARN,
        [
            provider.system(_REVIEW_SYSTEM),
            provider.user(f"{head}Transcript:\n{clipped}"),
        ],
    )
    return parse_review(response.text)


def _unfence(text: str) -> str:
    """Strip a ``` fence if the answer came wrapped in one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _string_list(value: Any) -> tuple[str, ...]:
    """Coerce one answer field to clean strings, dropping anything else.

    Lenient *within* a list and strict about the envelope: a model that returns
    a number among the actions has still answered in the format, while one that
    returns a string where a list belongs has not.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        # One item, unwrapped. Common enough to accept, and unambiguous.
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return tuple(out)


def parse_review(text: str) -> Review:
    """Strict parse of the review answer. Raises :class:`NotReviewed`.

    Strict about the envelope for :func:`omega.learn` reasons: hunting a JSON
    object out of prose would read a model that ignored the format as though it
    had followed it. ``summary`` must be present because the model is always
    told to return it; the three lists may be absent, because a transcript with
    no actions in it genuinely has none.
    """
    body = _unfence(text)
    if not body:
        raise NotReviewed("the review model returned no content")
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise NotReviewed(f"the answer was not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NotReviewed("the answer was not a JSON object")
    if "summary" not in parsed:
        raise NotReviewed('the answer had no "summary" field')
    summary = parsed["summary"]
    if not isinstance(summary, str):
        raise NotReviewed("summary must be a string")
    return Review(
        summary=summary.strip(),
        important=_string_list(parsed.get("important")),
        highlights=_string_list(parsed.get("highlights")),
        actions=_string_list(parsed.get("actions")),
    )
