"""Sarvam speech-to-text for recordings, local whisper behind it — DL-078.

**Sarvam first because it hears the person's languages.** Meetings here mix
Hindi and English, and Sarvam's ``saaras`` models are built for exactly that;
whisper stays as the fallback, so a Sarvam outage, a bad key or a rate limit
costs a slower transcript, never a lost one. This reverses part of DL-066's
"transcription is local": the person chose it, and the ledger records that
other people's voices now leave the Mac.

**Cut into pieces because the API takes at most 30 s a request.** ffmpeg turns
the recording into 25 s mono 16 kHz WAV pieces, each is posted in order, and
the transcripts are joined. Any piece failing fails the whole attempt: a
transcript with a silent hole in the middle reads as complete and is not,
which is worse than whisper's slower whole one.

**The key is only ever a header.** It never appears in a :class:`NotHeard`
reason, because that reason becomes a receipt in the log and a line on the
console. Reasons are scrubbed of it as a last guard, not as the plan.

**Only the ``SARVAM_*`` names are read from the ``.env``.** The relay holds no
model key (DL-072), and that still holds: :func:`settings` parses the file
without loading it, so ``OPENAI_API_KEY`` beside it never reaches the relay's
environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from omega import listen, provider

__all__ = [
    "URL",
    "DEFAULT_MODEL",
    "DEFAULT_LANGUAGE",
    "PIECE_SECONDS",
    "PACE_SECONDS",
    "Settings",
    "settings",
    "not_ready",
    "transcribe",
    "with_fallback",
]

URL = "https://api.sarvam.ai/speech-to-text"

#: The documented default and latest model (DL-078).
DEFAULT_MODEL = "saaras:v4"

#: Auto-detect, because a meeting here can switch language mid-sentence and a
#: named language would mishear the other half.
DEFAULT_LANGUAGE = "unknown"

#: Under the API's 30 s limit with room for ffmpeg's cut landing late.
PIECE_SECONDS = 25

#: At least this long between posts. The starter plan allows 60 requests a
#: minute, and an hour of audio is ~144 pieces; posting them back to back would
#: run into 429s by the second minute.
PACE_SECONDS = 1.1

#: How long to wait before the one retry a 429 gets.
RETRY_SECONDS = 10.0

#: Per request. A 25 s piece is a few hundred KB; a minute is generous.
TIMEOUT_SECONDS = 60.0

#: For the ffmpeg cut, which re-encodes but does not transcribe.
FFMPEG_TIMEOUT = 600.0

_FFMPEG_FALLBACK = "/opt/homebrew/bin/ffmpeg"

Post = Callable[[str, Mapping[str, str], bytes], tuple[int, bytes]]


@dataclass(frozen=True)
class Settings:
    key: str
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    #: ``codemix`` and the like (DL-079). ``None`` sends no ``mode`` field at
    #: all, so the API's own default applies.
    mode: Optional[str] = None


def settings(
    env_path: Optional[Path], *, environ: Optional[Mapping[str, str]] = None
) -> Optional[Settings]:
    """The Sarvam settings, or ``None`` when there is no key.

    The environment wins over the file, as in :func:`provider.load_env`. The
    file is *parsed*, never loaded: only the four ``SARVAM_*`` names are taken
    from it, so nothing else in it reaches ``os.environ``.
    """
    env = os.environ if environ is None else environ
    filed = provider.read_env(env_path) if env_path is not None else {}

    def pick(name: str) -> str:
        return (env.get(name) or filed.get(name) or "").strip()

    key = pick("SARVAM_API_KEY")
    if not key:
        return None
    return Settings(
        key=key,
        model=pick("SARVAM_STT_MODEL") or DEFAULT_MODEL,
        language=pick("SARVAM_STT_LANGUAGE") or DEFAULT_LANGUAGE,
        mode=pick("SARVAM_STT_MODE") or None,
    )


def not_ready() -> Optional[str]:
    """Why this machine cannot cut audio for Sarvam, or ``None`` when it can.

    About the machine, not a file: a caller checks it before taking a recording
    on, so a missing ffmpeg is never written down as one recording's failure.
    """
    return None if _ffmpeg() else "ffmpeg is not installed"


def _ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return _FFMPEG_FALLBACK if Path(_FFMPEG_FALLBACK).is_file() else None


def _post(url: str, headers: Mapping[str, str], body: bytes) -> tuple[int, bytes]:
    """The real HTTP call. An HTTP error is a status, not an exception."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""


def _multipart(
    piece: Path, model: str, language: str, mode: Optional[str] = None
) -> tuple[str, bytes]:
    boundary = "omega-" + uuid.uuid4().hex
    parts: list[bytes] = []
    fields = [("model", model), ("language_code", language)]
    if mode:
        # Only when set: an unset mode is the API's default, not an empty one.
        fields.append(("mode", mode))
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{piece.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + piece.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _why(status: int, raw: bytes) -> str:
    """A short reason for a refused piece: the status, and the API's message."""
    message = ""
    try:
        answer = json.loads(raw)
        error = answer.get("error") if isinstance(answer, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(answer, dict):
            message = str(answer.get("message") or answer.get("detail") or "")
    except (ValueError, UnicodeDecodeError):
        pass
    message = " ".join(message.split())[:120]
    return f"HTTP {status}" + (f": {message}" if message else "")


def _seconds(pieces: list[Path]) -> Optional[float]:
    """The recording's length, read from the pieces' own WAV headers.

    ``pieces * 25`` is only an upper bound — the last piece is short — so the
    frames are counted instead, and any piece that will not read makes the
    whole answer unknown rather than a guess.
    """
    total = 0.0
    try:
        for piece in pieces:
            with wave.open(str(piece), "rb") as wav:
                rate = wav.getframerate()
                if rate <= 0:
                    return None
                total += wav.getnframes() / rate
    except (wave.Error, EOFError, OSError):
        return None
    return total if total > 0 else None


def _transcribe(
    path: Path,
    *,
    key: str,
    model: str,
    language: str,
    mode: Optional[str],
    run: Callable[..., object],
    post: Post,
    sleep: Callable[[float], None],
) -> listen.Heard:
    exe = _ffmpeg()
    if exe is None:
        raise listen.NotHeard("ffmpeg is not installed")
    with tempfile.TemporaryDirectory(prefix="omega-sarvam-") as tmp:
        argv = [
            exe, "-nostdin", "-loglevel", "error", "-i", str(path),
            "-ac", "1", "-ar", "16000",
            "-f", "segment", "-segment_time", str(PIECE_SECONDS),
            "-c:a", "pcm_s16le",
            str(Path(tmp) / "part%04d.wav"),
        ]
        try:
            done = run(argv, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT, check=False)
        except subprocess.TimeoutExpired as exc:
            raise listen.NotHeard("ffmpeg timed out cutting the recording") from exc
        except OSError as exc:
            raise listen.NotHeard(f"could not run ffmpeg: {exc}") from exc
        if getattr(done, "returncode", 1) != 0:
            tail = (getattr(done, "stderr", "") or "").strip().splitlines()
            raise listen.NotHeard("ffmpeg failed: " + (tail[-1] if tail else "no error reported"))
        pieces = sorted(Path(tmp).glob("part*.wav"))
        if not pieces:
            raise listen.NotHeard("ffmpeg cut the recording into no pieces")

        said: list[str] = []
        for index, piece in enumerate(pieces):
            if index:
                sleep(PACE_SECONDS)
            content_type, body = _multipart(piece, model, language, mode)
            headers = {"api-subscription-key": key, "Content-Type": content_type}
            where = f"piece {index + 1} of {len(pieces)}"
            try:
                status, raw = post(URL, headers, body)
                if status == 429:
                    # Rate limited, not refused. Transcribing changes nothing
                    # on Sarvam's side, so asking again is safe; once, so a
                    # sustained limit falls through to whisper promptly.
                    sleep(RETRY_SECONDS)
                    status, raw = post(URL, headers, body)
            except (OSError, ValueError) as exc:
                raise listen.NotHeard(f"could not reach sarvam at {where}: {exc}") from exc
            if status != 200:
                raise listen.NotHeard(f"sarvam refused {where}: {_why(status, raw)}")
            try:
                answer = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise listen.NotHeard(f"sarvam answered {where} with something not JSON") from exc
            text = answer.get("transcript") if isinstance(answer, dict) else None
            if not isinstance(text, str):
                raise listen.NotHeard(f"sarvam's answer for {where} had no transcript")
            if text.strip():
                # A silent piece is an empty transcript, not a failure.
                said.append(text.strip())
        duration = _seconds(pieces)

    text = " ".join(said)
    if not text:
        raise listen.NotHeard("the recording transcribed to nothing")
    return listen.Heard(text=text, duration=duration, model=f"sarvam:{model}")


def transcribe(
    path: Path,
    *,
    key: str,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    mode: Optional[str] = None,
    run: Optional[Callable[..., object]] = None,
    post: Optional[Post] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> listen.Heard:
    """Transcribe one recording with Sarvam. Raises :class:`listen.NotHeard`.

    ``run`` (ffmpeg), ``post`` (HTTP, ``(url, headers, body) -> (status,
    bytes)``) and ``sleep`` (the pacing) are the seams, so a test needs neither
    ffmpeg nor the network nor the wall clock.
    """
    try:
        return _transcribe(
            path, key=key, model=model, language=language, mode=mode,
            run=run or subprocess.run, post=post or _post, sleep=sleep or time.sleep,
        )
    except listen.NotHeard as exc:
        reason = str(exc)
        if key and key in reason:
            raise listen.NotHeard(reason.replace(key, "[key]")) from None
        raise


def with_fallback(
    chosen: Optional[Settings],
    *,
    log: Callable[[str], None] = lambda line: None,
    whisper: Callable[[Path], listen.Heard] = listen.transcribe,
    sarvam: Callable[..., listen.Heard] = transcribe,
) -> Callable[[Path], listen.Heard]:
    """Sarvam, then whisper on any :class:`NotHeard`; whisper alone with no key.

    With no key this *is* whisper, so a relay without one behaves exactly as
    before and says nothing each pass about a service it was never told to use.
    """
    if chosen is None:
        return whisper

    def hear(path: Path) -> listen.Heard:
        try:
            return sarvam(
                path, key=chosen.key, model=chosen.model,
                language=chosen.language, mode=chosen.mode,
            )
        except listen.NotHeard as exc:
            log(f"relay: sarvam failed ({exc}); using whisper")
            return whisper(path)

    return hear
