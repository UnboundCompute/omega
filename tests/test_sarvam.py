"""Sarvam speech-to-text with whisper behind it — DL-078.

No network and no ffmpeg: ``run`` writes the pieces ffmpeg would have, ``post``
answers as the API would, ``sleep`` records the pacing instead of waiting.
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any

import pytest

from omega import __main__ as cli
from omega import listen, sarvam

KEY = "sk_test_secret_9f8e7d"


@pytest.fixture(autouse=True)
def _ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sarvam, "_ffmpeg", lambda: "/fake/ffmpeg")


def _recording(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.m4a"
    path.write_bytes(b"not really audio")
    return path


def _cutter(pieces: int, *, wav_seconds: float | None = None, returncode: int = 0):
    """A fake ffmpeg run that writes ``pieces`` part files into the output dir."""
    calls: list[list[str]] = []

    class Done:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stderr = "boom: bad input" if returncode else ""
            self.stdout = ""

    def run(argv: list[str], **_: Any) -> Done:
        calls.append(argv)
        if returncode == 0:
            out = Path(argv[-1]).parent
            for i in range(pieces):
                part = out / f"part{i:04d}.wav"
                if wav_seconds is None:
                    part.write_bytes(f"piece-{i}".encode())
                else:
                    with wave.open(str(part), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(b"\0\0" * int(16000 * wav_seconds))
        return Done()

    run.calls = calls  # type: ignore[attr-defined]
    return run


class Api:
    """Answers each post with the next scripted ``(status, body)``."""

    def __init__(self, answers: list[tuple[int, Any]]) -> None:
        self.answers = list(answers)
        self.posts: list[tuple[str, dict[str, str], bytes]] = []

    def __call__(self, url: str, headers: Any, body: bytes) -> tuple[int, bytes]:
        self.posts.append((url, dict(headers), body))
        status, answer = self.answers.pop(0)
        raw = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
        return status, raw


def _ok(text: str) -> tuple[int, Any]:
    return 200, {"request_id": "r", "transcript": text, "language_code": "hi-IN"}


def test_pieces_are_posted_in_order_and_joined(tmp_path: Path) -> None:
    run = _cutter(3)
    api = Api([_ok("namaste sab"), _ok(""), _ok("let's begin")])
    slept: list[float] = []
    heard = sarvam.transcribe(
        _recording(tmp_path), key=KEY, run=run, post=api, sleep=slept.append
    )
    assert heard.text == "namaste sab let's begin"
    assert heard.model == "sarvam:saaras:v4"
    # Not WAV headers, so the length is honestly unknown, not 3 * 25.
    assert heard.duration is None
    assert [b"piece-0" in p[2] for p in api.posts] == [True, False, False]
    assert [b"piece-2" in p[2] for p in api.posts] == [False, False, True]
    order = [p[2].index(b"filename=\"part") for p in api.posts]
    names = [p[2][i + 10 : i + 22] for p, i in zip(api.posts, order)]
    assert names == [b"part0000.wav", b"part0001.wav", b"part0002.wav"]
    # Paced between posts, not before the first.
    assert slept == [sarvam.PACE_SECONDS, sarvam.PACE_SECONDS]
    argv = run.calls[0]
    assert argv[0] == "/fake/ffmpeg"
    for flag, value in (("-ac", "1"), ("-ar", "16000"), ("-segment_time", "25"),
                        ("-c:a", "pcm_s16le"), ("-f", "segment")):
        assert argv[argv.index(flag) + 1] == value


def test_header_and_fields(tmp_path: Path) -> None:
    api = Api([_ok("hello")])
    sarvam.transcribe(
        _recording(tmp_path), key=KEY, model="saarika:v2.5", language="hi-IN",
        run=_cutter(1), post=api, sleep=lambda s: None,
    )
    url, headers, body = api.posts[0]
    assert url == "https://api.sarvam.ai/speech-to-text"
    assert headers["api-subscription-key"] == KEY
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="model"\r\n\r\nsaarika:v2.5\r\n' in body
    assert b'name="language_code"\r\n\r\nhi-IN\r\n' in body
    assert b'name="file"; filename="part0000.wav"\r\nContent-Type: audio/wav' in body


def test_duration_is_read_from_the_pieces(tmp_path: Path) -> None:
    heard = sarvam.transcribe(
        _recording(tmp_path), key=KEY, run=_cutter(2, wav_seconds=2.5),
        post=Api([_ok("a"), _ok("b")]), sleep=lambda s: None,
    )
    assert heard.duration == pytest.approx(5.0)


@pytest.mark.parametrize(
    "answers",
    [
        [_ok("fine"), (500, {"error": {"message": f"bad key {KEY}"}})],
        [_ok("fine"), (403, b"forbidden")],
        [_ok("fine"), (200, b"<html>not json")],
        [_ok("fine"), (200, {"request_id": "r"})],
        [_ok("fine"), (429, {}), (429, {})],
    ],
    ids=["500", "403", "not-json", "no-transcript", "429-twice"],
)
def test_one_piece_failing_fails_the_whole_attempt(tmp_path: Path, answers: list) -> None:
    with pytest.raises(listen.NotHeard) as caught:
        sarvam.transcribe(
            _recording(tmp_path), key=KEY, run=_cutter(3), post=Api(answers),
            sleep=lambda s: None,
        )
    assert "piece 2 of 3" in str(caught.value)
    assert KEY not in str(caught.value)


def test_unreachable_api_is_not_heard(tmp_path: Path) -> None:
    def down(url: str, headers: Any, body: bytes) -> tuple[int, bytes]:
        raise OSError(f"connection refused while sending {headers['api-subscription-key']}")

    with pytest.raises(listen.NotHeard) as caught:
        sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(1), post=down,
                          sleep=lambda s: None)
    assert "could not reach sarvam" in str(caught.value)
    assert KEY not in str(caught.value)


def test_ffmpeg_failures_are_not_heard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = Api([])
    with pytest.raises(listen.NotHeard, match="ffmpeg failed: boom"):
        sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(1, returncode=1), post=api)
    with pytest.raises(listen.NotHeard, match="no pieces"):
        sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(0), post=api)
    monkeypatch.setattr(sarvam, "_ffmpeg", lambda: None)
    with pytest.raises(listen.NotHeard, match="ffmpeg is not installed"):
        sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(1), post=api)
    assert api.posts == []


def test_all_silent_is_not_heard(tmp_path: Path) -> None:
    with pytest.raises(listen.NotHeard, match="nothing"):
        sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(2),
                          post=Api([_ok(" "), _ok("")]), sleep=lambda s: None)


def test_429_is_retried_once_after_a_pause(tmp_path: Path) -> None:
    api = Api([(429, {"error": {"message": "rate limited"}}), _ok("got it")])
    slept: list[float] = []
    heard = sarvam.transcribe(_recording(tmp_path), key=KEY, run=_cutter(1), post=api,
                              sleep=slept.append)
    assert heard.text == "got it"
    assert len(api.posts) == 2
    assert slept == [sarvam.RETRY_SECONDS]


# --- the combinator ---------------------------------------------------------


def _whisper(calls: list[Path]):
    def hear(path: Path) -> listen.Heard:
        calls.append(path)
        return listen.Heard(text="from whisper", model="small")

    return hear


def test_fallback_uses_whisper_when_sarvam_fails(tmp_path: Path) -> None:
    lines: list[str] = []
    whispered: list[Path] = []

    def failing(path: Path, **_: Any) -> listen.Heard:
        raise listen.NotHeard("sarvam refused piece 1 of 1: HTTP 503")

    hear = sarvam.with_fallback(
        sarvam.Settings(key=KEY), log=lines.append, whisper=_whisper(whispered), sarvam=failing,
    )
    heard = hear(_recording(tmp_path))
    assert heard.text == "from whisper"
    assert whispered == [_recording(tmp_path)]
    assert lines == ["relay: sarvam failed (sarvam refused piece 1 of 1: HTTP 503); using whisper"]


def test_fallback_prefers_sarvam_and_passes_settings(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    whispered: list[Path] = []

    def ok(path: Path, **kw: Any) -> listen.Heard:
        seen.update(kw)
        return listen.Heard(text="from sarvam", model="sarvam:x")

    hear = sarvam.with_fallback(
        sarvam.Settings(key=KEY, model="m", language="en-IN"),
        whisper=_whisper(whispered), sarvam=ok,
    )
    assert hear(_recording(tmp_path)).text == "from sarvam"
    assert seen == {"key": KEY, "model": "m", "language": "en-IN"}
    assert whispered == []


def test_no_key_is_whisper_directly() -> None:
    whispered: list[Path] = []
    whisper = _whisper(whispered)
    lines: list[str] = []
    assert sarvam.with_fallback(None, log=lines.append, whisper=whisper) is whisper
    assert sarvam.with_fallback(None) is listen.transcribe
    assert lines == []


# --- configuration ----------------------------------------------------------


def test_settings_take_only_the_sarvam_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "SARVAM_API_KEY", "SARVAM_STT_MODEL", "SARVAM_STT_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-openai-must-not-load\n"
        f"export SARVAM_API_KEY='{KEY}'\n"
        "SARVAM_STT_LANGUAGE=hi-IN\n"
    )
    chosen = sarvam.settings(env)
    assert chosen == sarvam.Settings(key=KEY, model="saaras:v4", language="hi-IN")
    assert "OPENAI_API_KEY" not in os.environ
    assert "SARVAM_API_KEY" not in os.environ


def test_settings_environment_wins_and_no_key_is_none(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=from-file\nSARVAM_STT_MODEL=file-model\n")
    chosen = sarvam.settings(env, environ={"SARVAM_API_KEY": "from-shell"})
    assert chosen is not None and chosen.key == "from-shell" and chosen.model == "file-model"
    assert sarvam.settings(tmp_path / "missing.env", environ={}) is None
    assert sarvam.settings(None, environ={"SARVAM_API_KEY": "  "}) is None


def _relay_with(monkeypatch: pytest.MonkeyPatch, env_path: Path | None) -> tuple[dict, list[str]]:
    captured: dict[str, Any] = {}

    class FakeRelay:
        def __init__(self, address: Any, **kw: Any) -> None:
            captured.update(kw)

        def run(self, stopping: Any) -> None:  # pragma: no cover - wait is injected
            pass

    monkeypatch.setattr(cli.relaying, "Relay", FakeRelay)
    lines: list[str] = []
    assert cli.relay("core", 1, write=lines.append, wait=lambda e: None, env_path=env_path) == 0
    return captured, lines


def test_relay_wires_sarvam_when_a_key_is_filed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "SARVAM_API_KEY", "SARVAM_STT_MODEL", "SARVAM_STT_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY=sk-openai-must-not-load\nSARVAM_API_KEY={KEY}\n")
    captured, lines = _relay_with(monkeypatch, env)
    assert "omega relay transcribes with sarvam (saaras:v4), whisper as fallback" in lines
    assert not any(KEY in line for line in lines)
    assert captured["transcribe"] is not listen.transcribe
    # Whisper missing must not stop a pass that Sarvam can serve.
    assert captured["not_ready"]() is None
    assert "OPENAI_API_KEY" not in os.environ


def test_relay_without_a_key_is_whisper_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    captured, lines = _relay_with(monkeypatch, tmp_path / "missing.env")
    assert "omega relay transcribes with local whisper" in lines
    assert captured["transcribe"] is listen.transcribe
    assert captured["not_ready"] is listen.not_ready
