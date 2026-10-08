"""Hearing what the person recorded — DL-066.

Five groups.

The **folder** cases are refusals, and they protect a transcript's honesty
rather than its privacy. A recording still being written, and an iCloud
placeholder whose bytes have not arrived, both transcribe into a fluent account
of half a meeting — which is worse than an error, because nothing about the
result looks wrong. Everything in the group is a reason to leave a file alone.

The **model** cases pin the one refusal that costs money if it is missing. The
whisper CLI's own default model is not cached on this machine, so an unpinned
call downloads gigabytes at the moment of use; :func:`transcribe` is required to
refuse instead, and to refuse without reaching the network to find out.

The **transcript** cases drive the subprocess through its injected seam, so the
suite needs neither whisper nor minutes of CPU. They pin what comes back out:
timestamps read for a duration and then stripped, and every way a failed run
becomes :class:`NotHeard` rather than an empty transcript nobody notices.

The **review** cases are the strict-parse group. An empty review is a real and
common outcome — a meeting with no decisions and nothing to do — so the thing
that must not happen is a *failure* presenting as one, and every malformed
answer here has to raise instead.

The **end-to-end** cases are the pair `CLAUDE.md` asks for. The capability: a
recording files what it taught, stores its review, and leaves a receipt. The
violations, which must not regress: a machine with no whisper installed writes
**no receipts at all** (every receipt is permanent, so a pass that failed for
environmental reasons would mark the whole folder heard and transcribe none of
it ever), and the same recording arriving twice under a different name is heard
once — because the cursor is keyed on the digest of the bytes and not the name.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest

from omega import blobs, episodes, learn, listen, provider
from omega.executor import Executor
from omega.queue import EventQueue

#: A whisper run's stdout, in the format the real CLI emits.
STAMPED = (
    "[00:00:00.000 --> 00:00:03.120]  Quick standup.\n"
    "[00:00:03.120 --> 00:00:08.760]  We agreed not to move the ship date.\n"
)


# --- doubles -----------------------------------------------------------------


class _Run:
    """A stand-in for ``subprocess.run`` that records what it was asked."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = STAMPED,
        txt: Optional[str] = STAMPED,
        raises: Optional[BaseException] = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.txt = txt
        self.raises = raises
        self.argv: list[str] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001
        self.argv = list(argv)
        if self.raises is not None:
            raise self.raises
        if self.txt is not None:
            # whisper writes beside its `--output_dir`, named after the input.
            out = Path(argv[argv.index("--output_dir") + 1])
            (out / "input.txt").write_text(self.txt, encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "boom")


def _installed(monkeypatch: pytest.MonkeyPatch, *, cached: bool = True) -> None:
    """Pretend whisper is present without requiring it to be."""
    monkeypatch.setattr(listen, "whisper_binary", lambda explicit=None: "/bin/whisper")
    monkeypatch.setattr(listen, "model_is_cached", lambda model=listen.WHISPER_MODEL: cached)


def _recording(folder: Path, name: str = "standup.m4a", body: bytes = b"audio") -> Path:
    path = folder / name
    path.write_bytes(body)
    return path


def _settled(folder: Path) -> list[Path]:
    """Everything in ``folder`` that would be offered once it has settled."""
    return listen.waiting(str(folder), now=1e12)


# --- the folder ---------------------------------------------------------------


def test_a_missing_folder_is_empty_not_an_error(tmp_path: Path) -> None:
    """Nothing to hear is the ordinary state of a folder nobody has used yet.

    Raising here would make the idle pass noisy on every machine where the
    person has not set this up, which is every machine on the first run.
    """
    assert listen.waiting(str(tmp_path / "nope")) == []


def test_only_audio_is_offered(tmp_path: Path) -> None:
    """The folder is a place a person drops things, so a stray file is ignored.

    An allowlist and not a content sniff: handing a PDF to whisper to find out
    would turn a screenshot somebody saved into a failure receipt.
    """
    _recording(tmp_path, "meeting.m4a")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF")
    (tmp_path / "shot.png").write_bytes(b"\x89PNG")
    (tmp_path / ".DS_Store").write_bytes(b"junk")

    assert [p.name for p in _settled(tmp_path)] == ["meeting.m4a"]


def test_an_icloud_placeholder_is_not_a_recording(tmp_path: Path) -> None:
    """The sharpest case in the group, and invisible without it.

    iCloud evicts a file's contents and leaves a few hundred bytes of plist
    named ``.thing.m4a.icloud``. It is not a recording that failed; it is a
    recording that has not arrived, and it will be a real file on a later pass.
    Transcribing the placeholder would mark the meeting heard — permanently —
    having heard an XML header.
    """
    (tmp_path / ".meeting.m4a.icloud").write_bytes(b"<plist></plist>")

    assert _settled(tmp_path) == []


def test_a_file_still_being_written_is_left_alone(tmp_path: Path) -> None:
    """The other half of the same danger, from the other direction.

    A recorder still writing, or iCloud still pulling bytes down, yields a
    confident transcript of the first half of a meeting. The only safe signal
    available to a folder watcher is that the file stopped changing.
    """
    _recording(tmp_path)

    assert listen.waiting(str(tmp_path)) == []  # just touched
    assert len(listen.waiting(str(tmp_path), now=1e12)) == 1


def test_an_empty_file_is_never_offered(tmp_path: Path) -> None:
    """Zero bytes is a sync in flight, not a recording — and the episode
    refuses to describe one, so offering it would only produce a crash."""
    (tmp_path / "empty.m4a").write_bytes(b"")

    assert _settled(tmp_path) == []


def test_recordings_come_out_oldest_first(tmp_path: Path) -> None:
    """A backlog should come out in the order it went in: somebody who recorded
    three meetings expects the first one reviewed first."""
    import os

    for n, name in enumerate(["third.m4a", "first.m4a", "second.m4a"]):
        p = _recording(tmp_path, name)
        os.utime(p, (1000 + n, 1000 + n))
    # Written third, second, first by mtime -- so name order and disk order
    # disagree, which is what makes the assertion mean anything.
    os.utime(tmp_path / "first.m4a", (100, 100))
    os.utime(tmp_path / "second.m4a", (200, 200))
    os.utime(tmp_path / "third.m4a", (300, 300))

    assert [p.name for p in _settled(tmp_path)] == [
        "first.m4a",
        "second.m4a",
        "third.m4a",
    ]


def test_the_heard_subfolder_is_not_rescanned(tmp_path: Path) -> None:
    """Filing a recording away is what takes it out of the queue, so the folder
    it is filed into must not be part of the queue."""
    done = tmp_path / listen.HEARD_DIRNAME
    done.mkdir()
    _recording(done, "old-meeting.m4a")

    assert _settled(tmp_path) == []


def test_filing_away_moves_and_never_deletes(tmp_path: Path) -> None:
    """The recording is the person's, and the only copy of a meeting may be
    sitting there. Nothing in this path is allowed to be what loses it."""
    path = _recording(tmp_path)

    landed = listen.file_away(path, folder=str(tmp_path))

    assert landed is not None and landed.read_bytes() == b"audio"
    assert not path.exists()
    assert landed.parent.name == listen.HEARD_DIRNAME


def test_filing_away_never_overwrites_a_different_meeting(tmp_path: Path) -> None:
    """Two recorders both producing ``audio.m4a`` is ordinary. Replacing one
    meeting with another silently is not a collision, it is data loss."""
    done = tmp_path / listen.HEARD_DIRNAME
    done.mkdir()
    (done / "standup.m4a").write_bytes(b"the first meeting")

    landed = listen.file_away(_recording(tmp_path), folder=str(tmp_path))

    assert landed is not None and landed.name != "standup.m4a"
    assert (done / "standup.m4a").read_bytes() == b"the first meeting"
    assert landed.read_bytes() == b"audio"


# --- the model ----------------------------------------------------------------


def test_an_uncached_model_is_refused_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expensive one. whisper's own default model is not on this machine, so
    an unpinned call fetches gigabytes at the moment somebody needed it — on a
    train, it simply fails. The refusal must also not touch the network to
    decide, which is why the ``run`` seam is asserted untouched.
    """
    _installed(monkeypatch, cached=False)
    ran = _Run()

    with pytest.raises(listen.NotHeard, match="refusing to download"):
        listen.transcribe(_recording(tmp_path), run=ran)

    assert ran.argv == []


def test_a_missing_whisper_never_reaches_for_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The privacy case: whisper's own path never reaches for a provider.

    A recording of a meeting is a recording of other people. Sending it to a
    vendor happens only through `sarvam`, which runs when the person has set
    SARVAM_API_KEY (DL-078), never as a quiet fallback from in here.
    """
    monkeypatch.setattr(listen, "whisper_binary", lambda explicit=None: None)

    with pytest.raises(listen.NotHeard, match="whisper is not installed"):
        listen.transcribe(_recording(tmp_path))


def test_not_ready_names_the_two_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked once per pass, and the distinction it draws is load-bearing: a
    receipt is permanent, so an environment failure must not produce one."""
    _installed(monkeypatch)
    assert listen.not_ready() is None

    _installed(monkeypatch, cached=False)
    assert "model" in (listen.not_ready() or "")

    monkeypatch.setattr(listen, "whisper_binary", lambda explicit=None: None)
    assert "not installed" in (listen.not_ready() or "")


def test_the_model_is_always_named_on_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belt to the refusal's braces. If the flag were ever dropped, whisper
    would silently fall back to its own uncached default."""
    _installed(monkeypatch)
    ran = _Run()

    listen.transcribe(_recording(tmp_path), run=ran)

    assert "--model" in ran.argv
    assert ran.argv[ran.argv.index("--model") + 1] == listen.WHISPER_MODEL


# --- the transcript -----------------------------------------------------------


def test_a_transcript_comes_back_without_its_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stamps are read for the duration and then stripped: the review model is
    being asked what was said, and timestamps in the prose are noise it would
    try to interpret."""
    _installed(monkeypatch)

    heard = listen.transcribe(_recording(tmp_path), run=_Run())

    assert "-->" not in heard.text
    assert heard.text.splitlines() == [
        "Quick standup.",
        "We agreed not to move the ship date.",
    ]
    assert heard.duration == pytest.approx(8.76)
    assert heard.words == 10


def test_a_transcript_with_no_stamps_has_no_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` and not zero. A duration of zero is a claim about a recording,
    and "I could not tell" is not that claim — the episode rejects a zero for
    exactly this reason."""
    _installed(monkeypatch)

    heard = listen.transcribe(
        _recording(tmp_path), run=_Run(stdout="", txt="just some words\n")
    )

    assert heard.duration is None
    assert heard.text == "just some words"


def test_a_failed_run_is_not_an_empty_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every way the subprocess can fail has to raise. An empty string returned
    quietly would be filed as a meeting that said nothing."""
    _installed(monkeypatch)
    path = _recording(tmp_path)

    for ran, match in [
        (_Run(returncode=1), "whisper failed"),
        (_Run(txt=None), "wrote no transcript"),
        (_Run(stdout="", txt="   \n"), "transcribed to nothing"),
        (_Run(raises=subprocess.TimeoutExpired("whisper", 1)), "timed out"),
        (_Run(raises=OSError("no such binary")), "could not run whisper"),
    ]:
        with pytest.raises(listen.NotHeard, match=match):
            listen.transcribe(path, run=ran)


# --- the review ---------------------------------------------------------------


def test_a_review_is_parsed_and_rendered(tmp_path: Path) -> None:
    answer = json.dumps(
        {
            "summary": "A standup about the release.",
            "important": ["The ship date is Friday"],
            "highlights": ['"we are not moving it again"'],
            "actions": ["Cut the release branch"],
        }
    )
    fp = provider.FakeProvider({provider.LEARN: lambda role, messages: answer})

    read = listen.review("some transcript", complete=fp.complete)

    assert not read.empty
    rendered = read.render(title="standup.m4a")
    assert "# standup.m4a" in rendered
    assert "## To do" in rendered
    assert "- Cut the release branch" in rendered


def test_an_empty_review_renders_no_empty_headings() -> None:
    """A meeting with no decisions and nothing to do is a real outcome. A
    heading that is usually blank teaches a reader to skip it, and by the time
    it matters they have learned to."""
    read = listen.parse_review(json.dumps({"summary": "Mostly scheduling.", "actions": []}))

    rendered = read.render()

    assert rendered == "Mostly scheduling."
    assert "To do" not in rendered
    assert not read.empty  # it has a summary


def test_a_wholly_empty_review_says_so() -> None:
    """The distinction the caller needs: reviewed and there was nothing in it,
    versus not reviewed. A check that could not tell these apart would pass on
    a dead model."""
    read = listen.parse_review(json.dumps({"summary": "  "}))

    assert read.empty
    assert read.render() == ""


def test_a_fenced_answer_is_accepted_but_prose_is_not() -> None:
    """The one leniency, and its limit. Hunting a JSON object out of prose would
    read a model that ignored the format as though it had followed it."""
    fenced = listen.parse_review('```json\n{"summary": "Fine."}\n```')
    assert fenced.summary == "Fine."

    with pytest.raises(listen.NotReviewed):
        listen.parse_review('Here is the review: {"summary": "Fine."}')


def test_every_malformed_answer_raises(tmp_path: Path) -> None:
    """A failure must never be able to present as an empty review."""
    for bad, match in [
        ("", "no content"),
        ("not json at all", "not JSON"),
        ("[1, 2]", "not a JSON object"),
        (json.dumps({"important": []}), "no .summary."),
        (json.dumps({"summary": 3}), "must be a string"),
    ]:
        with pytest.raises(listen.NotReviewed, match=match):
            listen.parse_review(bad)


def test_a_nonsense_field_is_dropped_not_fatal() -> None:
    """Lenient *inside* a list and strict about the envelope: a number among the
    actions is still an answer in the format; a string where the list belongs
    is not."""
    read = listen.parse_review(
        json.dumps({"summary": "s", "actions": ["real", 7, None, "  "], "important": "one"})
    )

    assert read.actions == ("real",)
    assert read.important == ("one",)  # a lone string is unambiguous


def test_an_empty_transcript_is_never_sent_to_a_model() -> None:
    def explode(role, messages):  # pragma: no cover - must not be reached
        raise AssertionError("the model was called with nothing to review")

    with pytest.raises(listen.NotReviewed, match="no transcript"):
        listen.review("   ", complete=provider.FakeProvider({provider.LEARN: explode}).complete)


# --- end to end ---------------------------------------------------------------


def _answers(claim: str = "They run standups on Mondays.") -> provider.FakeProvider:
    """One provider answering both model calls a recording makes.

    Both go through :data:`provider.LEARN` — the reflection pass and the review
    are the same role — so the double has to answer whichever arrives, and it
    tells them apart the way the real prompts differ.
    """

    def answer(role, messages):
        asked = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for m in messages
            for part in (m["content"] if isinstance(m["content"], list) else [m["content"]])
        )
        if "JSON only" in asked or "highlights" in asked:
            return json.dumps(
                {"summary": "A standup.", "important": ["Friday"], "actions": ["branch"]}
            )
        return json.dumps(
            {"claims": [{"text": claim, "situation": "when planning their week"}]}
        )

    return provider.FakeProvider({provider.LEARN: answer})


def _ready(store, folder: Path, monkeypatch: pytest.MonkeyPatch, **kw):
    """An executor over a log omega has already been spoken through.

    Not scenery: a claim names the episode it came from, so the pass refuses to
    run against an empty log — the first thing in it must not be a belief about
    a person omega has not met.
    """
    _installed(monkeypatch)
    monkeypatch.setattr(
        listen,
        "transcribe",
        kw.pop("transcribe", lambda path, **_: listen.Heard(text=STAMPED_TEXT, duration=8.76)),
    )
    q = EventQueue(store)
    q.append(episodes.inbound("morning", channel="tray"))
    fp = kw.pop("provider", None) or _answers()
    ex = Executor(q, complete=fp.complete)
    ex.recover()
    return q, ex, fp


STAMPED_TEXT = "Quick standup.\nWe agreed not to move the ship date."


def _receipts(store) -> list[dict[str, Any]]:
    """Every capture receipt, read back from the bytes — *grade the world, not
    the words*."""
    out = []
    for ep in store.episodes_since(0):
        payload = episodes.decode(ep.payload)
        if payload.get("kind") == episodes.AUDIO_CAPTURED:
            out.append(payload)
    return out


def test_a_recording_is_heard_reviewed_remembered_and_filed_away(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capability. One recording in the folder becomes, in one pass: a
    transcript and a review in the blob store, claims in memory, a receipt in
    the log, and a file that has left the queue."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder)
    q, ex, _ = _ready(store, folder, monkeypatch)

    assert ex.hear(folder=str(folder), now=1e12) == 1

    (receipt,) = _receipts(store)
    assert receipt["title"] == "standup.m4a"
    assert receipt["reason"] is None
    assert receipt["filed"] == 1
    assert receipt["duration"] == pytest.approx(8.76)

    bits = blobs.BlobStore.open(store.root)
    assert bits.has(receipt["recording"])
    assert bits.path_for(receipt["transcript"]).read_text() == STAMPED_TEXT
    assert "## To do" in bits.path_for(receipt["review"]).read_text()

    assert not (folder / "standup.m4a").exists()
    assert (folder / listen.HEARD_DIRNAME / "standup.m4a").exists()

    claims = [
        episodes.decode(ep.payload)
        for ep in store.episodes_since(0)
        if episodes.decode(ep.payload).get("kind") == episodes.CLAIM_EXTRACTED
    ]
    assert len(claims) == 1
    assert claims[0]["explicit"] is False  # inferred, not told


def test_a_machine_without_whisper_writes_no_receipts(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The violation that must not regress, and the one that would be worst.

    Every receipt is permanent and marks a recording heard forever. A pass that
    wrote one per file because whisper was not installed would burn the whole
    folder on the first idle moment of a fresh machine, and installing whisper
    afterwards would recover none of it. A failure to *look* is not a failure to
    hear any particular thing.
    """
    folder = tmp_path / "recordings"
    folder.mkdir()
    for name in ["a.m4a", "b.m4a", "c.m4a"]:
        _recording(folder, name)
    q, ex, _ = _ready(store, folder, monkeypatch)
    monkeypatch.setattr(listen, "whisper_binary", lambda explicit=None: None)

    assert ex.hear(folder=str(folder), now=1e12) == 0

    assert _receipts(store) == []
    assert len(_settled(folder)) == 3  # all three still waiting


def test_the_same_recording_renamed_is_not_heard_twice(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other violation, and why the cursor is keyed on content.

    iCloud re-syncs a file under a new name, and a person who cannot remember
    whether they already dropped a meeting in drops it again. A name-keyed
    cursor would re-hear it — paying for two model calls and filing the same
    claim twice — and would also skip a *different* meeting that happened to
    reuse an old name. The digest of the bytes can do neither.
    """
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder, "standup.m4a", b"identical bytes")
    q, ex, fp = _ready(store, folder, monkeypatch)
    assert ex.hear(folder=str(folder), now=1e12) == 1
    calls_after_first = len(fp.calls_for(provider.LEARN))
    assert calls_after_first == 2  # reflection + review: the baseline is real

    _recording(folder, "standup-copy.m4a", b"identical bytes")
    ex.hear(folder=str(folder), now=1e12)

    assert len(_receipts(store)) == 1, "the same audio earned a second receipt"
    assert not (folder / "standup-copy.m4a").exists(), "the copy was left in the queue"
    assert len(fp.calls_for(provider.LEARN)) == calls_after_first, (
        "a known recording cost a model call"
    )


def test_a_recording_that_cannot_be_transcribed_still_leaves_a_receipt(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counterpart to the group above. This failure *is* about the file —
    the environment was checked before the pass — so it earns a receipt, or the
    next pass reads it again forever."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder, "corrupt.m4a")

    def fail(path, **_):
        raise listen.NotHeard("whisper failed: invalid data")

    q, ex, _ = _ready(store, folder, monkeypatch, transcribe=fail)

    assert ex.hear(folder=str(folder), now=1e12) == 1

    (receipt,) = _receipts(store)
    assert "invalid data" in receipt["reason"]
    assert receipt["filed"] == 0
    assert receipt["transcript"] is None
    assert receipt["review"] is None


def test_a_failed_review_does_not_lose_the_transcript(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real partial, and the payload is built to say so. Reporting the
    transcript as absent would make work that landed look lost."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder)

    def answer(role, messages):
        return "the model is having a bad day"

    q, ex, _ = _ready(
        store, folder, monkeypatch,
        provider=provider.FakeProvider({provider.LEARN: answer}),
    )

    assert ex.hear(folder=str(folder), now=1e12) == 1

    (receipt,) = _receipts(store)
    assert receipt["reason"]
    assert receipt["filed"] == 0
    assert receipt["transcript"] is not None
    bits = blobs.BlobStore.open(store.root)
    assert bits.path_for(receipt["transcript"]).read_text() == STAMPED_TEXT


def test_one_recording_per_pass(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap, and it is not timidity. The executor is a single consumer and
    transcribing an hour of audio takes tens of minutes, so every recording in
    a pass is time omega cannot answer a message."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    for n in range(3):
        _recording(folder, f"m{n}.m4a", f"audio {n}".encode())
    q, ex, _ = _ready(store, folder, monkeypatch)

    assert ex.hear(folder=str(folder), now=1e12) == listen.MAX_PER_PASS == 1
    assert len(_receipts(store)) == 1
    assert len(_settled(folder)) == 2


def test_an_empty_log_hears_nothing(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omega never spoken to must not have, as the first thing in its log, a
    belief about a person it has not met inferred from a file it found."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder)
    _installed(monkeypatch)
    monkeypatch.setattr(
        listen, "transcribe", lambda path, **_: listen.Heard(text=STAMPED_TEXT)
    )
    ex = Executor(EventQueue(store), complete=_answers().complete)
    ex.recover()

    assert ex.hear(folder=str(folder), now=1e12) == 0
    assert _receipts(store) == []


def test_the_recording_lens_is_the_one_used(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reflection pass must read this material as a room with other people
    in it. Pointed at the wrong lens it would read a colleague's sentence as a
    fact about the person — which is the one invention this path is bounded
    against."""
    folder = tmp_path / "recordings"
    folder.mkdir()
    _recording(folder)
    seen: list[str] = []

    def answer(role, messages):
        seen.append(str(messages[0]["content"]))
        if "JSON only" in str(messages[0]["content"]):
            return json.dumps({"summary": "s"})
        return json.dumps({"claims": []})

    q, ex, _ = _ready(
        store, folder, monkeypatch,
        provider=provider.FakeProvider({provider.LEARN: answer}),
    )
    ex.hear(folder=str(folder), now=1e12)

    reflection = [s for s in seen if "JSON only" not in s]
    assert reflection, "the reflection pass never ran"
    assert learn.ROOM.opening[0] in reflection[0]
    assert "Other people are speaking" in reflection[0]
