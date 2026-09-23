"""The content-addressed blob store beside the log — DL-027.

Organised green / red / yellow, like the rest of the suite:

* **green** — bytes go in, are named by themselves, and come back identical;
* **red** — it refuses what it must refuse, by name, and leaves nothing behind;
* **yellow** — the awkward middle: a crash between write and rename, an empty
  file, the same content under two names.

Every digest asserted here is computed **independently in the test**, with
``hashlib`` over bytes the test itself holds. Comparing the store's digest to
the store's digest would prove only that it is consistent with itself, which is
the one property content addressing does not need a test for.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from omega.blobs import (
    BLOBS_DIRNAME,
    DEFAULT_MIME,
    BadDigest,
    BlobStore,
    IngestFailed,
    NotARegularFile,
    is_digest,
)


@pytest.fixture
def blobs(store_dir: Path) -> BlobStore:
    """A blob store beside a log store, opened the way the runtime opens it."""
    return BlobStore.open(store_dir)


def sha256_of(data: bytes) -> str:
    """The reference answer, computed here and not asked of the store."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stored_files(blobs: BlobStore) -> list[Path]:
    """Every file under the blob root — blobs, leftovers and all.

    Deliberately not filtered to things that look like blobs: a test asserting
    "nothing was left behind" has to be able to see a leftover temp file, which
    is exactly what a filter would hide.
    """
    return sorted(p for p in blobs.root.rglob("*") if p.is_file())


# --- green ------------------------------------------------------------------


def test_a_file_goes_in_and_its_bytes_come_back_unchanged(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """The whole contract in one case: identical bytes, and a digest that is the
    real sha256 of them rather than whatever the implementation computed."""
    content = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 7
    source = tmp_path / "capture.png"
    source.write_bytes(content)

    ref = blobs.put(source)

    assert ref.digest == sha256_of(content), "the digest must be the real sha256"
    assert ref.mime == "image/png"
    assert ref.bytes == len(content)
    assert blobs.has(ref.digest)
    assert blobs.path_for(ref.digest).read_bytes() == content


def test_the_layout_is_a_fanout_directory_and_the_rest_of_the_digest(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """Layout, stated once so a change to it is a decision rather than a drift.

    Two hex characters name the directory, the remaining sixty-two name the
    file, and there is **no extension** — the media type lives in the episode,
    where it is a fact about the attachment rather than a property of a filename
    anything on the box could rename.
    """
    source = tmp_path / "notes.md"
    source.write_bytes(b"# hello")
    ref = blobs.put(source)

    hex_digits = ref.digest.removeprefix("sha256:")
    stored = blobs.path_for(ref.digest)

    assert stored.parent.name == hex_digits[:2]
    assert stored.name == hex_digits[2:]
    assert len(stored.name) == 62 and stored.suffix == ""
    assert stored.parent.parent == blobs.root


def test_the_same_content_twice_is_one_file_and_the_same_reference(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """Dedup is not a feature here; it is what content addressing *is*. The
    second put has nowhere else to write, because the name is the content."""
    content = b"the same screenshot, dropped in twice"
    first_path = tmp_path / "one.txt"
    first_path.write_bytes(content)

    first = blobs.put(first_path)
    second = blobs.put(first_path)

    assert first == second
    assert len(stored_files(blobs)) == 1


def test_the_same_content_under_two_names_is_still_one_file(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """The filename is not part of the identity. Two names for the same bytes
    are one blob; what differs is the *mime*, which is read from the name and
    belongs to the episode rather than to the stored file."""
    content = b"identical bytes, two filenames"
    (tmp_path / "a.txt").write_bytes(content)
    (tmp_path / "b.html").write_bytes(content)

    text = blobs.put(tmp_path / "a.txt")
    html = blobs.put(tmp_path / "b.html")

    assert text.digest == html.digest == sha256_of(content)
    assert text.bytes == html.bytes == len(content)
    assert text.mime == "text/plain" and html.mime == "text/html"
    assert len(stored_files(blobs)) == 1, "one content, one file"


def test_a_multi_megabyte_file_round_trips(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """The streaming path, proven on something bigger than one chunk.

    Random bytes on purpose: a repeating pattern would survive a copy loop that
    mishandles chunk boundaries, and this is the only case in the file large
    enough for that loop to have boundaries at all.
    """
    content = os.urandom(5 * 1024 * 1024 + 12345)
    source = tmp_path / "recording.bin"
    source.write_bytes(content)

    ref = blobs.put(source)

    assert ref.digest == sha256_of(content)
    assert ref.bytes == len(content)
    assert ref.mime == DEFAULT_MIME, "an unknown extension is named, not guessed"
    assert blobs.path_for(ref.digest).read_bytes() == content


def test_a_reference_resolves_or_does_not_and_never_half_way(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """``has`` answers about the store, ``path_for`` answers about the digest.

    ``path_for`` is pure — it names where a blob *would* live — so it works for
    content that was never stored. That is what makes it usable for the "is this
    reference resolvable" question rather than only after the fact.
    """
    never_stored = sha256_of(b"nothing ever put this in")
    assert blobs.has(never_stored) is False
    assert blobs.path_for(never_stored).exists() is False

    source = tmp_path / "x.txt"
    source.write_bytes(b"nothing ever put this in")
    assert blobs.put(source).digest == never_stored
    assert blobs.has(never_stored) is True


def test_opening_the_store_twice_is_the_same_store(store_dir: Path) -> None:
    """There is no handle and no lock — a blob store is a directory and a naming
    rule — so two openers are two views of one thing, not a conflict."""
    first = BlobStore.open(store_dir)
    second = BlobStore.open(store_dir)
    assert first.root == second.root == store_dir / BLOBS_DIRNAME

    source = store_dir.parent / "shared.txt"
    source.write_bytes(b"written through one, read through the other")
    ref = first.put(source)
    assert second.has(ref.digest)


# --- red --------------------------------------------------------------------


def test_a_directory_is_refused_by_name_and_stores_nothing(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """A directory has no bytes to attach. The message says *which* kind of
    thing it was, because "cannot attach" alone sends the caller back to guess.
    """
    victim = tmp_path / "a-folder"
    victim.mkdir()

    with pytest.raises(NotARegularFile, match="directory"):
        blobs.put(victim)
    assert stored_files(blobs) == []


def test_a_missing_path_is_refused_by_name_and_stores_nothing(
    blobs: BlobStore, tmp_path: Path
) -> None:
    with pytest.raises(NotARegularFile):
        blobs.put(tmp_path / "was-never-here.png")
    assert stored_files(blobs) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs only")
def test_a_fifo_is_refused_rather_than_read_forever(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """The case the whole regular-file check exists for, and the reason the
    store opens with ``O_NONBLOCK``.

    A FIFO opened for reading blocks until a writer appears, so a check that
    opened first and asked what it had second would hang here — with no writer,
    forever, holding the connection that asked. This must return, and it must
    return an error. Character devices are the same family: ``/dev/zero`` has no
    end, so without this check a missing size cap is not "large copies allowed"
    but "one copy that never finishes".
    """
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(NotARegularFile, match="FIFO"):
        blobs.put(fifo)
    assert stored_files(blobs) == []


def test_a_malformed_digest_is_refused_rather_than_normalised(
    blobs: BlobStore,
) -> None:
    """Refused, never cleaned up. A digest tidied on the way in is no longer a
    claim about the bytes — it is a claim about the bytes after we changed the
    question, and content addressing is worth exactly what that claim is."""
    real = "sha256:" + "ab" * 32
    wrong = [
        "ab" * 32,  # no algorithm; the day there are two it is ambiguous
        "sha256:" + "AB" * 32,  # uppercase: two spellings of one content
        "sha256:" + "ab" * 31,  # too short
        "sha256:" + "ab" * 33,  # too long
        "sha256:" + "zz" * 32,  # not hex
        "sha1:" + "ab" * 32,  # a real algorithm, not this one
        "",
    ]
    for candidate in wrong:
        assert not is_digest(candidate)
        with pytest.raises(BadDigest):
            blobs.path_for(candidate)
        with pytest.raises(BadDigest):
            blobs.has(candidate)

    assert is_digest(real)
    assert blobs.has(real) is False, "well-formed and absent is not an error"


def test_a_digest_that_is_not_a_string_is_refused(blobs: BlobStore) -> None:
    """``is_digest`` is total and never raises, so it can be used as a predicate
    by the payload codec without that codec having to pre-check the type."""
    for candidate in (None, 42, b"sha256:" + b"ab" * 32, ["sha256:"], True):
        assert is_digest(candidate) is False
        with pytest.raises(BadDigest):
            blobs.path_for(candidate)  # type: ignore[arg-type]


# --- yellow -----------------------------------------------------------------


def test_an_interrupted_put_leaves_nothing_visible_in_the_store(
    blobs: BlobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between write-temp and rename must cost the disk space and
    nothing else.

    The rename is the only step that makes a file appear under its digest, so
    failing before it can leave a *leftover*, never a *blob*: a half-copied file
    under a digest would be content that lies about its own name, and every
    reader downstream trusts that name precisely because it cannot lie.

    The failure is injected at ``os.replace`` — the last possible moment, with
    the temp file fully written and fsynced — because that is the window an
    earlier, cheaper check would miss.
    """
    content = b"interrupted halfway to its name"
    source = tmp_path / "doomed.bin"
    source.write_bytes(content)

    def die(*args: object, **kwargs: object) -> None:
        raise OSError(5, "simulated I/O error during rename")

    monkeypatch.setattr(os, "replace", die)

    with pytest.raises(IngestFailed):
        blobs.put(source)

    monkeypatch.undo()
    assert blobs.has(sha256_of(content)) is False
    assert stored_files(blobs) == [], "not even the temp file survives"

    # And the store still works afterwards: the failure was about one put.
    assert blobs.put(source).digest == sha256_of(content)


def test_an_empty_file_is_a_real_attachment(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """Zero bytes is content, not a missing file.

    The digest of the empty string is a well-known constant, so this also pins
    that ``bytes`` is counted off the stream rather than inferred from anything
    that would special-case a zero-length read.
    """
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    ref = blobs.put(source)

    assert ref.digest == sha256_of(b"")
    assert ref.digest.endswith(
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert ref.bytes == 0
    assert blobs.has(ref.digest)
    assert blobs.path_for(ref.digest).read_bytes() == b""


def test_a_symlink_to_a_regular_file_attaches_what_it_points_at(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """A symlink is followed on purpose — the person named that path, and its
    content is a regular file's content. What the check refuses is the *kind* of
    thing finally opened, which is why a link to a FIFO is still refused."""
    target = tmp_path / "real.txt"
    target.write_bytes(b"behind a link")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    assert blobs.put(link).digest == sha256_of(b"behind a link")

    dangling = tmp_path / "dangling.txt"
    dangling.symlink_to(tmp_path / "nowhere")
    with pytest.raises(NotARegularFile):
        blobs.put(dangling)


def test_the_bytes_reported_are_the_bytes_hashed(
    blobs: BlobStore, tmp_path: Path
) -> None:
    """``bytes`` is counted off the stream that produced the digest, not read
    from ``stat``. The two can disagree if the file changes mid-ingest, and only
    one of them describes the content the digest names."""
    content = os.urandom(3 * 1024 * 1024)
    source = tmp_path / "big.bin"
    source.write_bytes(content)

    ref = blobs.put(source)

    assert ref.bytes == len(content)
    assert ref.bytes == blobs.path_for(ref.digest).stat().st_size
    assert ref.digest == sha256_of(content)


def test_the_blob_store_is_a_sibling_of_the_log_not_a_part_of_it(
    store_dir: Path
) -> None:
    """DL-027's seam, checked as a fact about the tree rather than a promise.

    The log owns ``episodes.log``; this owns ``blobs/``. They share a directory
    and nothing else — opening one does not create or touch the other's files,
    which is what lets either change later without dragging the other along.
    """
    import ast

    import omega.blobs

    tree = ast.parse(Path(omega.blobs.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not [name for name in imported if name.split(".")[0] == "omega"], (
        f"the blob store imports {sorted(imported)}; it must not reach into the "
        f"log, and the log must not learn it exists"
    )

    blobs = BlobStore.open(store_dir)
    assert blobs.root == store_dir / BLOBS_DIRNAME
    assert not (store_dir / "episodes.log").exists(), (
        "opening the blob store must not create anything of the log's"
    )
