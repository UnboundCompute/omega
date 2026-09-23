//! The log itself: open, recovery, append, scan, the two in-memory caches, and
//! the exclusive advisory lock.
//!
//! Durability rules (M0_SPEC.md):
//!   * `fsync` after every append, before the append returns;
//!   * `fsync` the parent directory once after creating the file;
//!   * exclusive advisory lock held for the lifetime of the open log.
//!
//! Both indexes are caches rebuilt from the file on open. They are never
//! authoritative (DL-016).

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointStore, SidecarLoad};
use crate::frame::{self, HEADER_LEN, PREFIX_LEN};
use crate::sync::{sync_dir, sync_file};
use crate::Error;

/// The spec names the log file. Callers that think in directories should join
/// this, rather than inventing a name of their own.
pub const EPISODES_FILENAME: &str = "episodes.log";

/// What a full scan of the file produced.
#[derive(Debug, Default)]
pub struct Scan {
    /// `offsets[n - 1]` is the file offset of the frame with `seq == n`.
    pub offsets: Vec<u64>,
    /// `key -> seq`, empty keys excluded.
    pub dedup: HashMap<String, u64>,
    /// Offset just past the last good frame.
    pub good_end: u64,
    /// `(offset, body_len)` for each frame whose length field was damaged but
    /// whose true length its own content identified. Recovery writes these
    /// back; see `recoverable_body_len`.
    pub repairs: Vec<(u64, u32)>,
}

pub struct Log {
    file: File,
    path: PathBuf,
    dir: PathBuf,
    /// Current file length in bytes; always `good_end` after recovery.
    len: u64,
    offsets: Vec<u64>,
    dedup: HashMap<String, u64>,
    checkpoints: CheckpointStore,
    /// Bytes truncated by recovery on this open. Diagnostic only.
    recovered_bytes: u64,
    /// Length fields recovery proved wrong and rewrote on this open. A
    /// non-zero value means the file was damaged and is now correct, which is
    /// survivable but worth knowing about. Diagnostic only.
    repaired_lengths: u64,
    /// Set when the sidecar could not be decoded and every checkpoint was reset
    /// to 0 on this open. The damaged file is preserved, not deleted.
    checkpoints_reset: bool,
    /// Where the unreadable sidecar was moved to, when it could be moved.
    damaged_checkpoints_path: Option<PathBuf>,
    /// Where a truncated tail's bytes were kept, when they were not just zeros.
    discarded_tail_path: Option<PathBuf>,
}

impl std::fmt::Debug for Log {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Log")
            .field("path", &self.path)
            .field("head", &self.head())
            .field("len", &self.len)
            .finish()
    }
}

impl Log {
    /// Open (creating if needed) the log at `path`, take the exclusive lock, and
    /// run the recovery procedure. `path` is the log **file**, not a directory.
    pub fn open(path: &Path) -> Result<Log, Error> {
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        // Lock first: everything below writes, and a second writer is the one
        // thing that can corrupt a correct implementation (DL-016).
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::AlreadyLocked),
            Err(TryLockError::Error(e)) => return Err(Error::Io(e)),
        }

        let mut log = Log {
            file,
            path: path.to_path_buf(),
            dir,
            len: 0,
            offsets: Vec::new(),
            dedup: HashMap::new(),
            checkpoints: CheckpointStore::empty(path),
            recovered_bytes: 0,
            repaired_lengths: 0,
            checkpoints_reset: false,
            damaged_checkpoints_path: None,
            discarded_tail_path: None,
        };
        log.recover()?;
        crate::checkpoint::clear_stale_temp(path);

        // Step 8, first half: load the checkpoints. A sidecar that cannot be
        // decoded is derived state, not the source of truth, so it is set aside
        // and every checkpoint reads 0 — the same outcome a *missing* sidecar
        // has always had. It must never be the reason acknowledged episodes
        // become unreachable.
        let SidecarLoad {
            store,
            damaged_moved_to,
            was_damaged,
        } = CheckpointStore::load_or_set_aside(path)?;
        log.checkpoints = store;
        log.checkpoints_reset = was_damaged;
        log.damaged_checkpoints_path = damaged_moved_to;

        // Step 8, second half: a *well-formed* checkpoint ahead of `head` is the
        // opposite case. It cannot legitimately happen, so it is proof that
        // acknowledged episodes were lost, and this open is the one moment that
        // proof exists. Checking it lazily on the next read left the log opening
        // "cleanly" and the consumer wedged: unable to read and unable to reset
        // its own position, because both paths raised. The documented way out is
        // to remove the sidecar by hand — a missing one reads as 0, consumers
        // replay, and DL-007's write-key dedup makes replay idempotent.
        let head = log.head();
        for name in log.checkpoints.names() {
            let at = log.checkpoints.get(&name);
            if at > head {
                return Err(Error::CheckpointAhead {
                    requested: at,
                    head,
                });
            }
        }
        Ok(log)
    }

    /// Recovery, run on every open. See M0_SPEC.md "Recovery".
    fn recover(&mut self) -> Result<(), Error> {
        let mut len = self.file.metadata()?.len();

        // Steps 1-3: missing / 0 bytes / torn header. No record can exist yet,
        // so truncate to 0 and (re)write the header — but only once the bytes
        // that are there have been shown to be the start of one of our headers.
        //
        // The order matters and did not always. Truncating first meant any short
        // file at this path — a note, a stray text file — was silently
        // overwritten with a log header, because the magic check sat behind the
        // destructive step and never ran for files under 32 bytes.
        if len < HEADER_LEN as u64 {
            if len > 0 {
                let n = len as usize;
                let mut present = vec![0u8; n];
                self.file.read_exact_at(&mut present, 0)?;
                let magic_seen = std::cmp::min(n, frame::MAGIC.len());
                if present[..magic_seen] != frame::MAGIC[..magic_seen] {
                    return Err(Error::NotAnOmegaLog);
                }
            }
            self.file.set_len(0)?;
            self.file.write_all_at(&frame::encode_header(), 0)?;
            sync_file(&self.file)?;
            // The file may be brand new, so the directory entry needs fsync too
            // or the file itself can vanish on crash.
            sync_dir(&self.dir)?;
            len = HEADER_LEN as u64;
        } else {
            // Step 4: magic and version. Both refuse to open.
            let mut header = [0u8; HEADER_LEN];
            self.file.read_exact_at(&mut header, 0)?;
            frame::validate_header(&header)?;
        }

        // Step 5: scan the frames.
        let scan = scan_frames(&self.file, len)?;

        // Step 6a: write back any length field the scan proved wrong, before
        // anything else can append. The evidence that identifies the true
        // length -- the body CRC, and for the last frame the distance to EOF --
        // is available now and may not be later: one more append puts that
        // frame in the middle, where EOF says nothing about where it ends. A
        // log that opens today and refuses forever after the next write is a
        // worse outcome than either, so the repair is not deferred.
        if !scan.repairs.is_empty() {
            for &(offset, body_len) in &scan.repairs {
                let mut fixed = [0u8; 8];
                fixed[0..4].copy_from_slice(&body_len.to_le_bytes());
                fixed[4..8].copy_from_slice(&frame::len_crc32(body_len).to_le_bytes());
                self.file.write_all_at(&fixed, offset)?;
            }
            sync_file(&self.file)?;
            self.repaired_lengths = scan.repairs.len() as u64;
        }

        // Step 6b: if anything was truncated, set_len and fsync.
        if scan.good_end < len {
            self.recovered_bytes = len - scan.good_end;
            // Recovery is the one moment these bytes would be destroyed, and a
            // tail is only truncated because nothing in it verifies as a frame
            // -- not because anyone knows what it was. Zeros hold no evidence,
            // so they go silently; anything else is kept beside the log.
            if !is_zero_to_eof(&self.file, scan.good_end, len)? {
                self.discarded_tail_path =
                    keep_discarded_tail(&self.file, &self.path, scan.good_end, len);
            }
            self.file.set_len(scan.good_end)?;
            sync_file(&self.file)?;
            len = scan.good_end;
        }

        // Step 7: the caches.
        self.len = len;
        self.offsets = scan.offsets;
        self.dedup = scan.dedup;
        Ok(())
    }

    /// Drop and rebuild both caches from the file. They are caches, so this must
    /// be a no-op observationally (spec case 35).
    pub fn rebuild_indexes(&mut self) -> Result<(), Error> {
        let scan = scan_frames(&self.file, self.len)?;
        if scan.good_end != self.len {
            return Err(Error::CorruptFrame {
                offset: scan.good_end,
                detail: "rescan of a recovered file found a short tail".into(),
            });
        }
        self.offsets = scan.offsets;
        self.dedup = scan.dedup;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes discarded by torn-tail truncation on this open (0 if none).
    pub fn recovered_bytes(&self) -> u64 {
        self.recovered_bytes
    }

    /// How many damaged length fields recovery rewrote on this open.
    pub fn repaired_lengths(&self) -> u64 {
        self.repaired_lengths
    }

    /// True when the checkpoint sidecar could not be decoded on this open, so
    /// it was set aside and every checkpoint now reads 0.
    ///
    /// This is deliberately not an error: the sidecar is derived state and must
    /// never be the reason acknowledged episodes become unreachable. It is
    /// surfaced here so the seam can report it — loud, without being fatal.
    pub fn checkpoints_reset(&self) -> bool {
        self.checkpoints_reset
    }

    /// Where an unreadable sidecar was moved on this open, if it was moved.
    ///
    /// It is kept as evidence, never deleted. `None` with
    /// [`Log::checkpoints_reset`] true means the move itself failed (a
    /// read-only directory, say); the log still opens, because refusing to
    /// would be the failure this whole rule exists to prevent.
    pub fn damaged_checkpoints_path(&self) -> Option<&Path> {
        self.damaged_checkpoints_path.as_deref()
    }

    /// Where a truncated non-zero tail was kept on this open, if there was one.
    pub fn discarded_tail_path(&self) -> Option<&Path> {
        self.discarded_tail_path.as_deref()
    }

    pub fn len_bytes(&self) -> u64 {
        self.len
    }

    /// Seq of the last record, or 0 if the log is empty.
    pub fn head(&self) -> u64 {
        self.offsets.len() as u64
    }

    /// A copy of the offset index, for tests that compare it to a fresh scan.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Append one record and `fsync` before returning.
    ///
    /// Everything is validated before a byte is written, and a write that fails
    /// midway is truncated back, so a rejected append leaves the file
    /// byte-identical and still openable.
    pub fn append(&mut self, payload: &[u8], key: &str, ts_micros: i64) -> Result<u64, Error> {
        frame::validate_append(key, payload.len())?;

        // Empty keys are not deduplicated against each other: "no key" is not a key.
        if !key.is_empty() {
            if let Some(&existing) = self.dedup.get(key) {
                let rec = self.read(existing)?;
                if rec.payload == payload {
                    return Ok(existing); // DL-007 dedup guard: write nothing.
                }
                return Err(Error::WriteKeyConflict {
                    key: key.to_owned(),
                    existing_seq: existing,
                });
            }
        }

        let seq = self.head() + 1;
        let bytes = frame::encode(seq, ts_micros, key, payload)?;
        let at = self.len;

        if let Err(e) = self.file.write_all_at(&bytes, at) {
            self.roll_back_to(at);
            return Err(Error::Io(e));
        }
        if let Err(e) = sync_file(&self.file) {
            self.roll_back_to(at);
            return Err(Error::Io(e));
        }

        self.len = at + bytes.len() as u64;
        self.offsets.push(at);
        if !key.is_empty() {
            self.dedup.insert(key.to_owned(), seq);
        }
        Ok(seq)
    }

    /// Best-effort restoration of the previous file length after a failed write.
    /// Failures here are ignored on purpose: recovery on the next open handles
    /// whatever is left, and the original error is the one worth reporting.
    fn roll_back_to(&mut self, at: u64) {
        let _ = self.file.set_len(at);
        let _ = sync_file(&self.file);
    }

    /// Read one record by sequence number.
    pub fn read(&self, seq: u64) -> Result<frame::Record, Error> {
        if seq == 0 {
            return Err(Error::InvalidArgument(
                "seq 0 is reserved for \"nothing consumed\" and is never a record".into(),
            ));
        }
        if seq > self.head() {
            return Err(Error::CheckpointAhead {
                requested: seq,
                head: self.head(),
            });
        }
        let offset = self.offsets[(seq - 1) as usize];
        read_record_at(&self.file, offset)
    }

    /// Validate a `since` position and return the exclusive range to iterate:
    /// records `since+1 ..= head`. A consumer ahead of the log is an error, not
    /// an empty iterator (spec case 20).
    pub fn range_since(&self, since: u64) -> Result<(u64, u64), Error> {
        let head = self.head();
        if since > head {
            return Err(Error::CheckpointAhead {
                requested: since,
                head,
            });
        }
        Ok((since + 1, head))
    }

    /// All records with `seq > since`, eagerly. The PyO3 layer iterates lazily
    /// instead; this exists for Rust tests and small callers.
    pub fn records_since(&self, since: u64) -> Result<Vec<frame::Record>, Error> {
        let (from, to) = self.range_since(since)?;
        let mut out = Vec::new();
        for seq in from..=to {
            out.push(self.read(seq)?);
        }
        Ok(out)
    }

    pub fn checkpoint(&self, name: &str) -> u64 {
        self.checkpoints.get(name)
    }

    pub fn set_checkpoint(&mut self, name: &str, seq: u64) -> Result<(), Error> {
        let head = self.head();
        if seq > head {
            return Err(Error::CheckpointAhead {
                requested: seq,
                head,
            });
        }
        self.checkpoints.set(name, seq)
    }

    pub fn checkpoint_names(&self) -> Vec<String> {
        self.checkpoints.names()
    }
}

/// Is every byte from `from` to `file_len` zero?
///
/// This is the test that separates a zero-filled tail (crash artifact,
/// truncate) from damage (refuse to open). Zeros only mean "crash artifact"
/// when they run all the way to EOF: zeros followed by real frame bytes are
/// corruption, because whatever wrote those later bytes proves the zeroed
/// region was once something else.
fn is_zero_to_eof(file: &File, from: u64, file_len: u64) -> std::io::Result<bool> {
    const CHUNK: usize = 64 * 1024;
    let mut buf = vec![0u8; CHUNK];

    // Look at the last chunk first. The answer is the same either way, but a
    // crafted file with a multi-gigabyte hole and one non-zero byte at the end
    // is then rejected after two reads instead of after reading the whole hole.
    if file_len - from > CHUNK as u64 {
        let tail_at = file_len - CHUNK as u64;
        file.read_exact_at(&mut buf, tail_at)?;
        if buf.iter().any(|b| *b != 0) {
            return Ok(false);
        }
    }

    let mut at = from;
    while at < file_len {
        let want = std::cmp::min(CHUNK as u64, file_len - at) as usize;
        file.read_exact_at(&mut buf[..want], at)?;
        if buf[..want].iter().any(|b| *b != 0) {
            return Ok(false);
        }
        at += want as u64;
    }
    Ok(true)
}

/// Copy a tail that recovery is about to discard into a file beside the log.
///
/// The same reasoning as the checkpoint sidecar's `.damaged` file: a tail that
/// does not verify is still evidence about what happened to this machine, and
/// recovery destroying it silently means nobody ever gets to look. Written to a
/// fresh name so a second recovery never clobbers the first.
///
/// Best effort by design. Refusing to open the log because the evidence could
/// not be filed would recreate the exact failure this path exists to prevent,
/// so every error here yields `None` and the log opens anyway.
fn keep_discarded_tail(file: &File, path: &Path, from: u64, to: u64) -> Option<PathBuf> {
    let mut bytes = vec![0u8; (to - from) as usize];
    file.read_exact_at(&mut bytes, from).ok()?;

    for n in 0..1000u32 {
        let mut name = path.as_os_str().to_os_string();
        name.push(".discarded-tail");
        if n > 0 {
            name.push(format!(".{n}"));
        }
        let candidate = PathBuf::from(name);
        if candidate.exists() {
            continue;
        }
        return match std::fs::write(&candidate, &bytes) {
            Ok(()) => Some(candidate),
            Err(_) => None,
        };
    }
    None
}

/// The true length of the frame at `offset`, when its own content identifies
/// it despite a damaged length field. `None` when nothing does.
///
/// Reached only when `len_crc` has failed, so `body_len` cannot be believed.
/// A candidate length is accepted only when four facts agree *independently*
/// of the broken field: it is a length we could have written, the body it
/// names fits inside the file, that body matches the body CRC, and the `seq`
/// inside it is the one recovery was expecting next. A 32-bit checksum over
/// the exact bytes plus the expected sequence number is the same standard of
/// evidence every other frame in the file is read under.
///
/// **Why this repairs rather than refuses.** Refusing loses nothing but leaves
/// a log that will not open, and truncating destroys an episode that is
/// provably whole. Both are worse than writing back the four bytes we can
/// prove. The repair is not optional bookkeeping either: the length implied by
/// EOF only identifies the *last* frame, so a log left unrepaired would open
/// today and refuse forever after the next append pushed that frame into the
/// middle. Recovery therefore fixes it on disk before anything else runs.
fn recoverable_body_len(
    file: &File,
    file_len: u64,
    offset: u64,
    p: &frame::Prefix,
    expected_seq: u64,
) -> std::io::Result<Option<u32>> {
    let body_at = offset + PREFIX_LEN as u64;

    // The length the prefix claims. This catches a `body_len` that was
    // rewritten on disk while the rest of the frame stayed whole.
    if body_of_len_verifies(file, file_len, offset, p.body_len, p, expected_seq)? {
        return Ok(Some(p.body_len));
    }

    // The length implied by the end of the file. This catches bit rot in the
    // length field itself -- the case `len_crc` exists to detect, and the one
    // that must never truncate (spec case 40). If the frame really does run
    // from here to EOF, and that body still checksums against the `crc` the
    // writer stored, and still carries the sequence number recovery expects,
    // then every byte of it landed and only the four length bytes rotted.
    //
    // A half-landed append cannot pass this test, because its body is
    // precisely the part that did not arrive: the bytes between `body_at` and
    // EOF are a prefix of the body, or stale blocks, and either way they do
    // not checksum to a `crc` computed over the whole of it.
    let implied = u32::try_from(file_len.saturating_sub(body_at)).unwrap_or(u32::MAX);
    if implied != p.body_len
        && body_of_len_verifies(file, file_len, offset, implied, p, expected_seq)?
    {
        return Ok(Some(implied));
    }

    Ok(None)
}

/// Does a body of exactly `body_len` bytes at `body_at` verify end to end --
/// right size, right checksum, right sequence number?
fn body_of_len_verifies(
    file: &File,
    file_len: u64,
    offset: u64,
    body_len: u32,
    p: &frame::Prefix,
    expected_seq: u64,
) -> std::io::Result<bool> {
    if !frame::body_len_is_plausible(body_len) {
        return Ok(false);
    }
    let body_at = offset + PREFIX_LEN as u64;
    if file_len.saturating_sub(body_at) < body_len as u64 {
        return Ok(false);
    }
    let mut body = vec![0u8; body_len as usize];
    file.read_exact_at(&mut body, body_at)?;
    if frame::crc32(&body) != p.crc {
        return Ok(false);
    }
    match frame::decode_meta(&body, offset) {
        Ok(meta) => Ok(meta.seq == expected_seq),
        Err(_) => Ok(false),
    }
}

/// Starting from a frame that already verifies, does an unbroken run of frames
/// with consecutive sequence numbers reach *exactly* the end of the file?
///
/// One verifying frame is not evidence that anything was durable, because a
/// payload is opaque bytes by design: an episode may legitimately contain an
/// attachment, an export, or a re-ingested record that is itself frame-shaped.
/// Stale disk blocks are worse, because the blocks most likely to be recycled
/// near a log are older generations of that same log. Either way the bytes are
/// frame-shaped without ever having been a durable frame *here*.
///
/// What a genuine trailing run of durable frames always has, and what neither
/// of those has, is continuity: each frame ends exactly where the next begins,
/// the sequence numbers step by one, and the last one ends exactly at EOF with
/// nothing left over. An embedded frame is followed by the rest of its
/// enclosing payload, so it does not chain. A stale run ends where the old file
/// ended, not where this one does.
fn chain_reaches_eof(
    file: &File,
    file_len: u64,
    at: u64,
    head: &frame::Prefix,
    head_body: &[u8],
    expected_seq: u64,
) -> std::io::Result<bool> {
    let mut seq = match frame::decode_meta(head_body, at) {
        Ok(meta) => meta.seq,
        Err(_) => return Ok(false),
    };
    // A durable frame sitting after the damaged one carries a *later* sequence
    // number. One carrying an earlier number is from some previous life of
    // these bytes, not from this log's tail.
    if seq < expected_seq {
        return Ok(false);
    }

    let mut cursor = at + PREFIX_LEN as u64 + head.body_len as u64;
    while cursor < file_len {
        if file_len - cursor < PREFIX_LEN as u64 {
            return Ok(false);
        }
        let mut prefix = [0u8; PREFIX_LEN];
        file.read_exact_at(&mut prefix, cursor)?;
        let p = frame::decode_prefix(&prefix);
        if !p.len_is_intact() || !frame::body_len_is_plausible(p.body_len) {
            return Ok(false);
        }
        let body_at = cursor + PREFIX_LEN as u64;
        if file_len.saturating_sub(body_at) < p.body_len as u64 {
            return Ok(false);
        }
        let mut body = vec![0u8; p.body_len as usize];
        file.read_exact_at(&mut body, body_at)?;
        if frame::crc32(&body) != p.crc {
            return Ok(false);
        }
        match frame::decode_meta(&body, cursor) {
            Ok(meta) if meta.seq == seq + 1 => seq = meta.seq,
            _ => return Ok(false),
        }
        cursor = body_at + p.body_len as u64;
    }

    Ok(cursor == file_len)
}

/// Does the region `from..file_len` hold no frame that was ever durable?
///
/// Reached when a prefix fails its own length checksum and the frame is not
/// otherwise whole, so `body_len` says nothing about where the frame ends and
/// every byte to EOF is unexplained. The spec's rule is that *data following a
/// frame proves that frame was durable*. The precise form of "data" is a frame
/// that verifies end to end; zeros are not one, and neither are the bytes of a
/// write that half landed. Reading any non-zero byte as proof of durability is
/// what made a log refuse to open forever when its only damage was its own
/// interrupted final append.
///
/// An append writes exactly one frame and fsyncs before returning, so at most
/// one frame is ever in flight. More unexplained non-zero bytes than one
/// maximum frame is therefore more than a crash can account for.
fn tail_holds_no_durable_frame(
    file: &File,
    from: u64,
    file_len: u64,
    expected_seq: u64,
) -> std::io::Result<bool> {
    if file_len - from > PREFIX_LEN as u64 + frame::MAX_BODY as u64 {
        return Ok(false);
    }

    const CHUNK: usize = 64 * 1024;
    let span = (file_len - from) as usize;
    let mut buf = vec![0u8; std::cmp::min(CHUNK, span)];
    let mut base = from;

    while base + PREFIX_LEN as u64 <= file_len {
        let want = std::cmp::min(buf.len() as u64, file_len - base) as usize;
        file.read_exact_at(&mut buf[..want], base)?;
        // Only candidates whose whole prefix is inside this chunk; the next
        // pass starts back far enough to catch one straddling the edge.
        let candidates = want.saturating_sub(PREFIX_LEN - 1);
        if candidates == 0 {
            break;
        }
        for i in 0..candidates {
            let mut prefix = [0u8; PREFIX_LEN];
            prefix.copy_from_slice(&buf[i..i + PREFIX_LEN]);
            let p = frame::decode_prefix(&prefix);
            // A cheap filter, not a proof. The one-in-four-billion reading of a
            // 32-bit checksum assumes uniformly random bytes, and this region
            // holds neither: payloads are opaque by design and recycled blocks
            // are usually older generations of this same log. So a match here
            // only earns the candidate a look at whether it *chains* to EOF.
            if !p.len_is_intact() || !frame::body_len_is_plausible(p.body_len) {
                continue;
            }
            let body_at = base + i as u64 + PREFIX_LEN as u64;
            if file_len.saturating_sub(body_at) < p.body_len as u64 {
                continue;
            }
            let mut body = vec![0u8; p.body_len as usize];
            file.read_exact_at(&mut body, body_at)?;
            if frame::crc32(&body) == p.crc
                && chain_reaches_eof(file, file_len, base + i as u64, &p, &body, expected_seq)?
            {
                return Ok(false); // a frame that verifies end to end
            }
        }
        if want < buf.len() {
            break;
        }
        base += candidates as u64;
    }
    Ok(true)
}

/// Read one complete frame at `offset`, verifying its CRC.
pub fn read_record_at(file: &File, offset: u64) -> Result<frame::Record, Error> {
    let mut prefix = [0u8; PREFIX_LEN];
    file.read_exact_at(&mut prefix, offset)?;
    let p = frame::decode_prefix(&prefix);
    // Same order as recovery: the length is checked against its own checksum
    // before it is used to size a read.
    if !p.len_is_intact() {
        return Err(Error::CorruptFrame {
            offset,
            detail: "body_len fails its own checksum".into(),
        });
    }
    if !frame::body_len_is_plausible(p.body_len) {
        return Err(Error::CorruptFrame {
            offset,
            detail: format!("body_len {} out of range", p.body_len),
        });
    }
    let mut body = vec![0u8; p.body_len as usize];
    file.read_exact_at(&mut body, offset + PREFIX_LEN as u64)?;
    if frame::crc32(&body) != p.crc {
        return Err(Error::CorruptFrame {
            offset,
            detail: "CRC mismatch".into(),
        });
    }
    frame::decode_body(&body, offset)
}

/// Step 5 of recovery: walk the frames from offset 32.
///
/// The whole milestone turns on one distinction:
///   * a frame that is **incomplete** at EOF is a torn tail -> stop, truncate;
///   * a frame that is **complete** but invalid, with data after it, is
///     corruption -> refuse to open.
///
/// Data following a frame proves that frame was fully durable at some point, so
/// a bad CRC there is real corruption and truncating it would silently destroy
/// real episodes.
pub fn scan_frames(file: &File, file_len: u64) -> Result<Scan, Error> {
    let mut scan = Scan {
        good_end: HEADER_LEN as u64,
        ..Default::default()
    };
    let mut offset = HEADER_LEN as u64;
    let mut expected_seq: u64 = 1;

    loop {
        let remaining = file_len.saturating_sub(offset);
        if remaining == 0 {
            break; // clean end, exactly on a frame boundary
        }
        if remaining < PREFIX_LEN as u64 {
            break; // torn tail: fewer than a whole prefix remains
        }

        let mut prefix = [0u8; PREFIX_LEN];
        file.read_exact_at(&mut prefix, offset)?;
        // Mutable because a damaged length field that the frame's own content
        // identifies is corrected here and written back after the scan.
        let mut p = frame::decode_prefix(&prefix);

        // FIRST, before `body_len` is used for anything at all: is it the
        // number we wrote? This ordering is the entire version-2 fix. A length
        // that fails its own checksum says nothing about where this frame ends,
        // so the only evidence left is what the bytes look like:
        //
        //   * all zeros to EOF -> a crash that persisted a write's size
        //     extension without its data pages. A frame prefix we wrote is
        //     never all zeros, so this is a zero-filled tail: truncate.
        //   * anything else -> damage. Under version 1 this branch did not
        //     exist and a flipped length bit was *believed*, read as a frame
        //     claiming more bytes than the file held, classified as a torn tail
        //     and silently truncated — destroying every acknowledged episode
        //     in the file with no error raised.
        if !p.len_is_intact() {
            if is_zero_to_eof(file, offset, file_len)? {
                break; // zero-filled tail
            }

            // The length is damaged, but the rest of the frame may not be. If
            // the frame's own body and seq identify a length unambiguously,
            // the frame is whole and only these four bytes are wrong: repair
            // them and carry on reading. Nothing is lost and nothing is
            // refused, which is strictly better than both of the alternatives.
            if let Some(body_len) = recoverable_body_len(file, file_len, offset, &p, expected_seq)?
            {
                scan.repairs.push((offset, body_len));
                p.body_len = body_len;
                p.len_crc = frame::len_crc32(body_len);
            } else {
            // Otherwise the frame's extent is unknown and every byte to EOF is
            // unexplained. Only something durable in there makes this damage.
            // Non-zero bytes alone do not: a write that half landed leaves its
            // own bytes behind, and they were never acknowledged.
            if !tail_holds_no_durable_frame(file, offset, file_len, expected_seq)? {
                return Err(Error::CorruptFrame {
                    offset,
                    detail: format!(
                        "body_len {} fails its own checksum (len_crc is {:#010x}, want {:#010x}), \
                         and a frame that verifies end to end follows it",
                        p.body_len,
                        p.len_crc,
                        frame::len_crc32(p.body_len),
                    ),
                });
            }

                break; // torn tail: a write that landed in part
            }
        }

        // A checksum-valid length we could never have written means the file was
        // built by something other than this code. That is not a torn write.
        if !frame::body_len_is_plausible(p.body_len) {
            return Err(Error::CorruptFrame {
                offset,
                detail: format!(
                    "body_len {} is checksum-valid but outside {}..={}",
                    p.body_len,
                    frame::FIXED_BODY_LEN,
                    frame::MAX_BODY
                ),
            });
        }

        // Only now, with the length verified, may "it claims more bytes than
        // are here" be read as an interrupted write rather than a bad number
        // pointing past the end of intact data.
        if remaining - (PREFIX_LEN as u64) < p.body_len as u64 {
            break; // torn tail: the body did not all make it
        }

        let frame_end = offset + PREFIX_LEN as u64 + p.body_len as u64;
        let ends_at_eof = frame_end == file_len;

        let mut body = vec![0u8; p.body_len as usize];
        file.read_exact_at(&mut body, offset + PREFIX_LEN as u64)?;

        if frame::crc32(&body) != p.crc {
            if ends_at_eof {
                break; // torn tail
            }
            return Err(Error::CorruptFrame {
                offset,
                detail: "CRC mismatch in a frame with data after it".into(),
            });
        }

        let meta = frame::decode_meta(&body, offset)?;
        if meta.seq != expected_seq {
            if ends_at_eof {
                break; // torn tail
            }
            return Err(Error::SequenceBreak {
                offset,
                expected: expected_seq,
                found: meta.seq,
            });
        }

        scan.offsets.push(offset);
        if !meta.key.is_empty() {
            // A well-formed log never repeats a key; if a hand-made one does,
            // the lowest seq wins, matching what append would have returned.
            scan.dedup.entry(meta.key.to_owned()).or_insert(meta.seq);
        }

        expected_seq += 1;
        offset = frame_end;
        scan.good_end = offset;
    }

    Ok(scan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FIXED_BODY_LEN, MAX_BODY, MAX_KEY};
    use crate::testutil::TempDir;
    use std::io::Write;

    fn log_path(dir: &TempDir) -> PathBuf {
        dir.path().join(EPISODES_FILENAME)
    }

    fn open(dir: &TempDir) -> Log {
        Log::open(&log_path(dir)).unwrap()
    }

    fn raw(dir: &TempDir) -> Vec<u8> {
        std::fs::read(log_path(dir)).unwrap()
    }

    fn write_raw(dir: &TempDir, bytes: &[u8]) {
        let mut f = File::create(log_path(dir)).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
    }

    fn reopen_err(dir: &TempDir) -> Error {
        Log::open(&log_path(dir)).unwrap_err()
    }

    /// Offsets of every frame start in a well-formed log's bytes.
    fn frame_offsets(bytes: &[u8]) -> Vec<usize> {
        let mut out = Vec::new();
        let mut at = HEADER_LEN;
        while at + PREFIX_LEN <= bytes.len() {
            let bl = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            out.push(at);
            at += PREFIX_LEN + bl;
        }
        out
    }

    fn body_len_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    /// Overwrite a frame's `body_len` and leave `len_crc` alone — what a flipped
    /// bit on disk looks like, as opposed to a forgery.
    fn damage_body_len(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Recompute a frame's body CRC in place, to forge a frame that is *valid*
    /// and wrong in some other way.
    fn fix_body_crc(bytes: &mut [u8], at: usize) {
        let bl = body_len_at(bytes, at) as usize;
        let crc = frame::crc32(&bytes[at + PREFIX_LEN..at + PREFIX_LEN + bl]);
        bytes[at + 8..at + 12].copy_from_slice(&crc.to_le_bytes());
    }

    /// Opening these bytes must fail as corruption and must not touch the file.
    fn assert_refuses_and_leaves_the_file_alone(dir: &TempDir, bytes: &[u8], label: &str) -> Error {
        write_raw(dir, bytes);
        let err = reopen_err(dir);
        assert_eq!(
            raw(dir),
            bytes,
            "{label}: a refused log must be left byte-identical"
        );
        err
    }

    // ---- green ----------------------------------------------------------

    #[test]
    fn fresh_log_is_empty_and_has_a_header() {
        let d = TempDir::new("fresh");
        let log = open(&d);
        assert_eq!(log.head(), 0);
        assert_eq!(log.len_bytes(), HEADER_LEN as u64);
        drop(log);
        assert_eq!(&raw(&d)[..8], b"OMEGALOG");
    }

    #[test]
    fn append_round_trips_byte_identically() {
        let d = TempDir::new("round");
        let mut log = open(&d);
        let payload = b"\x00\xff\xfe hello \x00".to_vec();
        let seq = log.append(&payload, "k1", 1_700_000_000_000_001).unwrap();
        assert_eq!(seq, 1);
        let rec = log.read(1).unwrap();
        assert_eq!(rec.seq, 1);
        assert_eq!(rec.ts_micros, 1_700_000_000_000_001);
        assert_eq!(rec.key, "k1");
        assert_eq!(rec.payload, payload);
    }

    #[test]
    fn sequences_are_contiguous_and_survive_reopen() {
        let d = TempDir::new("contig");
        {
            let mut log = open(&d);
            for i in 0..50u64 {
                assert_eq!(
                    log.append(format!("p{i}").as_bytes(), "", i as i64)
                        .unwrap(),
                    i + 1
                );
            }
            assert_eq!(log.head(), 50);
        }
        let log = open(&d);
        assert_eq!(log.head(), 50);
        let all = log.records_since(0).unwrap();
        assert_eq!(all.len(), 50);
        for (i, r) in all.iter().enumerate() {
            assert_eq!(r.seq, i as u64 + 1);
            assert_eq!(r.payload, format!("p{i}").as_bytes());
        }
        assert_eq!(log.records_since(40).unwrap().len(), 10);
        assert_eq!(log.records_since(50).unwrap().len(), 0);
    }

    #[test]
    fn dedup_same_key_same_payload_writes_nothing() {
        let d = TempDir::new("dedup");
        let mut log = open(&d);
        let a = log.append(b"body", "key", 1).unwrap();
        let len_after_first = log.len_bytes();
        let b = log.append(b"body", "key", 999).unwrap();
        assert_eq!(a, b);
        assert_eq!(log.head(), 1);
        assert_eq!(log.len_bytes(), len_after_first);
        // and the stored timestamp is the original one
        assert_eq!(log.read(1).unwrap().ts_micros, 1);
    }

    #[test]
    fn dedup_survives_restart() {
        let d = TempDir::new("dedup_restart");
        {
            let mut log = open(&d);
            log.append(b"body", "key", 1).unwrap();
        }
        let mut log = open(&d);
        assert_eq!(log.append(b"body", "key", 2).unwrap(), 1);
        assert_eq!(log.head(), 1);
    }

    #[test]
    fn different_keys_make_different_records() {
        let d = TempDir::new("twokeys");
        let mut log = open(&d);
        assert_eq!(log.append(b"a", "k1", 1).unwrap(), 1);
        assert_eq!(log.append(b"a", "k2", 1).unwrap(), 2);
        assert_eq!(log.head(), 2);
    }

    #[test]
    fn empty_key_is_not_a_key() {
        let d = TempDir::new("emptykey");
        let mut log = open(&d);
        assert_eq!(log.append(b"same", "", 1).unwrap(), 1);
        assert_eq!(log.append(b"same", "", 1).unwrap(), 2);
        assert_eq!(log.head(), 2);
    }

    #[test]
    fn payload_edges_round_trip() {
        let d = TempDir::new("edges");
        let mut log = open(&d);
        let big = vec![0xabu8; 10 * 1024 * 1024];
        let cases: Vec<Vec<u8>> = vec![vec![], vec![0, 0, 0], vec![0xff, 0xfe, 0xfd], big.clone()];
        for (i, c) in cases.iter().enumerate() {
            assert_eq!(log.append(c, "", i as i64).unwrap(), i as u64 + 1);
        }
        for (i, c) in cases.iter().enumerate() {
            assert_eq!(&log.read(i as u64 + 1).unwrap().payload, c);
        }
        drop(log);
        let log = open(&d);
        assert_eq!(log.head(), 4);
        assert_eq!(log.read(4).unwrap().payload, big);
    }

    #[test]
    fn many_appends_are_contiguous() {
        let d = TempDir::new("many");
        let mut log = open(&d);
        for i in 0..10_000u64 {
            log.append(&i.to_le_bytes(), "", i as i64).unwrap();
        }
        assert_eq!(log.head(), 10_000);
        drop(log);
        let log = open(&d);
        assert_eq!(log.head(), 10_000);
        let all = log.records_since(0).unwrap();
        for (i, r) in all.iter().enumerate() {
            assert_eq!(r.seq, i as u64 + 1);
            assert_eq!(r.payload, (i as u64).to_le_bytes());
        }
    }

    // ---- red ------------------------------------------------------------

    #[test]
    fn crc_break_in_a_middle_frame_is_corruption_not_truncation() {
        let d = TempDir::new("midcrc");
        {
            let mut log = open(&d);
            for i in 0..3 {
                log.append(b"payload", "", i).unwrap();
            }
        }
        let before = raw(&d);
        let mut bytes = before.clone();
        // first frame's body starts at 32 + 8
        bytes[HEADER_LEN + PREFIX_LEN + FIXED_BODY_LEN] ^= 0xff;
        write_raw(&d, &bytes);

        let err = reopen_err(&d);
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
        // the file is not truncated and later episodes are not discarded
        assert_eq!(
            std::fs::read(d.path().join(EPISODES_FILENAME))
                .unwrap()
                .len(),
            bytes.len()
        );
    }

    #[test]
    fn bad_magic_and_unknown_version_refuse_to_open() {
        let d = TempDir::new("magic");
        open(&d);
        let mut bytes = raw(&d);
        bytes[0] = b'X';
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::NotAnOmegaLog));

        let mut bytes = raw(&d);
        bytes[0..8].copy_from_slice(b"OMEGALOG");
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::UnsupportedVersion(99)));
    }

    #[test]
    fn oversize_body_len_field_is_corruption() {
        let d = TempDir::new("bodylen");
        {
            let mut log = open(&d);
            log.append(b"a", "", 1).unwrap();
            log.append(b"b", "", 2).unwrap();
        }
        let mut bytes = raw(&d);
        bytes[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&(MAX_BODY + 1).to_le_bytes());
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::CorruptFrame { .. }));

        let mut bytes = raw(&d);
        bytes[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&0u32.to_le_bytes());
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::CorruptFrame { .. }));
    }

    #[test]
    fn broken_sequence_in_a_middle_frame_is_a_sequence_break() {
        let d = TempDir::new("seqbreak");
        {
            let mut log = open(&d);
            for i in 0..3 {
                log.append(b"payload", "", i).unwrap();
            }
        }
        let mut bytes = raw(&d);
        // rewrite frame 1's seq to 7 and fix its CRC, so the frame is *valid*
        // but out of sequence, with data after it
        let body_start = HEADER_LEN + PREFIX_LEN;
        bytes[body_start..body_start + 8].copy_from_slice(&7u64.to_le_bytes());
        fix_body_crc(&mut bytes, HEADER_LEN);
        write_raw(&d, &bytes);

        let err = reopen_err(&d);
        assert!(
            matches!(
                err,
                Error::SequenceBreak {
                    expected: 1,
                    found: 7,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn oversize_append_is_rejected_and_the_file_is_byte_identical() {
        let d = TempDir::new("toobig");
        let mut log = open(&d);
        log.append(b"good", "k", 1).unwrap();
        drop(log);
        let before = raw(&d);

        let mut log = open(&d);
        let huge = vec![7u8; MAX_BODY as usize - FIXED_BODY_LEN + 1];
        assert!(matches!(log.append(&huge, "", 1), Err(Error::TooLarge(_))));
        let long_key = "k".repeat(MAX_KEY + 1);
        assert!(matches!(
            log.append(b"x", &long_key, 1),
            Err(Error::TooLarge(_))
        ));
        assert_eq!(log.head(), 1);
        drop(log);

        assert_eq!(raw(&d), before);
        let log = open(&d); // still openable
        assert_eq!(log.head(), 1);
    }

    #[test]
    fn write_key_conflict_writes_nothing() {
        let d = TempDir::new("conflict");
        let mut log = open(&d);
        log.append(b"first", "k", 1).unwrap();
        drop(log);
        let before = raw(&d);

        let mut log = open(&d);
        let err = log.append(b"second", "k", 2).unwrap_err();
        assert!(
            matches!(
                err,
                Error::WriteKeyConflict {
                    existing_seq: 1,
                    ..
                }
            ),
            "got {err:?}"
        );
        assert_eq!(log.head(), 1);
        drop(log);
        assert_eq!(raw(&d), before);
    }

    #[test]
    fn reading_past_head_is_checkpoint_ahead() {
        let d = TempDir::new("ahead");
        let mut log = open(&d);
        log.append(b"a", "", 1).unwrap();
        assert!(matches!(
            log.range_since(2),
            Err(Error::CheckpointAhead {
                requested: 2,
                head: 1
            })
        ));
        assert!(log.range_since(1).is_ok());
        // seq 0 is reserved: it is never a record, and asking for it is a
        // caller mistake rather than "you are ahead of the log"
        assert!(matches!(log.read(0), Err(Error::InvalidArgument(_))));
        assert!(matches!(
            log.read(2),
            Err(Error::CheckpointAhead {
                requested: 2,
                head: 1
            })
        ));
        // and on an empty log, since=1 is already ahead
        let d2 = TempDir::new("ahead2");
        let log2 = open(&d2);
        assert!(matches!(
            log2.range_since(1),
            Err(Error::CheckpointAhead { .. })
        ));
        assert!(log2.range_since(0).is_ok());
    }

    #[test]
    fn second_open_is_already_locked() {
        let d = TempDir::new("lock");
        let _held = open(&d);
        let err = reopen_err(&d);
        assert!(matches!(err, Error::AlreadyLocked), "got {err:?}");
        drop(_held);
        // and the lock is released on close
        open(&d);
    }

    #[test]
    fn unwritable_path_is_a_clear_io_error() {
        let d = TempDir::new("unwritable");
        let missing = d
            .path()
            .join("no")
            .join("such")
            .join("dir")
            .join("episodes.log");
        let err = Log::open(&missing).unwrap_err();
        assert!(matches!(err, Error::Io(_)), "got {err:?}");
    }

    // ---- yellow: torn tails and recovery --------------------------------

    fn three_records(d: &TempDir) -> Vec<u8> {
        {
            let mut log = open(d);
            for i in 0..3 {
                log.append(format!("payload-{i}").as_bytes(), "", i)
                    .unwrap();
            }
        }
        raw(d)
    }

    #[test]
    fn torn_tail_mid_payload_truncates() {
        let d = TempDir::new("torn_mid");
        let full = three_records(&d);
        let cut = full.len() - 3;
        write_raw(&d, &full[..cut]);

        let log = open(&d);
        assert_eq!(log.head(), 2);
        assert!(log.recovered_bytes() > 0);
        drop(log);
        // the truncation is persisted
        let log = open(&d);
        assert_eq!(log.head(), 2);
        assert_eq!(log.recovered_bytes(), 0);
    }

    #[test]
    fn torn_tail_at_an_exact_frame_boundary() {
        let d = TempDir::new("torn_boundary");
        let full = three_records(&d);
        // keep two whole frames plus the frame prefix of the third
        let mut end = HEADER_LEN;
        for _ in 0..2 {
            let bl = u32::from_le_bytes(full[end..end + 4].try_into().unwrap()) as usize;
            end += PREFIX_LEN + bl;
        }
        write_raw(&d, &full[..end + PREFIX_LEN]);
        let log = open(&d);
        assert_eq!(log.head(), 2);
        assert_eq!(log.len_bytes(), end as u64);
    }

    #[test]
    // Spec case 27. Every length shorter than a frame prefix, up to the last
    // one: below PREFIX_LEN there is no length field to read at all.
    fn torn_tail_shorter_than_a_frame_prefix() {
        for extra in [1usize, 7, PREFIX_LEN - 1] {
            let d = TempDir::new(&format!("torn_{extra}"));
            let full = three_records(&d);
            let mut end = HEADER_LEN;
            for _ in 0..3 {
                let bl = u32::from_le_bytes(full[end..end + 4].try_into().unwrap()) as usize;
                end += PREFIX_LEN + bl;
            }
            let mut bytes = full[..end].to_vec();
            bytes.extend(std::iter::repeat_n(0u8, extra));
            write_raw(&d, &bytes);

            let log = open(&d);
            assert_eq!(log.head(), 3, "extra={extra}");
            assert_eq!(log.len_bytes(), end as u64, "extra={extra}");
        }
    }

    /// Spec case 28, as amended by recovery step 3. A *torn header* is the
    /// first n bytes of a header we were in the middle of writing, so that is
    /// what this writes. A short file whose bytes are not a header prefix is a
    /// different thing entirely and is refused, not truncated — see
    /// `a_small_non_log_file_is_refused_rather_than_overwritten`.
    #[test]
    fn torn_header_recovers_as_an_empty_log() {
        let header = frame::encode_header();
        for n in [0usize, 1, 7, 8, 12, 17, 31] {
            let d = TempDir::new(&format!("tornhdr_{n}"));
            write_raw(&d, &header[..n]);
            let log = open(&d);
            assert_eq!(log.head(), 0, "n={n}");
            assert_eq!(log.len_bytes(), HEADER_LEN as u64, "n={n}");
            drop(log);
            assert_eq!(raw(&d), header, "n={n}");
        }
    }

    #[test]
    fn append_after_torn_tail_continues_the_sequence_with_no_gap() {
        let d = TempDir::new("torn_then_append");
        let full = three_records(&d);
        write_raw(&d, &full[..full.len() - 3]);

        let mut log = open(&d);
        assert_eq!(log.head(), 2);
        assert_eq!(log.append(b"next", "", 9).unwrap(), 3);
        assert_eq!(log.head(), 3);
        drop(log);

        let log = open(&d);
        assert_eq!(log.head(), 3);
        let all = log.records_since(0).unwrap();
        assert_eq!(all.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(all[2].payload, b"next");
    }

    #[test]
    fn recovery_is_idempotent() {
        let d = TempDir::new("idempotent");
        let full = three_records(&d);
        write_raw(&d, &full[..full.len() - 5]);

        let first = {
            let log = open(&d);
            assert!(log.recovered_bytes() > 0);
            (log.head(), log.len_bytes(), log.offsets().to_vec())
        };
        let after_first = raw(&d);

        for _ in 0..3 {
            let log = open(&d);
            assert_eq!(log.recovered_bytes(), 0);
            assert_eq!((log.head(), log.len_bytes(), log.offsets().to_vec()), first);
            drop(log);
            assert_eq!(raw(&d), after_first);
        }
    }

    #[test]
    fn rebuilt_indexes_match_a_fresh_scan() {
        let d = TempDir::new("indexes");
        let mut log = open(&d);
        for i in 0..20u64 {
            log.append(&i.to_le_bytes(), &format!("k{i}"), i as i64)
                .unwrap();
        }
        let offsets_before = log.offsets().to_vec();
        let head_before = log.head();

        log.rebuild_indexes().unwrap();
        assert_eq!(log.offsets(), &offsets_before[..]);
        assert_eq!(log.head(), head_before);
        // dedup still works after a rebuild
        assert_eq!(log.append(&5u64.to_le_bytes(), "k5", 0).unwrap(), 6);
        assert_eq!(log.head(), 20);

        // and a completely independent scan agrees
        let file = File::open(d.path().join(EPISODES_FILENAME)).unwrap();
        let scan = scan_frames(&file, log.len_bytes()).unwrap();
        assert_eq!(scan.offsets, offsets_before);
        assert_eq!(scan.good_end, log.len_bytes());
    }

    #[test]
    fn a_torn_tail_that_looks_like_a_sequence_break_at_eof_truncates() {
        // A frame that is complete and CRC-valid but has the wrong seq, sitting
        // at exact EOF, is a torn tail (the spec's rule), not a SequenceBreak.
        let d = TempDir::new("seq_at_eof");
        let full = three_records(&d);
        let mut bytes = full.clone();
        let last = *frame_offsets(&bytes).last().unwrap();
        let body_start = last + PREFIX_LEN;
        bytes[body_start..body_start + 8].copy_from_slice(&42u64.to_le_bytes());
        fix_body_crc(&mut bytes, last);
        write_raw(&d, &bytes);

        let log = open(&d);
        assert_eq!(log.head(), 2);
        assert_eq!(log.len_bytes(), last as u64);
    }

    // Spec case 37. A crash that is not a process kill can persist a write's
    // size extension without its data pages, leaving the tail reading as
    // zeros. That is a crash artifact, not damage: the log must open.
    #[test]
    fn zero_filled_tail_recovers_with_no_loss() {
        // Runs shorter than PREFIX_LEN take the "fewer than a prefix remains"
        // path; PREFIX_LEN and above take the implausible-body_len-but-all-zeros
        // path. Both must end in the same place, so the loop straddles the
        // boundary rather than stopping at version 1's 8.
        for zeros in [1usize, 7, PREFIX_LEN - 1, PREFIX_LEN, 18, 64, 4096] {
            let d = TempDir::new(&format!("zerotail_{zeros}"));
            let full = three_records(&d);
            let mut bytes = full.clone();
            bytes.extend(std::iter::repeat_n(0u8, zeros));
            write_raw(&d, &bytes);

            let log = open(&d);
            assert_eq!(log.head(), 3, "zeros={zeros}");
            assert_eq!(log.len_bytes(), full.len() as u64, "zeros={zeros}");
            assert_eq!(log.recovered_bytes(), zeros as u64, "zeros={zeros}");
            let all = log.records_since(0).unwrap();
            assert_eq!(all.iter().map(|r| r.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
            for (i, r) in all.iter().enumerate() {
                assert_eq!(
                    r.payload,
                    format!("payload-{i}").as_bytes(),
                    "zeros={zeros}"
                );
            }
            drop(log);

            // the zeros are gone from the file, and reopening is a no-op
            assert_eq!(raw(&d), full, "zeros={zeros}");
            let log = open(&d);
            assert_eq!(log.recovered_bytes(), 0, "zeros={zeros}");
            assert_eq!(log.head(), 3, "zeros={zeros}");
        }
    }

    // Spec case 37, on an empty log: nothing to lose, but it must still open.
    #[test]
    fn zero_filled_tail_on_an_empty_log_recovers() {
        let d = TempDir::new("zerotail_empty");
        open(&d);
        let mut bytes = raw(&d);
        assert_eq!(bytes.len(), HEADER_LEN);
        bytes.extend(std::iter::repeat_n(0u8, 64));
        write_raw(&d, &bytes);

        let log = open(&d);
        assert_eq!(log.head(), 0);
        assert_eq!(log.len_bytes(), HEADER_LEN as u64);
        assert_eq!(log.recovered_bytes(), 64);
    }

    // Spec case 37, continued: after a zero-filled tail is truncated the next
    // append continues the sequence with no gap (the case-29 invariant).
    #[test]
    fn append_after_a_zero_filled_tail_continues_the_sequence() {
        let d = TempDir::new("zerotail_append");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend(std::iter::repeat_n(0u8, 64));
        write_raw(&d, &bytes);

        let mut log = open(&d);
        assert_eq!(log.append(b"next", "", 9).unwrap(), 4);
        drop(log);
        let log = open(&d);
        assert_eq!(log.head(), 4);
        assert_eq!(
            log.records_since(0)
                .unwrap()
                .iter()
                .map(|r| r.seq)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(log.read(4).unwrap().payload, b"next");
    }

    // Spec case 38. Zeros only mean "crash artifact" when they run to EOF.
    // Zeros with real frame bytes after them are damage: refuse to open, and
    // do not truncate.
    #[test]
    fn zeros_followed_by_real_frame_bytes_are_corruption() {
        // (a) a zero run written over the start of a middle frame, with a
        //     whole valid frame still after it
        let d = TempDir::new("zeros_mid");
        let full = three_records(&d);
        let frame2 = {
            let bl =
                u32::from_le_bytes(full[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
            HEADER_LEN + PREFIX_LEN + bl
        };
        let mut bytes = full.clone();
        for b in bytes[frame2..frame2 + 16].iter_mut() {
            *b = 0;
        }
        write_raw(&d, &bytes);
        let err = reopen_err(&d);
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
        assert_eq!(
            raw(&d).len(),
            bytes.len(),
            "a corrupt log must not be truncated"
        );

        // (b) a zero run appended, then non-zero bytes after it, none of which
        //     verify as a frame. This used to refuse, on the rule "any non-zero
        //     byte after the damage is data, and data proves durability". Case
        //     49 is what that rule cost: a write that half lands leaves its own
        //     non-zero bytes behind, and reading them as durable data made the
        //     log refuse forever over its own interrupted append. Nothing here
        //     was ever acknowledged -- all three episodes are still readable --
        //     so the tail is truncated and kept beside the log as evidence.
        let d = TempDir::new("zeros_then_garbage");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend(std::iter::repeat_n(0u8, 64));
        bytes.extend_from_slice(b"not zero");
        write_raw(&d, &bytes);
        let log = open(&d);
        assert_eq!(log.head(), 3);
        assert_eq!(log.len_bytes(), full.len() as u64);
        let kept = log.discarded_tail_path().expect("the tail must be kept");
        assert_eq!(std::fs::read(kept).unwrap(), bytes[full.len()..]);

        // (c) a single non-zero byte at the very end: same reasoning.
        let d = TempDir::new("zeros_then_one_byte");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend(std::iter::repeat_n(0u8, 64));
        bytes.push(1);
        write_raw(&d, &bytes);
        let log = open(&d);
        assert_eq!(log.head(), 3);
        assert_eq!(log.len_bytes(), full.len() as u64);
    }

    // Spec case 38, the other direction. Non-zero garbage at a frame start that
    // verifies as nothing is a tail we never acknowledged, so it is truncated
    // and preserved -- but a whole frame after it still makes it corruption,
    // which is what the first half of the test above pins down.
    #[test]
    fn a_non_zero_garbage_tail_is_truncated_and_kept() {
        let d = TempDir::new("garbage_tail");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend_from_slice(&[0xab; 64]);
        write_raw(&d, &bytes);
        let log = open(&d);
        assert_eq!(log.head(), 3, "every acknowledged episode survives");
        assert_eq!(log.len_bytes(), full.len() as u64);
        assert_eq!(log.recovered_bytes(), 64);
        let kept = log.discarded_tail_path().expect("the tail must be kept");
        assert_eq!(std::fs::read(kept).unwrap(), vec![0xabu8; 64]);
    }

    /// An all-`0xff` run is the one garbage pattern that is *not* reached by
    /// the rule above: `len_crc32(0xffffffff)` happens to be `0xffffffff`, so
    /// the length vouches for itself and the scan moves on to the next test —
    /// a checksum-valid length outside `18..=MAX_BODY`, which says the file was
    /// written by something that is not this code. That stays a refusal, and it
    /// is a different rule from case 49's. Pinned so the coincidence is on the
    /// record rather than rediscovered as a surprise.
    #[test]
    fn a_self_validating_length_that_is_out_of_range_still_refuses() {
        let d = TempDir::new("ff_tail");
        let full = three_records(&d);
        assert_eq!(frame::len_crc32(u32::MAX), u32::MAX, "the coincidence holds");
        let mut bytes = full.clone();
        bytes.extend_from_slice(&[0xff; 64]);
        let err = assert_refuses_and_leaves_the_file_alone(&d, &bytes, "0xff tail");
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
    }

    /// A second recovery must not clobber the first tail it filed away.
    #[test]
    fn a_second_discarded_tail_gets_its_own_name() {
        let d = TempDir::new("two_tails");
        let full = three_records(&d);
        let mut first = full.clone();
        first.extend_from_slice(&[0xaa; 32]);
        write_raw(&d, &first);
        let a = open(&d).discarded_tail_path().unwrap().to_path_buf();

        let mut second = full.clone();
        second.extend_from_slice(&[0xbb; 32]);
        write_raw(&d, &second);
        let b = open(&d).discarded_tail_path().unwrap().to_path_buf();

        assert_ne!(a, b, "the second tail must not overwrite the first");
        assert_eq!(std::fs::read(&a).unwrap(), vec![0xaau8; 32]);
        assert_eq!(std::fs::read(&b).unwrap(), vec![0xbbu8; 32]);
    }

    #[test]
    fn a_crc_break_at_exact_eof_truncates() {
        let d = TempDir::new("crc_at_eof");
        let full = three_records(&d);
        let mut bytes = full.clone();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        write_raw(&d, &bytes);

        let log = open(&d);
        assert_eq!(log.head(), 2);
    }

    // ---- added after the adversarial audit ------------------------------
    // Each test below is a defect that shipped and that the first 39 cases
    // missed. They are named with what they caught, because a case whose
    // origin is forgotten is a case someone later deletes as redundant.

    /// Spec case 40, the audit's finding 1 — the whole reason for version 2.
    ///
    /// A single flipped bit in a frame's length field that lands inside
    /// `18..=MAX_BODY` makes the frame claim more bytes than the file holds.
    /// Version 1 believed the number, called that a torn tail, and truncated:
    /// measured on a healthy three-episode log, one bit destroyed all three
    /// acknowledged episodes and cut the file from 143 bytes to 32, raising
    /// nothing. The old suite missed it because both of its length-mutation
    /// tests used *implausible* values, which take a different branch.
    ///
    /// The mutation is a bit flip, so `len_crc` is deliberately **not**
    /// repaired: that is precisely what makes the damage detectable.
    #[test]
    fn a_plausible_but_oversized_body_len_is_corruption_never_truncation() {
        let d = TempDir::new("case40");
        let full = three_records(&d);
        let offsets = frame_offsets(&full);
        assert_eq!(offsets.len(), 3);

        // `| 0x4000` is one bit, keeps the value inside the plausible range, and
        // puts the claimed end far past a 150-byte file.
        let first = offsets[0];
        let middle = offsets[1];
        let mut cases: Vec<(&str, usize, u32)> = vec![
            ("first frame", first, body_len_at(&full, first) | 0x4000),
            ("middle frame", middle, body_len_at(&full, middle) | 0x4000),
        ];
        // ...and a length stretched to end *exactly* at EOF, which under
        // version 1 was the nastiest shape of all: plausible, ending on the
        // boundary, so the body CRC failed at exact EOF and the frame was
        // truncated as a torn tail.
        cases.push((
            "stretched to EOF",
            first,
            (full.len() - (first + PREFIX_LEN)) as u32,
        ));

        for (label, at, claimed) in cases {
            assert!(
                frame::body_len_is_plausible(claimed),
                "{label}: the test's own mutation must be inside 18..=MAX_BODY"
            );
            assert_ne!(claimed, body_len_at(&full, at), "{label}: not a mutation");

            let mut bytes = full.clone();
            damage_body_len(&mut bytes, at, claimed);

            let err = assert_refuses_and_leaves_the_file_alone(&d, &bytes, label);
            assert!(
                matches!(err, Error::CorruptFrame { .. }),
                "{label}: got {err:?}"
            );

            // Nothing was lost: put the four bytes back and every episode is
            // still there, in order, byte-identical.
            write_raw(&d, &full);
            let log = open(&d);
            assert_eq!(log.head(), 3, "{label}");
            for (i, r) in log.records_since(0).unwrap().iter().enumerate() {
                assert_eq!(r.seq, i as u64 + 1, "{label}");
                assert_eq!(r.payload, format!("payload-{i}").as_bytes(), "{label}");
            }
        }
    }

    /// Spec case 40, on the only frame where the old code's mistake was not
    /// merely wrong but total: damage the *first* frame's length and the whole
    /// file went with it.
    #[test]
    fn a_damaged_first_length_does_not_take_the_whole_log_with_it() {
        let d = TempDir::new("case40_total");
        let full = three_records(&d);
        let mut bytes = full.clone();
        damage_body_len(
            &mut bytes,
            HEADER_LEN,
            body_len_at(&full, HEADER_LEN) | 0x4000,
        );
        write_raw(&d, &bytes);

        assert!(matches!(reopen_err(&d), Error::CorruptFrame { .. }));
        assert_eq!(
            raw(&d).len(),
            full.len(),
            "the file must not be cut back to the header"
        );
    }

    /// Spec case 41 — `len_crc` is checked before `body_len` is used, and the
    /// two verdicts it can reach are separated by evidence, not by guesswork.
    #[test]
    fn a_length_that_fails_its_own_checksum_is_classified_on_the_evidence() {
        let d = TempDir::new("case41");
        let full = three_records(&d);
        let offsets = frame_offsets(&full);

        // (a) the *checksum* is the flipped field and `body_len` is intact.
        //     `body_len` is not trusted on its own here -- it is trusted
        //     because the body it names checksums and carries the sequence
        //     number expected next, which a wrong length cannot fake. So the
        //     frame is whole, only these four bytes are damaged, and recovery
        //     rewrites them: no episode is lost and nothing is refused.
        let mut bytes = full.clone();
        let at = offsets[1];
        bytes[at + 4] ^= 0x01;
        write_raw(&d, &bytes);
        let log = open(&d);
        assert_eq!(log.head(), 3, "every episode still readable");
        assert_eq!(log.repaired_lengths(), 1, "the damaged field was rewritten");
        assert_eq!(log.len_bytes(), full.len() as u64, "nothing was truncated");
        drop(log);

        // The repair is on disk, not merely in memory. That is the point: the
        // evidence identifying the true length is available now and might not
        // be after another append, so a second open must find nothing to fix.
        let log = open(&d);
        assert_eq!(log.head(), 3);
        assert_eq!(log.repaired_lengths(), 0, "already correct on disk");
        assert_eq!(raw(&d), full, "the file is byte-identical to the healthy one");
        drop(log);

        // (a2) but a damaged length on a *middle* frame is still corruption.
        //      Only the last frame's extent is implied by EOF, so here nothing
        //      identifies the true length and there is no repair to make.
        let d2 = TempDir::new("case41_middle");
        let full2 = three_records(&d2);
        let mut bytes = full2.clone();
        let at = frame_offsets(&full2)[1];
        damage_body_len(&mut bytes, at, 7);
        let err = assert_refuses_and_leaves_the_file_alone(&d2, &bytes, "middle body_len");
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");

        // (b) zeros to EOF -> a zero-filled tail, which must keep working after
        //     the format change. An all-zero prefix fails its length checksum,
        //     so this is now reached through the new branch rather than through
        //     the old implausible-length one.
        let mut bytes = full.clone();
        for b in bytes[offsets[2]..].iter_mut() {
            *b = 0;
        }
        write_raw(&d, &bytes);
        let log = open(&d);
        assert_eq!(log.head(), 2);
        assert_eq!(log.len_bytes(), offsets[2] as u64);
        assert_eq!(log.recovered_bytes(), (full.len() - offsets[2]) as u64);

        // (c) a checksum-*valid* length we could never have written is not a
        //     torn tail either: the file was built by something else.
        let d = TempDir::new("case41_forged");
        let full = three_records(&d);
        let mut bytes = full.clone();
        let at = frame_offsets(&full)[1];
        damage_body_len(&mut bytes, at, MAX_BODY + 1);
        bytes[at + 4..at + 8].copy_from_slice(&frame::len_crc32(MAX_BODY + 1).to_le_bytes());
        let err = assert_refuses_and_leaves_the_file_alone(&d, &bytes, "forged oversize length");
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
    }

    /// Version 1 is not readable and is not migrated. A version-1 header is
    /// otherwise perfectly well formed, so this is the one shape that would
    /// quietly misparse if the version check were ever relaxed.
    #[test]
    fn a_version_one_log_is_refused() {
        let d = TempDir::new("v1");
        open(&d);
        let mut bytes = raw(&d);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        let err = assert_refuses_and_leaves_the_file_alone(&d, &bytes, "version 1");
        assert!(matches!(err, Error::UnsupportedVersion(1)), "got {err:?}");
    }

    /// Spec case 43, the audit's finding 3. A checkpoint ahead of `head` is
    /// proof that acknowledged episodes were lost, and open is the one moment
    /// that proof exists. Checking it lazily left the log opening "cleanly" and
    /// the consumer wedged: unable to read and unable to reset, both raising.
    #[test]
    fn a_checkpoint_ahead_of_head_fails_at_open() {
        use crate::checkpoint::{sidecar_path, CheckpointStore};

        let d = TempDir::new("case43");
        let p = log_path(&d);
        three_records(&d);

        // The realistic route there: the consumer acknowledged seq 3, then a
        // torn tail took frame 3 away.
        {
            let mut log = open(&d);
            log.set_checkpoint("graph", 3).unwrap();
        }
        let full = raw(&d);
        write_raw(&d, &full[..full.len() - 4]);

        let err = Log::open(&p).unwrap_err();
        assert!(
            matches!(
                err,
                Error::CheckpointAhead {
                    requested: 3,
                    head: 2
                }
            ),
            "got {err:?}"
        );

        // The documented way out is to remove the sidecar by hand. It has to
        // actually work: a missing sidecar reads as 0, the consumer replays,
        // and DL-007's write-key dedup makes the replay idempotent.
        std::fs::remove_file(sidecar_path(&p)).unwrap();
        let log = Log::open(&p).unwrap();
        assert_eq!(log.head(), 2);
        assert_eq!(log.checkpoint("graph"), 0);
        drop(log);

        // And a sidecar that is merely *ahead* — no damage, nothing else wrong
        // — is refused on its own, with the log intact behind it.
        {
            let mut store = CheckpointStore::load(&p).unwrap();
            store.set("graph", 99).unwrap();
        }
        let err = Log::open(&p).unwrap_err();
        assert!(
            matches!(
                err,
                Error::CheckpointAhead {
                    requested: 99,
                    head: 2
                }
            ),
            "got {err:?}"
        );
    }

    /// Spec case 43's boundary: a checkpoint exactly *at* head is legitimate
    /// and must open. A check that also rejects the legal value is not a check,
    /// it is an outage.
    #[test]
    fn a_checkpoint_exactly_at_head_opens_normally() {
        let d = TempDir::new("case43_boundary");
        three_records(&d);
        {
            let mut log = open(&d);
            log.set_checkpoint("graph", 3).unwrap();
        }
        let log = open(&d);
        assert_eq!(log.head(), 3);
        assert_eq!(log.checkpoint("graph"), 3);
    }

    /// Spec case 42, the audit's finding 2. Seven ways a sidecar can be
    /// unreadable; all seven used to make every acknowledged episode
    /// unreachable, over state that is not the source of truth.
    #[test]
    fn a_damaged_sidecar_never_blocks_the_log() {
        use crate::checkpoint::{sidecar_path, CheckpointStore};

        let clean = {
            let d = TempDir::new("case42_clean");
            let p = log_path(&d);
            three_records(&d);
            let mut store = CheckpointStore::load(&p).unwrap();
            store.set("graph", 2).unwrap();
            std::fs::read(sidecar_path(&p)).unwrap()
        };

        type Damage = Box<dyn Fn(&[u8]) -> Vec<u8>>;
        let damage: Vec<(&str, Damage)> = vec![
            ("zero bytes", Box::new(|_: &[u8]| Vec::new())),
            ("all zeros", Box::new(|c: &[u8]| vec![0u8; c.len()])),
            (
                "truncated by one",
                Box::new(|c: &[u8]| c[..c.len() - 1].to_vec()),
            ),
            (
                "one flipped bit",
                Box::new(|c: &[u8]| {
                    let mut b = c.to_vec();
                    let n = b.len();
                    b[n - 6] ^= 0x01;
                    b
                }),
            ),
            (
                "bad magic",
                Box::new(|c: &[u8]| {
                    let mut b = c.to_vec();
                    b[0] = b'X';
                    b
                }),
            ),
            (
                "absurd count",
                Box::new(|c: &[u8]| {
                    let mut b = c.to_vec();
                    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
                    b
                }),
            ),
            (
                "absurd name_len",
                Box::new(|c: &[u8]| {
                    let mut b = c.to_vec();
                    b[16..18].copy_from_slice(&u16::MAX.to_le_bytes());
                    b
                }),
            ),
        ];

        for (label, make) in damage {
            let d = TempDir::new("case42");
            let p = log_path(&d);
            three_records(&d);
            let broken = make(&clean);
            std::fs::write(sidecar_path(&p), &broken).unwrap();

            let log = Log::open(&p).unwrap_or_else(|e| panic!("{label}: the log must open: {e}"));
            assert_eq!(log.head(), 3, "{label}");
            assert_eq!(log.records_since(0).unwrap().len(), 3, "{label}");
            assert_eq!(log.checkpoint("graph"), 0, "{label}");
            assert!(log.checkpoints_reset(), "{label}: the seam must be told");

            // The evidence is preserved, not deleted.
            let moved = log
                .damaged_checkpoints_path()
                .unwrap_or_else(|| panic!("{label}: no .damaged path"))
                .to_path_buf();
            assert_eq!(std::fs::read(&moved).unwrap(), broken, "{label}");
            assert!(!sidecar_path(&p).exists(), "{label}");
            drop(log);

            // And the log is usable afterwards: a fresh checkpoint round-trips
            // and the next open is clean.
            {
                let mut log = Log::open(&p).unwrap();
                assert!(!log.checkpoints_reset(), "{label}: second open");
                log.set_checkpoint("graph", 3).unwrap();
            }
            let log = Log::open(&p).unwrap();
            assert_eq!(log.checkpoint("graph"), 3, "{label}");
        }
    }

    /// Spec case 42, continued: a second damaged sidecar must not overwrite the
    /// first one's evidence.
    #[test]
    fn a_second_damaged_sidecar_gets_its_own_name() {
        use crate::checkpoint::sidecar_path;

        let d = TempDir::new("case42_unique");
        let p = log_path(&d);
        three_records(&d);

        let mut seen: Vec<PathBuf> = Vec::new();
        for i in 0..3u8 {
            std::fs::write(sidecar_path(&p), [i; 40]).unwrap();
            let log = Log::open(&p).unwrap();
            let moved = log.damaged_checkpoints_path().unwrap().to_path_buf();
            assert!(!seen.contains(&moved), "reused {moved:?}");
            assert_eq!(std::fs::read(&moved).unwrap(), vec![i; 40]);
            seen.push(moved);
        }
        // every earlier one is still on disk, unmodified
        for (i, path) in seen.iter().enumerate() {
            assert_eq!(std::fs::read(path).unwrap(), vec![i as u8; 40]);
        }
    }

    /// Spec case 44, the audit's finding 4 — the most uncomfortable one. Every
    /// durability call in the crate could be deleted and the entire suite
    /// stayed green, 94x faster, including 40 `kill -9` trials. `kill -9`
    /// cannot test `fsync`: the kernel completes the in-flight write and the
    /// page cache outlives the process. So this observes the call.
    ///
    /// Each assertion below pins exactly one `fsync` site. Delete that site and
    /// this test goes red; nothing else does.
    #[test]
    fn every_fsync_site_is_observable() {
        use crate::sync::{dir_syncs, file_syncs};

        let d = TempDir::new("case44");

        // Site 1 of 4: creating the log fsyncs the header, and site 2 of 4:
        // fsyncs the parent directory, or the file itself can vanish on crash.
        let (f0, d0) = (file_syncs(), dir_syncs());
        let mut log = open(&d);
        assert_eq!(
            file_syncs() - f0,
            1,
            "writing the header must fsync the file"
        );
        assert_eq!(
            dir_syncs() - d0,
            1,
            "creating the log must fsync its parent directory"
        );

        // Site 3 of 4: every append fsyncs before it returns. This is the
        // invariant M0 exists for — an append that returned survives kill -9.
        for i in 0..3 {
            let before = file_syncs();
            log.append(format!("p{i}").as_bytes(), "", i).unwrap();
            assert_eq!(
                file_syncs() - before,
                1,
                "append must fsync before returning"
            );
        }

        // A deduped append writes nothing, so it syncs nothing. The counter
        // measures real syncs, not calls that went nowhere.
        log.append(b"x", "k", 1).unwrap();
        let before = file_syncs();
        assert_eq!(log.append(b"x", "k", 2).unwrap(), 4);
        assert_eq!(file_syncs() - before, 0, "a deduped append writes nothing");

        // Site 4 of 4: the sidecar is write-temp, fsync, rename, fsync-dir, so
        // a crash never leaves a half-written checkpoint.
        let (f, dd) = (file_syncs(), dir_syncs());
        log.set_checkpoint("graph", 1).unwrap();
        assert_eq!(
            file_syncs() - f,
            1,
            "the sidecar temp file must be fsynced before the rename"
        );
        assert_eq!(
            dir_syncs() - dd,
            1,
            "the directory must be fsynced after the sidecar rename"
        );
    }

    /// Spec case 44, continued: recovery's truncation is durable too. A torn
    /// tail that is truncated but not fsynced can come back.
    #[test]
    fn truncating_a_torn_tail_is_fsynced() {
        use crate::sync::file_syncs;

        let d = TempDir::new("case44_truncate");
        let full = three_records(&d);
        write_raw(&d, &full[..full.len() - 3]);

        let before = file_syncs();
        let log = open(&d);
        assert!(log.recovered_bytes() > 0);
        assert_eq!(
            file_syncs() - before,
            1,
            "a truncation must be fsynced before open returns"
        );
    }

    /// Spec case 45, the audit's finding 5. The magic check sat behind the
    /// truncation step and never ran for files under 32 bytes, so any short
    /// file at the log's path — a note, a stray text file — was silently
    /// overwritten with a log header.
    #[test]
    fn a_small_non_log_file_is_refused_rather_than_overwritten() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("a note", b"remember to call the bank\n".to_vec()), // 26 bytes
            ("one wrong byte", b"x".to_vec()),
            ("all zeros", vec![0u8; 31]),
            ("nearly right", b"OMEGALOh".to_vec()),
            ("right magic, wrong length", b"OMEGALOG".to_vec()),
        ];

        for (label, bytes) in cases {
            let seen = std::cmp::min(bytes.len(), frame::MAGIC.len());
            let d = TempDir::new("case45");
            write_raw(&d, &bytes);
            if bytes[..seen] == frame::MAGIC[..seen] {
                // a genuine torn header: truncated and rewritten, no error
                assert_eq!(open(&d).head(), 0, "{label}");
                continue;
            }
            let err = reopen_err(&d);
            assert!(matches!(err, Error::NotAnOmegaLog), "{label}: got {err:?}");
            assert_eq!(raw(&d), bytes, "{label}: the file must be byte-identical");
        }

        // 26 bytes is the spec's own example. Spell it out on its own, and
        // check it twice: a refusal that damages the file on the *second* try
        // is still a data-loss bug.
        let d = TempDir::new("case45_note");
        let note = b"remember to call the bank\n".to_vec();
        assert_eq!(note.len(), 26);
        write_raw(&d, &note);
        for _ in 0..2 {
            assert!(matches!(reopen_err(&d), Error::NotAnOmegaLog));
            assert_eq!(raw(&d), note);
        }
    }

    /// Spec case 45's other half, and the reason the ordering is subtle: a file
    /// that *is* a torn header still gets truncated and rewritten. The magic
    /// check must gate the destructive step without disabling it.
    #[test]
    fn a_genuine_torn_header_is_still_rewritten() {
        let header = frame::encode_header();
        for n in 1..HEADER_LEN {
            let d = TempDir::new("case45_torn");
            write_raw(&d, &header[..n]);
            let mut log = open(&d);
            assert_eq!(log.head(), 0, "n={n}");
            assert_eq!(log.append(b"first", "", 1).unwrap(), 1, "n={n}");
        }
    }

    /// Spec case 47, the audit's finding 7. The old suite tested garbage runs of
    /// 1, 2 and 7 bytes but never 8, so the exact point where behaviour changes
    /// was untested. The boundary moved to 12 with the version-2 prefix, so it
    /// is pinned here on both sides.
    #[test]
    fn the_classification_boundary_is_the_prefix_length() {
        let d = TempDir::new("case47");
        let full = three_records(&d);

        // Shorter than a prefix: there is not enough there to be a frame at
        // all, so it is a torn tail whatever the bytes say.
        for n in 1..PREFIX_LEN {
            let mut bytes = full.clone();
            bytes.extend(std::iter::repeat_n(0xabu8, n));
            write_raw(&d, &bytes);
            let log = open(&d);
            assert_eq!(log.head(), 3, "n={n}");
            assert_eq!(log.len_bytes(), full.len() as u64, "n={n}");
            assert_eq!(log.recovered_bytes(), n as u64, "n={n}");
        }

        // At the prefix length and past it: there is enough to read a length
        // and its checksum, and the checksum fails. That used to be the
        // boundary between torn tail and damage, on the bytes' zero-ness
        // alone. Case 49 retired that test -- a run of 0xab verifies as no
        // frame, so nothing in it was ever acknowledged, and it is truncated
        // rather than allowed to brick the log. The bytes are kept beside it.
        for n in [PREFIX_LEN, PREFIX_LEN + 1, PREFIX_LEN * 2] {
            let mut bytes = full.clone();
            bytes.extend(std::iter::repeat_n(0xabu8, n));
            write_raw(&d, &bytes);
            let log = open(&d);
            assert_eq!(log.head(), 3, "n={n}");
            assert_eq!(log.len_bytes(), full.len() as u64, "n={n}");
            assert_eq!(log.recovered_bytes(), n as u64, "n={n}");
            let kept = log.discarded_tail_path().unwrap_or_else(|| panic!("n={n}"));
            assert_eq!(std::fs::read(kept).unwrap(), vec![0xabu8; n], "n={n}");
        }

        // The zero-filled version of the same lengths takes the other branch at
        // every one of them (spec case 39 alongside case 27), which is why the
        // two are spelled out separately rather than left to the reader.
        for n in 1..=PREFIX_LEN * 2 {
            let mut bytes = full.clone();
            bytes.extend(std::iter::repeat_n(0u8, n));
            write_raw(&d, &bytes);
            let log = open(&d);
            assert_eq!(log.head(), 3, "zeros={n}");
            assert_eq!(log.len_bytes(), full.len() as u64, "zeros={n}");
        }
    }

    /// Spec case 48. Two records sharing a dedup key resolve to the lower seq:
    /// the first write of a key is the one it refers to. Unpinned until now —
    /// reversing it broke no test.
    #[test]
    fn the_first_write_of_a_dedup_key_is_the_one_it_refers_to() {
        let d = TempDir::new("case48");
        let mut bytes = {
            open(&d);
            raw(&d)
        };
        // A well-formed log never repeats a key, so this one is built by hand.
        bytes.extend(frame::encode(1, 1, "k", b"first").unwrap());
        bytes.extend(frame::encode(2, 2, "k", b"second").unwrap());
        write_raw(&d, &bytes);

        let mut log = open(&d);
        assert_eq!(log.head(), 2);

        // Re-appending the *first* payload dedupes to seq 1 and writes nothing.
        // Under last-key-wins this would compare against seq 2 and conflict, so
        // the two rules are distinguishable here and nowhere else.
        assert_eq!(log.append(b"first", "k", 3).unwrap(), 1);
        assert_eq!(log.head(), 2);
        let err = log.append(b"second", "k", 4).unwrap_err();
        assert!(
            matches!(
                err,
                Error::WriteKeyConflict {
                    existing_seq: 1,
                    ..
                }
            ),
            "got {err:?}"
        );

        // A rebuild agrees, and so does a reopen: the index is a cache, and a
        // cache that disagrees with a rescan is the bug this pins.
        log.rebuild_indexes().unwrap();
        assert_eq!(log.append(b"first", "k", 5).unwrap(), 1);
        drop(log);
        let mut log = open(&d);
        assert_eq!(log.append(b"first", "k", 6).unwrap(), 1);
    }
}
