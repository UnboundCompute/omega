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

use crate::checkpoint::CheckpointStore;
use crate::frame::{self, HEADER_LEN, PREFIX_LEN};
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
    /// True when recovery truncated a torn tail on this open. Diagnostic only.
    recovered_bytes: u64,
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
        };
        log.recover()?;
        crate::checkpoint::clear_stale_temp(path);
        log.checkpoints = CheckpointStore::load(path)?;
        Ok(log)
    }

    /// Recovery, run on every open. See M0_SPEC.md "Recovery".
    fn recover(&mut self) -> Result<(), Error> {
        let mut len = self.file.metadata()?.len();

        // Steps 1-3: missing / 0 bytes / torn header. No record can exist yet,
        // so truncate to 0 and (re)write the header.
        if len < HEADER_LEN as u64 {
            self.file.set_len(0)?;
            self.file.write_all_at(&frame::encode_header(), 0)?;
            self.file.sync_all()?;
            // The file may be brand new, so the directory entry needs fsync too
            // or the file itself can vanish on crash.
            fsync_dir(&self.dir)?;
            len = HEADER_LEN as u64;
        } else {
            // Step 4: magic and version. Both refuse to open.
            let mut header = [0u8; HEADER_LEN];
            self.file.read_exact_at(&mut header, 0)?;
            frame::validate_header(&header)?;
        }

        // Step 5: scan the frames.
        let scan = scan_frames(&self.file, len)?;

        // Step 6: if anything was truncated, set_len and fsync.
        if scan.good_end < len {
            self.recovered_bytes = len - scan.good_end;
            self.file.set_len(scan.good_end)?;
            self.file.sync_all()?;
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
        if let Err(e) = self.file.sync_all() {
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
        let _ = self.file.sync_all();
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

/// `fsync` a directory so a newly created or renamed entry in it is durable.
pub(crate) fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

/// Read one complete frame at `offset`, verifying its CRC.
pub fn read_record_at(file: &File, offset: u64) -> Result<frame::Record, Error> {
    let mut prefix = [0u8; PREFIX_LEN];
    file.read_exact_at(&mut prefix, offset)?;
    let (body_len, crc) = frame::decode_prefix(&prefix);
    if !frame::body_len_is_plausible(body_len) {
        return Err(Error::CorruptFrame {
            offset,
            detail: format!("body_len {body_len} out of range"),
        });
    }
    let mut body = vec![0u8; body_len as usize];
    file.read_exact_at(&mut body, offset + PREFIX_LEN as u64)?;
    if frame::crc32(&body) != crc {
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
            break; // torn tail: fewer than 8 bytes remain
        }

        let mut prefix = [0u8; PREFIX_LEN];
        file.read_exact_at(&mut prefix, offset)?;
        let (body_len, crc) = frame::decode_prefix(&prefix);

        // An implausible body_len is one we could never have written. Whether
        // that is a crash artifact or damage depends on what follows it:
        //
        //   * all zeros to EOF -> a crash that persisted a write's size
        //     extension without its data pages. A frame header we wrote is
        //     never all zeros, so this is a zero-filled tail: truncate.
        //   * anything else -> non-zero garbage at a frame start is damage,
        //     and truncating it would silently discard real episodes.
        //
        // The zero clause is not hypothetical: classifying it as corruption
        // made a log that had survived exactly the crash this milestone
        // promises to survive refuse to open, losing every episode in it.
        if !frame::body_len_is_plausible(body_len) {
            if is_zero_to_eof(file, offset, file_len)? {
                break; // zero-filled tail
            }
            return Err(Error::CorruptFrame {
                offset,
                detail: format!(
                    "body_len {body_len} out of range ({}..={}), and the bytes here are not a zero-filled tail",
                    frame::FIXED_BODY_LEN,
                    frame::MAX_BODY
                ),
            });
        }

        if remaining - (PREFIX_LEN as u64) < body_len as u64 {
            break; // torn tail: the body did not all make it
        }

        let frame_end = offset + PREFIX_LEN as u64 + body_len as u64;
        let ends_at_eof = frame_end == file_len;

        let mut body = vec![0u8; body_len as usize];
        file.read_exact_at(&mut body, offset + PREFIX_LEN as u64)?;

        if frame::crc32(&body) != crc {
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

    fn open(dir: &TempDir) -> Log {
        Log::open(&dir.path().join(EPISODES_FILENAME)).unwrap()
    }

    fn raw(dir: &TempDir) -> Vec<u8> {
        std::fs::read(dir.path().join(EPISODES_FILENAME)).unwrap()
    }

    fn write_raw(dir: &TempDir, bytes: &[u8]) {
        let mut f = File::create(dir.path().join(EPISODES_FILENAME)).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
    }

    fn reopen_err(dir: &TempDir) -> Error {
        Log::open(&dir.path().join(EPISODES_FILENAME)).unwrap_err()
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
                assert_eq!(log.append(format!("p{i}").as_bytes(), "", i as i64).unwrap(), i + 1);
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
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0, 0, 0],
            vec![0xff, 0xfe, 0xfd],
            big.clone(),
        ];
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
        assert_eq!(std::fs::read(d.path().join(EPISODES_FILENAME)).unwrap().len(), bytes.len());
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
        let body_len =
            u32::from_le_bytes(bytes[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
        bytes[body_start..body_start + 8].copy_from_slice(&7u64.to_le_bytes());
        let crc = frame::crc32(&bytes[body_start..body_start + body_len]);
        bytes[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&crc.to_le_bytes());
        write_raw(&d, &bytes);

        let err = reopen_err(&d);
        assert!(matches!(err, Error::SequenceBreak { expected: 1, found: 7, .. }), "got {err:?}");
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
        assert!(matches!(log.append(b"x", &long_key, 1), Err(Error::TooLarge(_))));
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
        assert!(matches!(err, Error::WriteKeyConflict { existing_seq: 1, .. }), "got {err:?}");
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
            Err(Error::CheckpointAhead { requested: 2, head: 1 })
        ));
        assert!(log.range_since(1).is_ok());
        // seq 0 is reserved: it is never a record, and asking for it is a
        // caller mistake rather than "you are ahead of the log"
        assert!(matches!(log.read(0), Err(Error::InvalidArgument(_))));
        assert!(matches!(
            log.read(2),
            Err(Error::CheckpointAhead { requested: 2, head: 1 })
        ));
        // and on an empty log, since=1 is already ahead
        let d2 = TempDir::new("ahead2");
        let log2 = open(&d2);
        assert!(matches!(log2.range_since(1), Err(Error::CheckpointAhead { .. })));
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
        let missing = d.path().join("no").join("such").join("dir").join("episodes.log");
        let err = Log::open(&missing).unwrap_err();
        assert!(matches!(err, Error::Io(_)), "got {err:?}");
    }

    // ---- yellow: torn tails and recovery --------------------------------

    fn three_records(d: &TempDir) -> Vec<u8> {
        {
            let mut log = open(d);
            for i in 0..3 {
                log.append(format!("payload-{i}").as_bytes(), "", i).unwrap();
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
        // keep two whole frames plus the 8-byte prefix of the third
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
    fn torn_tail_of_one_and_seven_bytes() {
        for extra in [1usize, 7] {
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

    #[test]
    fn torn_header_recovers_as_an_empty_log() {
        for n in [0usize, 1, 17, 31] {
            let d = TempDir::new(&format!("tornhdr_{n}"));
            write_raw(&d, &vec![0u8; n]);
            let log = open(&d);
            assert_eq!(log.head(), 0, "n={n}");
            assert_eq!(log.len_bytes(), HEADER_LEN as u64, "n={n}");
            drop(log);
            assert_eq!(&raw(&d)[..8], b"OMEGALOG");
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
            log.append(&i.to_le_bytes(), &format!("k{i}"), i as i64).unwrap();
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
        // find the last frame's start
        let mut start = HEADER_LEN;
        let mut last = start;
        while start < bytes.len() {
            let bl = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
            last = start;
            start += PREFIX_LEN + bl;
        }
        let body_start = last + PREFIX_LEN;
        let body_len =
            u32::from_le_bytes(bytes[last..last + 4].try_into().unwrap()) as usize;
        bytes[body_start..body_start + 8].copy_from_slice(&42u64.to_le_bytes());
        let crc = frame::crc32(&bytes[body_start..body_start + body_len]);
        bytes[last + 4..last + 8].copy_from_slice(&crc.to_le_bytes());
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
        // 1 and 7 bytes take the "fewer than 8 bytes remain" path; 8, 18 and 64
        // take the new implausible-body_len-but-all-zeros path. Both must end
        // in the same place.
        for zeros in [1usize, 7, 8, 18, 64, 4096] {
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
                assert_eq!(r.payload, format!("payload-{i}").as_bytes(), "zeros={zeros}");
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
            log.records_since(0).unwrap().iter().map(|r| r.seq).collect::<Vec<_>>(),
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
            let bl = u32::from_le_bytes(full[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
            HEADER_LEN + PREFIX_LEN + bl
        };
        let mut bytes = full.clone();
        for b in bytes[frame2..frame2 + 16].iter_mut() {
            *b = 0;
        }
        write_raw(&d, &bytes);
        let err = reopen_err(&d);
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
        assert_eq!(raw(&d).len(), bytes.len(), "a corrupt log must not be truncated");

        // (b) a zero run appended, then non-zero bytes after it
        let d = TempDir::new("zeros_then_garbage");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend(std::iter::repeat_n(0u8, 64));
        bytes.extend_from_slice(b"not zero");
        write_raw(&d, &bytes);
        let err = reopen_err(&d);
        assert!(matches!(err, Error::CorruptFrame { .. }), "got {err:?}");
        assert_eq!(raw(&d).len(), bytes.len());

        // (c) even a single non-zero byte at the very end is enough
        let d = TempDir::new("zeros_then_one_byte");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend(std::iter::repeat_n(0u8, 64));
        bytes.push(1);
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::CorruptFrame { .. }));
    }

    // Spec case 38, the other direction: non-zero garbage at a frame start is
    // damage even when it runs to EOF, because we never wrote it.
    #[test]
    fn non_zero_garbage_tail_is_still_corruption() {
        let d = TempDir::new("garbage_tail");
        let full = three_records(&d);
        let mut bytes = full.clone();
        bytes.extend_from_slice(&[0xff; 64]); // body_len = 0xffffffff, > MAX_BODY
        write_raw(&d, &bytes);
        assert!(matches!(reopen_err(&d), Error::CorruptFrame { .. }));
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
}
