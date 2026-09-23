//! Named consumer checkpoints, stored beside the log.
//!
//! Checkpoints are **not** episodes: they are mutable derived state and never
//! enter the log. They are replaced atomically — write-temp, fsync, rename,
//! fsync-dir — so a crash never leaves a half-written checkpoint.
//!
//! The spec fixes the semantics (`unset reads as 0`, `seq > head()` is an
//! error) but not the sidecar's byte layout, so this is the log's own layout
//! scaled down:
//!
//! ```text
//!   magic    8 bytes   b"OMEGACKP"
//!   version  u32       = 1
//!   count    u32
//!   entries, repeated:
//!     name_len u16     1..=MAX_KEY
//!     name     name_len bytes, UTF-8
//!     seq      u64
//!   crc32    u32       CRC-32/ISO-HDLC over everything above
//! ```

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::frame::{crc32, MAX_KEY};
use crate::log::fsync_dir;
use crate::Error;

const CKPT_MAGIC: [u8; 8] = *b"OMEGACKP";
const CKPT_VERSION: u32 = 1;
const CKPT_HEADER_LEN: usize = 16; // magic + version + count

/// Sidecar path for a log at `log_path`: `<log_path>.checkpoints`.
pub fn sidecar_path(log_path: &Path) -> PathBuf {
    let mut name = log_path.as_os_str().to_os_string();
    name.push(".checkpoints");
    PathBuf::from(name)
}

pub struct CheckpointStore {
    path: PathBuf,
    dir: PathBuf,
    /// BTreeMap so the on-disk order is stable, which makes the file diffable
    /// and the rewrite deterministic.
    map: BTreeMap<String, u64>,
}

impl CheckpointStore {
    /// An empty store for `log_path`, not yet loaded from disk.
    pub fn empty(log_path: &Path) -> CheckpointStore {
        let path = sidecar_path(log_path);
        let dir = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        CheckpointStore {
            path,
            dir,
            map: BTreeMap::new(),
        }
    }

    /// Load the sidecar for `log_path`. A missing file is an empty store.
    pub fn load(log_path: &Path) -> Result<CheckpointStore, Error> {
        let mut store = CheckpointStore::empty(log_path);
        let bytes = match fs::read(&store.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(store),
            Err(e) => return Err(Error::Io(e)),
        };
        store.map = decode(&bytes)?;
        Ok(store)
    }

    /// An unset checkpoint reads as 0, never null.
    pub fn get(&self, name: &str) -> u64 {
        self.map.get(name).copied().unwrap_or(0)
    }

    pub fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    /// Set a checkpoint and replace the sidecar atomically. The caller has
    /// already checked `seq <= head()`.
    pub fn set(&mut self, name: &str, seq: u64) -> Result<(), Error> {
        if name.is_empty() {
            return Err(Error::InvalidArgument(
                "checkpoint name must not be empty".into(),
            ));
        }
        if name.len() > MAX_KEY {
            return Err(Error::TooLarge(format!(
                "checkpoint name is {} bytes, max is {MAX_KEY}",
                name.len()
            )));
        }
        let previous = self.map.insert(name.to_owned(), seq);
        if let Err(e) = self.persist() {
            // Keep memory and disk in agreement: roll the map back.
            match previous {
                Some(v) => self.map.insert(name.to_owned(), v),
                None => self.map.remove(name),
            };
            return Err(e);
        }
        Ok(())
    }

    /// write-temp, fsync, rename, fsync-dir.
    fn persist(&self) -> Result<(), Error> {
        let bytes = encode(&self.map);
        let mut tmp = self.path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);

        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &self.path)?;
        fsync_dir(&self.dir)?;
        Ok(())
    }
}

fn encode(map: &BTreeMap<String, u64>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(CKPT_HEADER_LEN + map.len() * 32 + 4);
    buf.extend_from_slice(&CKPT_MAGIC);
    buf.extend_from_slice(&CKPT_VERSION.to_le_bytes());
    buf.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (name, seq) in map {
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&seq.to_le_bytes());
    }
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

fn decode(bytes: &[u8]) -> Result<BTreeMap<String, u64>, Error> {
    let bad = |detail: String| Error::CorruptFrame { offset: 0, detail };

    if bytes.len() < CKPT_HEADER_LEN + 4 {
        return Err(bad(format!(
            "checkpoint file is {} bytes, too short to be valid",
            bytes.len()
        )));
    }
    if bytes[0..8] != CKPT_MAGIC {
        return Err(bad("checkpoint file has bad magic".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != CKPT_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    // `count` locates the CRC, rather than the CRC being wherever the file
    // happens to end. That is what lets a zero-filled tail be recognised and
    // ignored here, exactly as the log recognises one (M0_SPEC.md "Recovery",
    // the zero-fill clause): the checkpoint file's real length is a fact it
    // carries, not a fact about its size on disk.
    let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let entries_limit = bytes.len() - 4;
    let mut map = BTreeMap::new();
    let mut at = CKPT_HEADER_LEN;
    for _ in 0..count {
        if at + 2 > entries_limit {
            return Err(bad("checkpoint entry truncated".into()));
        }
        let name_len = u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize;
        at += 2;
        if name_len == 0 || name_len > MAX_KEY || at + name_len + 8 > entries_limit {
            return Err(bad(format!("checkpoint entry has bad name_len {name_len}")));
        }
        let name = std::str::from_utf8(&bytes[at..at + name_len])
            .map_err(|e| bad(format!("checkpoint name is not UTF-8: {e}")))?
            .to_owned();
        at += name_len;
        let seq = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        at += 8;
        map.insert(name, seq);
    }

    let stored_crc = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if crc32(&bytes[..at]) != stored_crc {
        return Err(bad("checkpoint file CRC mismatch".into()));
    }

    // Same rule as the log: zeros to EOF are a crash artifact and are ignored;
    // anything non-zero after a complete, CRC-valid file is damage.
    let tail = &bytes[at + 4..];
    if tail.iter().any(|b| *b != 0) {
        return Err(bad(format!(
            "checkpoint file has {} non-zero trailing bytes",
            tail.len()
        )));
    }
    Ok(map)
}

/// Remove a stale `.tmp` sidecar left by a crash between write and rename.
/// Nothing reads it, so this is hygiene rather than correctness.
pub fn clear_stale_temp(log_path: &Path) {
    let mut tmp = sidecar_path(log_path).into_os_string();
    tmp.push(".tmp");
    let _ = fs::remove_file(PathBuf::from(tmp));
}

/// Open the sidecar directly, for tests.
#[cfg(test)]
fn read_sidecar(log_path: &Path) -> Option<Vec<u8>> {
    fs::read(sidecar_path(log_path)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn log_path(d: &TempDir) -> PathBuf {
        d.path().join("episodes.log")
    }

    #[test]
    fn unset_reads_zero() {
        let d = TempDir::new("ckpt_unset");
        let store = CheckpointStore::load(&log_path(&d)).unwrap();
        assert_eq!(store.get("graph"), 0);
        assert_eq!(store.get(""), 0);
        assert!(read_sidecar(&log_path(&d)).is_none());
    }

    #[test]
    fn round_trips_across_reload() {
        let d = TempDir::new("ckpt_round");
        let p = log_path(&d);
        {
            let mut store = CheckpointStore::load(&p).unwrap();
            store.set("graph", 7).unwrap();
            store.set("user-model", 3).unwrap();
            store.set("graph", 9).unwrap();
        }
        let store = CheckpointStore::load(&p).unwrap();
        assert_eq!(store.get("graph"), 9);
        assert_eq!(store.get("user-model"), 3);
        assert_eq!(store.get("nope"), 0);
        assert_eq!(
            store.names(),
            vec!["graph".to_string(), "user-model".to_string()]
        );
    }

    #[test]
    fn no_temp_file_is_left_behind() {
        let d = TempDir::new("ckpt_tmp");
        let p = log_path(&d);
        let mut store = CheckpointStore::load(&p).unwrap();
        store.set("graph", 1).unwrap();
        let mut tmp = sidecar_path(&p).into_os_string();
        tmp.push(".tmp");
        assert!(!PathBuf::from(tmp).exists());
    }

    #[test]
    fn corrupt_sidecar_is_rejected() {
        let d = TempDir::new("ckpt_corrupt");
        let p = log_path(&d);
        let mut store = CheckpointStore::load(&p).unwrap();
        store.set("graph", 1).unwrap();

        let mut bytes = read_sidecar(&p).unwrap();
        let n = bytes.len();
        bytes[n - 6] ^= 0xff;
        fs::write(sidecar_path(&p), &bytes).unwrap();
        assert!(matches!(
            CheckpointStore::load(&p),
            Err(Error::CorruptFrame { .. })
        ));

        let mut bytes = read_sidecar(&p).unwrap();
        bytes[0] = b'X';
        fs::write(sidecar_path(&p), &bytes).unwrap();
        assert!(matches!(
            CheckpointStore::load(&p),
            Err(Error::CorruptFrame { .. })
        ));
    }

    /// The sidecar is replaced atomically (write-temp, fsync, rename,
    /// fsync-dir), so our own writes can never leave a zero-filled tail on it.
    /// Tolerate one anyway: the cost is a few lines, and the alternative is
    /// that reconstructible derived state makes the whole log refuse to open.
    #[test]
    fn zero_filled_tail_on_the_sidecar_is_ignored() {
        let d = TempDir::new("ckpt_zerotail");
        let p = log_path(&d);
        {
            let mut store = CheckpointStore::load(&p).unwrap();
            store.set("graph", 7).unwrap();
            store.set("user-model", 3).unwrap();
        }
        let clean = read_sidecar(&p).unwrap();
        for zeros in [1usize, 4, 64, 4096] {
            let mut bytes = clean.clone();
            bytes.extend(std::iter::repeat_n(0u8, zeros));
            fs::write(sidecar_path(&p), &bytes).unwrap();
            let store = CheckpointStore::load(&p).unwrap();
            assert_eq!(store.get("graph"), 7, "zeros={zeros}");
            assert_eq!(store.get("user-model"), 3, "zeros={zeros}");
        }
    }

    #[test]
    fn non_zero_trailing_bytes_on_the_sidecar_are_still_rejected() {
        let d = TempDir::new("ckpt_garbagetail");
        let p = log_path(&d);
        {
            let mut store = CheckpointStore::load(&p).unwrap();
            store.set("graph", 7).unwrap();
        }
        let clean = read_sidecar(&p).unwrap();

        let mut bytes = clean.clone();
        bytes.extend_from_slice(b"garbage");
        fs::write(sidecar_path(&p), &bytes).unwrap();
        assert!(matches!(
            CheckpointStore::load(&p),
            Err(Error::CorruptFrame { .. })
        ));

        // zeros followed by a non-zero byte is damage, same as in the log
        let mut bytes = clean.clone();
        bytes.extend(std::iter::repeat_n(0u8, 32));
        bytes.push(1);
        fs::write(sidecar_path(&p), &bytes).unwrap();
        assert!(matches!(
            CheckpointStore::load(&p),
            Err(Error::CorruptFrame { .. })
        ));
    }

    #[test]
    fn name_limits() {
        let d = TempDir::new("ckpt_names");
        let p = log_path(&d);
        let mut store = CheckpointStore::load(&p).unwrap();
        assert!(matches!(store.set("", 1), Err(Error::InvalidArgument(_))));
        let long = "n".repeat(MAX_KEY + 1);
        assert!(matches!(store.set(&long, 1), Err(Error::TooLarge(_))));
        let ok = "n".repeat(MAX_KEY);
        store.set(&ok, 5).unwrap();
        assert_eq!(CheckpointStore::load(&p).unwrap().get(&ok), 5);
    }
}
