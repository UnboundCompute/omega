//! `omega._log` — the structural half of omega's memory (DL-018: Rust owns
//! structure, Python owns meaning).
//!
//! This crate knows about frames, offsets, sequences and durability. It does
//! not know what an episode *means*: the payload is opaque bytes. The only
//! module permitted to import this one is `omega.memory` — the seam.

pub mod checkpoint;
pub mod frame;
pub mod log;

use std::fmt;

/// Everything that can go wrong, as distinct variants. The Python layer maps
/// each to its own exception type so callers can assert precisely rather than
/// matching on a message.
#[derive(Debug)]
pub enum Error {
    /// The file does not begin with `OMEGALOG`.
    NotAnOmegaLog,
    /// The header's version is not one we can read.
    UnsupportedVersion(u32),
    /// A frame that was fully durable is damaged. Fail closed; never truncate.
    CorruptFrame {
        offset: u64,
        detail: String,
    },
    /// A complete frame in the middle of the file has the wrong sequence number.
    SequenceBreak {
        offset: u64,
        expected: u64,
        found: u64,
    },
    /// The write key exists with a different payload. Nothing was written.
    WriteKeyConflict {
        key: String,
        existing_seq: u64,
    },
    /// A consumer position is past the end of the log. Cannot legitimately
    /// happen, and must not be reported as "no new episodes".
    CheckpointAhead {
        requested: u64,
        head: u64,
    },
    /// Another open log holds the exclusive lock (the singleton rule, DL-016).
    AlreadyLocked,
    /// A caller-supplied payload, key or name is over its limit. Nothing was
    /// written; the file is byte-identical.
    TooLarge(String),
    /// A caller-supplied argument is not meaningful (e.g. `seq` 0, which is
    /// reserved for "nothing consumed" and is never a record's sequence).
    InvalidArgument(String),
    /// The log has been closed.
    Closed,
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotAnOmegaLog => write!(f, "not an omega log: bad magic"),
            Error::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported log version {v}, this build reads version {}",
                    frame::VERSION
                )
            }
            Error::CorruptFrame { offset, detail } => {
                write!(f, "corrupt frame at offset {offset}: {detail}")
            }
            Error::SequenceBreak {
                offset,
                expected,
                found,
            } => write!(
                f,
                "sequence break at offset {offset}: expected seq {expected}, found {found}"
            ),
            Error::WriteKeyConflict { key, existing_seq } => write!(
                f,
                "write key {key:?} already stored at seq {existing_seq} with a different payload"
            ),
            Error::CheckpointAhead { requested, head } => {
                write!(f, "position {requested} is ahead of head {head}")
            }
            Error::AlreadyLocked => {
                write!(f, "another process or handle holds the lock on this log")
            }
            Error::TooLarge(what) => write!(f, "{what}"),
            Error::InvalidArgument(what) => write!(f, "{what}"),
            Error::Closed => write!(f, "the log is closed"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique temp directory, removed on drop. Keeps the crate at three
    /// dependencies by not pulling in `tempfile`.
    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn new(tag: &str) -> TempDir {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "omega-log-test-{}-{}-{}",
                std::process::id(),
                n,
                tag
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

mod py {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use pyo3::create_exception;
    use pyo3::exceptions::{PyException, PyOSError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyModule};
    use pyo3::PyTypeInfo;

    use crate::log::Log;
    use crate::Error;

    create_exception!(
        _log,
        OmegaLogError,
        PyException,
        "Base for every omega log error."
    );
    create_exception!(
        _log,
        NotAnOmegaLog,
        OmegaLogError,
        "The file is not an omega log."
    );
    create_exception!(
        _log,
        UnsupportedVersion,
        OmegaLogError,
        "The log's format version is unreadable."
    );
    create_exception!(
        _log,
        CorruptFrame,
        OmegaLogError,
        "A durable frame is damaged; the log refuses to open."
    );
    create_exception!(
        _log,
        SequenceBreak,
        OmegaLogError,
        "A mid-file frame has the wrong sequence number."
    );
    create_exception!(
        _log,
        WriteKeyConflict,
        OmegaLogError,
        "The write key exists with a different payload."
    );
    create_exception!(
        _log,
        CheckpointAhead,
        OmegaLogError,
        "A consumer position is ahead of the log's head."
    );
    create_exception!(
        _log,
        AlreadyLocked,
        OmegaLogError,
        "Another handle holds the exclusive lock."
    );
    create_exception!(
        _log,
        TooLarge,
        OmegaLogError,
        "A payload, key or name is over its limit."
    );
    create_exception!(_log, LogClosed, OmegaLogError, "The log has been closed.");

    fn to_py(e: Error) -> PyErr {
        let msg = e.to_string();
        match e {
            Error::NotAnOmegaLog => NotAnOmegaLog::new_err(msg),
            Error::UnsupportedVersion(_) => UnsupportedVersion::new_err(msg),
            Error::CorruptFrame { .. } => CorruptFrame::new_err(msg),
            Error::SequenceBreak { .. } => SequenceBreak::new_err(msg),
            Error::WriteKeyConflict { .. } => WriteKeyConflict::new_err(msg),
            Error::CheckpointAhead { .. } => CheckpointAhead::new_err(msg),
            Error::AlreadyLocked => AlreadyLocked::new_err(msg),
            Error::TooLarge(_) => TooLarge::new_err(msg),
            Error::InvalidArgument(_) => PyValueError::new_err(msg),
            Error::Closed => LogClosed::new_err(msg),
            Error::Io(_) => PyOSError::new_err(msg),
        }
    }

    type Shared = Arc<Mutex<Option<Log>>>;

    /// A poisoned mutex only means some earlier call panicked; the log itself is
    /// still whatever the file says, so recover the guard rather than hiding the
    /// log behind a second, unrelated error.
    fn guard(shared: &Shared) -> std::sync::MutexGuard<'_, Option<Log>> {
        shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with_log<R>(
        shared: &Shared,
        f: impl FnOnce(&mut Log) -> Result<R, Error>,
    ) -> Result<R, Error> {
        match guard(shared).as_mut() {
            Some(log) => f(log),
            None => Err(Error::Closed),
        }
    }

    /// The raw log. `omega.memory` is the only module allowed to touch this.
    #[pyclass(name = "Log", module = "omega._log")]
    pub struct PyLog {
        shared: Shared,
        path: PathBuf,
    }

    #[pymethods]
    impl PyLog {
        /// Open (creating if needed) the log file at `path` and take the
        /// exclusive lock. `path` is the log **file**, not a directory.
        #[new]
        fn new(py: Python<'_>, path: PathBuf) -> PyResult<PyLog> {
            let opened = py.detach(|| Log::open(&path)).map_err(to_py)?;
            Ok(PyLog {
                shared: Arc::new(Mutex::new(Some(opened))),
                path,
            })
        }

        #[getter]
        fn path(&self) -> PathBuf {
            self.path.clone()
        }

        #[getter]
        fn closed(&self) -> bool {
            guard(&self.shared).is_none()
        }

        /// Bytes discarded by torn-tail truncation when this handle opened.
        #[getter]
        fn recovered_bytes(&self) -> PyResult<u64> {
            with_log(&self.shared, |l| Ok(l.recovered_bytes())).map_err(to_py)
        }

        /// Seq of the last record, or 0 if the log is empty.
        fn head(&self) -> PyResult<u64> {
            with_log(&self.shared, |l| Ok(l.head())).map_err(to_py)
        }

        /// Append one record, `fsync`, then return its seq.
        #[pyo3(signature = (payload, key = "", ts_micros = None))]
        fn append(
            &self,
            py: Python<'_>,
            payload: &[u8],
            key: &str,
            ts_micros: Option<i64>,
        ) -> PyResult<u64> {
            let ts = ts_micros.unwrap_or_else(now_micros);
            py.detach(|| with_log(&self.shared, |l| l.append(payload, key, ts)))
                .map_err(to_py)
        }

        /// One record by seq, as `(seq, ts_micros, key, payload)`.
        fn read<'py>(&self, py: Python<'py>, seq: u64) -> PyResult<Bound<'py, PyAny>> {
            let rec = with_log(&self.shared, |l| l.read(seq)).map_err(to_py)?;
            record_to_py(py, rec)
        }

        /// A lazy iterator over records with `seq > since`.
        ///
        /// `since > head()` raises `CheckpointAhead`, never an empty iterator.
        /// The view is a snapshot taken now, so appends during iteration do not
        /// change what it yields.
        fn episodes_since(&self, since: u64) -> PyResult<PyRecordIter> {
            let (from, to) = with_log(&self.shared, |l| l.range_since(since)).map_err(to_py)?;
            Ok(PyRecordIter {
                shared: Arc::clone(&self.shared),
                next: from,
                end: to,
            })
        }

        /// An unset checkpoint reads as 0, never None.
        fn checkpoint(&self, name: &str) -> PyResult<u64> {
            with_log(&self.shared, |l| Ok(l.checkpoint(name))).map_err(to_py)
        }

        fn set_checkpoint(&self, py: Python<'_>, name: &str, seq: u64) -> PyResult<()> {
            py.detach(|| with_log(&self.shared, |l| l.set_checkpoint(name, seq)))
                .map_err(to_py)
        }

        fn checkpoint_names(&self) -> PyResult<Vec<String>> {
            with_log(&self.shared, |l| Ok(l.checkpoint_names())).map_err(to_py)
        }

        /// Current file size in bytes. Diagnostic.
        fn size_bytes(&self) -> PyResult<u64> {
            with_log(&self.shared, |l| Ok(l.len_bytes())).map_err(to_py)
        }

        /// The offset index, as a list. Diagnostic: the tests compare it to a
        /// fresh scan (both indexes are caches, never authoritative).
        fn offsets(&self) -> PyResult<Vec<u64>> {
            with_log(&self.shared, |l| Ok(l.offsets().to_vec())).map_err(to_py)
        }

        /// Drop both caches and rebuild them from the file.
        fn rebuild_indexes(&self, py: Python<'_>) -> PyResult<()> {
            py.detach(|| with_log(&self.shared, |l| l.rebuild_indexes()))
                .map_err(to_py)
        }

        /// Release the lock and the file handle. Idempotent.
        fn close(&self) {
            *guard(&self.shared) = None;
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[pyo3(signature = (_exc_type = None, _exc_value = None, _traceback = None))]
        fn __exit__(
            &self,
            _exc_type: Option<Bound<'_, PyAny>>,
            _exc_value: Option<Bound<'_, PyAny>>,
            _traceback: Option<Bound<'_, PyAny>>,
        ) -> bool {
            self.close();
            false
        }

        fn __repr__(&self) -> String {
            match guard(&self.shared).as_ref() {
                Some(l) => format!("<omega._log.Log {:?} head={}>", self.path, l.head()),
                None => format!("<omega._log.Log {:?} closed>", self.path),
            }
        }
    }

    #[pyclass(name = "RecordIter", module = "omega._log")]
    pub struct PyRecordIter {
        shared: Shared,
        next: u64,
        end: u64,
    }

    #[pymethods]
    impl PyRecordIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
            if self.next > self.end {
                return Ok(None);
            }
            let seq = self.next;
            let rec = with_log(&self.shared, |l| l.read(seq)).map_err(to_py)?;
            self.next += 1;
            Ok(Some(record_to_py(py, rec)?))
        }
    }

    fn record_to_py(py: Python<'_>, rec: crate::frame::Record) -> PyResult<Bound<'_, PyAny>> {
        let payload = PyBytes::new(py, &rec.payload);
        let tuple = (rec.seq, rec.ts_micros, rec.key, payload);
        Ok(tuple.into_pyobject(py)?.into_any())
    }

    fn now_micros() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_micros() as i64,
            Err(e) => -(e.duration().as_micros() as i64),
        }
    }

    #[pymodule]
    #[pyo3(name = "_log")]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyLog>()?;
        m.add_class::<PyRecordIter>()?;

        // `create_exception!` takes the module as an identifier, so it records
        // `__module__ = "_log"`. Correct it to the real dotted name so
        // tracebacks and pickling name the type properly.
        fn add_exc<T: PyTypeInfo>(m: &Bound<'_, PyModule>, name: &str) -> PyResult<()> {
            let ty = m.py().get_type::<T>();
            ty.setattr("__module__", "omega._log")?;
            m.add(name, ty)
        }

        add_exc::<OmegaLogError>(m, "OmegaLogError")?;
        add_exc::<NotAnOmegaLog>(m, "NotAnOmegaLog")?;
        add_exc::<UnsupportedVersion>(m, "UnsupportedVersion")?;
        add_exc::<CorruptFrame>(m, "CorruptFrame")?;
        add_exc::<SequenceBreak>(m, "SequenceBreak")?;
        add_exc::<WriteKeyConflict>(m, "WriteKeyConflict")?;
        add_exc::<CheckpointAhead>(m, "CheckpointAhead")?;
        add_exc::<AlreadyLocked>(m, "AlreadyLocked")?;
        add_exc::<TooLarge>(m, "TooLarge")?;
        add_exc::<LogClosed>(m, "LogClosed")?;

        m.add("MAX_BODY", crate::frame::MAX_BODY)?;
        m.add("MAX_KEY", crate::frame::MAX_KEY)?;
        m.add("HEADER_LEN", crate::frame::HEADER_LEN)?;
        m.add("VERSION", crate::frame::VERSION)?;
        m.add("EPISODES_FILENAME", crate::log::EPISODES_FILENAME)?;
        Ok(())
    }
}
