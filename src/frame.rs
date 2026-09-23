//! Encode / decode / validate one frame, and the 32-byte file header.
//!
//! Byte layout (M0_SPEC.md "File format"), all integers little-endian:
//!
//! ```text
//! Header, 32 bytes, written once at creation:
//!   magic     8 bytes   b"OMEGALOG"
//!   version   u32       = 1
//!   reserved  20 bytes  zero
//!
//! Record frame, repeated:
//!   body_len  u32       byte length of body, 1..=MAX_BODY
//!   crc32     u32       CRC-32/ISO-HDLC over the body bytes
//!   body:
//!     seq        u64    starts at 1, strictly +1 per record
//!     ts_micros  i64    unix microseconds UTC
//!     key_len    u16    0 means "no dedup key", else 1..=MAX_KEY
//!     key        key_len bytes, UTF-8
//!     payload    body_len - (8+8+2+key_len) bytes, OPAQUE
//! ```

use crate::Error;

pub const MAGIC: [u8; 8] = *b"OMEGALOG";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 32;

/// Maximum body length, 64 MiB.
pub const MAX_BODY: u32 = 64 * 1024 * 1024;
/// Maximum dedup key length, 512 bytes.
pub const MAX_KEY: usize = 512;

/// `body_len` + `crc32` in front of every body.
pub const PREFIX_LEN: usize = 8;
/// `seq` + `ts_micros` + `key_len`: the part of a body that is always present.
pub const FIXED_BODY_LEN: usize = 8 + 8 + 2;

/// A decoded record, owning its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub seq: u64,
    pub ts_micros: i64,
    pub key: String,
    pub payload: Vec<u8>,
}

/// The 32-byte file header.
pub fn encode_header() -> [u8; HEADER_LEN] {
    let mut buf = [0u8; HEADER_LEN];
    buf[0..8].copy_from_slice(&MAGIC);
    buf[8..12].copy_from_slice(&VERSION.to_le_bytes());
    // bytes 12..32 stay zero (reserved)
    buf
}

/// Validate a 32-byte header. `NotAnOmegaLog` on bad magic, `UnsupportedVersion`
/// on anything but version 1. Both refuse to open (spec, Recovery step 4).
pub fn validate_header(buf: &[u8]) -> Result<u32, Error> {
    if buf.len() < HEADER_LEN {
        return Err(Error::CorruptFrame {
            offset: 0,
            detail: format!("header is {} bytes, want {HEADER_LEN}", buf.len()),
        });
    }
    if buf[0..8] != MAGIC {
        return Err(Error::NotAnOmegaLog);
    }
    let version = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if version != VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    Ok(version)
}

pub fn crc32(body: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(body);
    h.finalize()
}

/// Body length a record with this key and payload would need.
/// `None` if it does not fit in a u32 / exceeds `MAX_BODY`.
pub fn body_len_for(key: &str, payload_len: usize) -> Option<u32> {
    let total = FIXED_BODY_LEN
        .checked_add(key.len())?
        .checked_add(payload_len)?;
    let total: u32 = u32::try_from(total).ok()?;
    if total > MAX_BODY {
        None
    } else {
        Some(total)
    }
}

/// Validate what a caller handed us, before a single byte is written.
/// `TooLarge` with a precise reason; the file is untouched.
pub fn validate_append(key: &str, payload_len: usize) -> Result<u32, Error> {
    if key.len() > MAX_KEY {
        return Err(Error::TooLarge(format!(
            "write key is {} bytes, max is {MAX_KEY}",
            key.len()
        )));
    }
    body_len_for(key, payload_len).ok_or_else(|| {
        Error::TooLarge(format!(
            "record body would be {} bytes (payload {} + key {} + {FIXED_BODY_LEN}), max is {MAX_BODY}",
            FIXED_BODY_LEN as u128 + key.len() as u128 + payload_len as u128,
            payload_len,
            key.len(),
        ))
    })
}

/// Encode one complete frame (prefix + body). Validates first, so a rejected
/// encode never produces bytes.
pub fn encode(seq: u64, ts_micros: i64, key: &str, payload: &[u8]) -> Result<Vec<u8>, Error> {
    let body_len = validate_append(key, payload.len())?;

    let mut frame = Vec::with_capacity(PREFIX_LEN + body_len as usize);
    frame.extend_from_slice(&body_len.to_le_bytes());
    frame.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
    frame.extend_from_slice(&seq.to_le_bytes());
    frame.extend_from_slice(&ts_micros.to_le_bytes());
    frame.extend_from_slice(&(key.len() as u16).to_le_bytes());
    frame.extend_from_slice(key.as_bytes());
    frame.extend_from_slice(payload);

    debug_assert_eq!(frame.len(), PREFIX_LEN + body_len as usize);
    let crc = crc32(&frame[PREFIX_LEN..]);
    frame[4..8].copy_from_slice(&crc.to_le_bytes());
    Ok(frame)
}

/// Read `body_len` and `crc32` out of an 8-byte prefix.
pub fn decode_prefix(prefix: &[u8; PREFIX_LEN]) -> (u32, u32) {
    let body_len = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]);
    let crc = u32::from_le_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]);
    (body_len, crc)
}

/// Is this `body_len` one we could ever have written?
///
/// The spec calls out `0` and `> MAX_BODY` as `CorruptFrame` ("we wrote that
/// field; it cannot be legitimately out of range"). The same reasoning covers
/// `1..FIXED_BODY_LEN`: a body shorter than seq+ts+key_len is not a body.
pub fn body_len_is_plausible(body_len: u32) -> bool {
    body_len >= FIXED_BODY_LEN as u32 && body_len <= MAX_BODY
}

/// Everything in a body except the payload, borrowed. Recovery uses this so a
/// full rescan never copies payload bytes it is going to throw away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta<'a> {
    pub seq: u64,
    pub ts_micros: i64,
    pub key: &'a str,
    /// Offset of the payload within the body.
    pub payload_start: usize,
}

/// Decode a CRC-validated body's fixed part. Anything wrong here is
/// `CorruptFrame`: the CRC already proved these are the bytes we wrote, so an
/// unparseable body is real damage, not a torn write.
pub fn decode_meta(body: &[u8], offset: u64) -> Result<Meta<'_>, Error> {
    let bad = |detail: String| Error::CorruptFrame { offset, detail };

    if body.len() < FIXED_BODY_LEN {
        return Err(bad(format!(
            "body is {} bytes, minimum is {FIXED_BODY_LEN}",
            body.len()
        )));
    }
    let seq = u64::from_le_bytes(body[0..8].try_into().unwrap());
    let ts_micros = i64::from_le_bytes(body[8..16].try_into().unwrap());
    let key_len = u16::from_le_bytes(body[16..18].try_into().unwrap()) as usize;

    if key_len > MAX_KEY {
        return Err(bad(format!("key_len is {key_len}, max is {MAX_KEY}")));
    }
    let key_end = FIXED_BODY_LEN + key_len;
    if key_end > body.len() {
        return Err(bad(format!(
            "key_len {key_len} runs past the {}-byte body",
            body.len()
        )));
    }
    let key = std::str::from_utf8(&body[FIXED_BODY_LEN..key_end])
        .map_err(|e| bad(format!("key is not UTF-8: {e}")))?;

    Ok(Meta {
        seq,
        ts_micros,
        key,
        payload_start: key_end,
    })
}

/// Decode a CRC-validated body into an owning `Record`.
pub fn decode_body(body: &[u8], offset: u64) -> Result<Record, Error> {
    let meta = decode_meta(body, offset)?;
    Ok(Record {
        seq: meta.seq,
        ts_micros: meta.ts_micros,
        key: meta.key.to_owned(),
        payload: body[meta.payload_start..].to_vec(),
    })
}

/// Just the `seq` of a CRC-validated body, without copying the payload.
/// Used by recovery, which checks `seq` before it decides to keep the frame.
pub fn peek_seq(body: &[u8]) -> Option<u64> {
    if body.len() < FIXED_BODY_LEN {
        return None;
    }
    Some(u64::from_le_bytes(body[0..8].try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = encode_header();
        assert_eq!(h.len(), 32);
        assert_eq!(&h[0..8], b"OMEGALOG");
        assert_eq!(&h[8..12], &1u32.to_le_bytes());
        assert!(h[12..32].iter().all(|b| *b == 0));
        assert_eq!(validate_header(&h).unwrap(), 1);
    }

    #[test]
    fn header_bad_magic_and_version() {
        let mut h = encode_header();
        h[0] = b'X';
        assert!(matches!(validate_header(&h), Err(Error::NotAnOmegaLog)));

        let mut h = encode_header();
        h[8..12].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            validate_header(&h),
            Err(Error::UnsupportedVersion(7))
        ));
    }

    #[test]
    fn frame_round_trip() {
        let frame = encode(1, -12345, "k", b"hello").unwrap();
        let (body_len, crc) = decode_prefix(&frame[0..8].try_into().unwrap());
        assert_eq!(body_len as usize, FIXED_BODY_LEN + 1 + 5);
        assert_eq!(frame.len(), PREFIX_LEN + body_len as usize);
        let body = &frame[PREFIX_LEN..];
        assert_eq!(crc, crc32(body));

        let rec = decode_body(body, 32).unwrap();
        assert_eq!(rec.seq, 1);
        assert_eq!(rec.ts_micros, -12345);
        assert_eq!(rec.key, "k");
        assert_eq!(rec.payload, b"hello");
    }

    #[test]
    fn frame_round_trip_edges() {
        // empty payload, empty key
        let f = encode(9, 0, "", b"").unwrap();
        let r = decode_body(&f[PREFIX_LEN..], 0).unwrap();
        assert_eq!((r.seq, r.key.as_str(), r.payload.len()), (9, "", 0));

        // embedded NULs and invalid UTF-8 in the payload (payload is opaque)
        let payload = vec![0u8, 0xff, 0xfe, 0x00, b'a'];
        let f = encode(2, 1, "key with spaces", &payload).unwrap();
        let r = decode_body(&f[PREFIX_LEN..], 0).unwrap();
        assert_eq!(r.payload, payload);
        assert_eq!(r.key, "key with spaces");

        // max key
        let key = "k".repeat(MAX_KEY);
        let f = encode(3, 1, &key, b"x").unwrap();
        let r = decode_body(&f[PREFIX_LEN..], 0).unwrap();
        assert_eq!(r.key.len(), MAX_KEY);
    }

    #[test]
    fn oversize_key_and_payload_rejected_before_encoding() {
        let key = "k".repeat(MAX_KEY + 1);
        assert!(matches!(encode(1, 0, &key, b""), Err(Error::TooLarge(_))));

        let payload_len = MAX_BODY as usize; // + FIXED_BODY_LEN overflows MAX_BODY
        assert!(matches!(
            validate_append("", payload_len),
            Err(Error::TooLarge(_))
        ));
        // exactly at the limit is fine
        assert_eq!(
            validate_append("", MAX_BODY as usize - FIXED_BODY_LEN).unwrap(),
            MAX_BODY
        );
    }

    #[test]
    fn body_len_plausibility() {
        assert!(!body_len_is_plausible(0));
        assert!(!body_len_is_plausible(1));
        assert!(!body_len_is_plausible(FIXED_BODY_LEN as u32 - 1));
        assert!(body_len_is_plausible(FIXED_BODY_LEN as u32));
        assert!(body_len_is_plausible(MAX_BODY));
        assert!(!body_len_is_plausible(MAX_BODY + 1));
    }

    #[test]
    fn decode_body_rejects_key_len_past_end() {
        let mut f = encode(1, 0, "ab", b"cd").unwrap();
        // key_len is at body offset 16 -> frame offset 24
        f[24..26].copy_from_slice(&9000u16.to_le_bytes());
        assert!(matches!(
            decode_body(&f[PREFIX_LEN..], 0),
            Err(Error::CorruptFrame { .. })
        ));
    }

    #[test]
    fn decode_body_rejects_non_utf8_key() {
        let mut f = encode(1, 0, "ab", b"cd").unwrap();
        f[26] = 0xff; // first key byte
        assert!(matches!(
            decode_body(&f[PREFIX_LEN..], 0),
            Err(Error::CorruptFrame { .. })
        ));
    }

    #[test]
    fn crc_detects_a_flipped_payload_bit() {
        let mut f = encode(1, 0, "", b"payload").unwrap();
        let (_, crc) = decode_prefix(&f[0..8].try_into().unwrap());
        let last = f.len() - 1;
        f[last] ^= 0x01;
        assert_ne!(crc, crc32(&f[PREFIX_LEN..]));
    }
}
