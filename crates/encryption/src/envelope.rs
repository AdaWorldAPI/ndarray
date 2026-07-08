//! The sealed envelope — password-locked, self-describing, zero-knowledge.
//!
//! [`seal`] runs client-side (browser wasm or native): it derives a key
//! from the password with Argon2id, draws a fresh salt + XChaCha20
//! nonce from the CSPRNG, and returns one opaque blob. A server that
//! stores the blob learns **nothing** — the password never leaves the
//! client, and the blob authenticates its own header, so any tampering
//! (including with the cost parameters) makes [`open`] fail.
//!
//! ## Byte layout (fixed, little-endian, parsed by hand — no serde)
//!
//! ```text
//! offset  len  field
//!  0       4   magic       = b"ADAC"
//!  4       1   version     = 1
//!  5       4   m_cost_kib  (u32 LE)   Argon2id memory cost
//!  9       4   t_cost      (u32 LE)   Argon2id passes
//! 13       4   p_cost      (u32 LE)   Argon2id lanes
//! 17      16   salt                   fresh CSPRNG bytes per seal
//! 33      24   nonce                  fresh CSPRNG bytes per seal
//! 57       …   ciphertext ‖ 16-byte Poly1305 tag
//! ```
//!
//! The first 57 bytes (the header) are the AEAD's associated data, so
//! they are integrity-bound to the ciphertext: an attacker cannot, for
//! example, lower `m_cost_kib` to cheapen an offline guess and still
//! have the blob open.

use crate::aead::{self, NONCE_LEN};
pub use crate::kdf::KdfParams;
use crate::kdf::{self, KdfError};

/// Envelope magic: "ADAC" (Ada crypto).
pub const MAGIC: [u8; 4] = *b"ADAC";
/// Current envelope layout version.
pub const VERSION: u8 = 1;
/// Salt length stored in the header.
pub const SALT_LEN: usize = 16;
/// Total header length preceding the ciphertext.
pub const HEADER_LEN: usize = 4 + 1 + 4 + 4 + 4 + SALT_LEN + NONCE_LEN; // 57

/// Why an envelope could not be sealed or opened. Field-free — an
/// `open` failure never says *which* part was wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeError {
    /// The platform CSPRNG was unavailable (seal only).
    Rng,
    /// The Argon2id parameters were rejected.
    Kdf(KdfError),
    /// The blob is not an envelope (bad magic, short buffer).
    Malformed,
    /// The blob's layout version is newer than this code understands.
    UnsupportedVersion,
    /// Wrong password, or the blob was tampered with. Indistinguishable
    /// by design.
    Decrypt,
    /// Encryption failed (should not happen with valid inputs).
    Encrypt,
}

impl core::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EnvelopeError::Rng => f.write_str("platform CSPRNG unavailable"),
            EnvelopeError::Kdf(e) => write!(f, "key derivation failed: {e}"),
            EnvelopeError::Malformed => f.write_str("not a sealed envelope"),
            EnvelopeError::UnsupportedVersion => f.write_str("unsupported envelope version"),
            EnvelopeError::Decrypt => f.write_str("wrong password or tampered envelope"),
            EnvelopeError::Encrypt => f.write_str("encryption failed"),
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Seal `plaintext` under `password`. Draws salt + nonce from the
/// CSPRNG, so sealing the same input twice yields different blobs.
pub fn seal(password: &[u8], plaintext: &[u8], params: &KdfParams) -> Result<Vec<u8>, EnvelopeError> {
    let mut salt = [0u8; SALT_LEN];
    crate::fill_random(&mut salt).map_err(|_| EnvelopeError::Rng)?;
    let mut nonce = [0u8; NONCE_LEN];
    crate::fill_random(&mut nonce).map_err(|_| EnvelopeError::Rng)?;

    let header = encode_header(params, &salt, &nonce);
    let key = kdf::derive_key(password, &salt, params).map_err(EnvelopeError::Kdf)?;
    let ciphertext =
        aead::seal_with_key(key.as_bytes(), &nonce, &header, plaintext).map_err(|_| EnvelopeError::Encrypt)?;

    let mut blob = Vec::with_capacity(HEADER_LEN + ciphertext.len());
    blob.extend_from_slice(&header);
    blob.extend_from_slice(&ciphertext);
    Ok(blob)
}

/// Open a sealed envelope with `password`. The KDF parameters are read
/// from the (authenticated) header, so cost bumps never orphan old blobs.
pub fn open(password: &[u8], blob: &[u8]) -> Result<Vec<u8>, EnvelopeError> {
    let (params, salt, nonce) = decode_header(blob)?;
    let header = &blob[..HEADER_LEN];
    let ciphertext = &blob[HEADER_LEN..];

    let key = kdf::derive_key(password, &salt, &params).map_err(EnvelopeError::Kdf)?;
    aead::open_with_key(key.as_bytes(), &nonce, header, ciphertext).map_err(|_| EnvelopeError::Decrypt)
}

fn encode_header(params: &KdfParams, salt: &[u8; SALT_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = VERSION;
    h[5..9].copy_from_slice(&params.m_cost_kib.to_le_bytes());
    h[9..13].copy_from_slice(&params.t_cost.to_le_bytes());
    h[13..17].copy_from_slice(&params.p_cost.to_le_bytes());
    h[17..17 + SALT_LEN].copy_from_slice(salt);
    h[33..33 + NONCE_LEN].copy_from_slice(nonce);
    h
}

fn decode_header(blob: &[u8]) -> Result<(KdfParams, [u8; SALT_LEN], [u8; NONCE_LEN]), EnvelopeError> {
    if blob.len() < HEADER_LEN || blob[0..4] != MAGIC {
        return Err(EnvelopeError::Malformed);
    }
    if blob[4] != VERSION {
        return Err(EnvelopeError::UnsupportedVersion);
    }
    let le_u32 = |at: usize| u32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]]);
    let params = KdfParams {
        m_cost_kib: le_u32(5),
        t_cost: le_u32(9),
        p_cost: le_u32(13),
    };
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&blob[17..17 + SALT_LEN]);
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&blob[33..33 + NONCE_LEN]);
    Ok((params, salt, nonce))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAST: KdfParams = KdfParams {
        m_cost_kib: 32,
        t_cost: 1,
        p_cost: 1,
    };

    #[test]
    fn round_trip() {
        let blob = seal(b"hunter2", b"the secret", &FAST).unwrap();
        assert_eq!(open(b"hunter2", &blob).unwrap(), b"the secret");
    }

    #[test]
    fn wrong_password_fails_opaquely() {
        let blob = seal(b"right", b"s", &FAST).unwrap();
        assert_eq!(open(b"wrong", &blob).unwrap_err(), EnvelopeError::Decrypt);
    }

    #[test]
    fn same_input_two_seals_differ() {
        let a = seal(b"pw", b"s", &FAST).unwrap();
        let b = seal(b"pw", b"s", &FAST).unwrap();
        assert_ne!(a, b, "salt+nonce must be fresh per seal");
    }

    #[test]
    fn header_tamper_fails() {
        let mut blob = seal(b"pw", b"s", &FAST).unwrap();
        // Attacker tries to cheapen the KDF cost in the header.
        blob[5] = 1; // m_cost_kib low byte
        assert!(open(b"pw", &blob).is_err());
    }

    #[test]
    fn ciphertext_tamper_fails() {
        let mut blob = seal(b"pw", b"s", &FAST).unwrap();
        let last = blob.len() - 1;
        blob[last] ^= 0x80;
        assert_eq!(open(b"pw", &blob).unwrap_err(), EnvelopeError::Decrypt);
    }

    #[test]
    fn malformed_inputs_rejected() {
        assert_eq!(open(b"pw", b"").unwrap_err(), EnvelopeError::Malformed);
        assert_eq!(open(b"pw", &[0u8; HEADER_LEN]).unwrap_err(), EnvelopeError::Malformed);
        let mut blob = seal(b"pw", b"s", &FAST).unwrap();
        blob[4] = 99;
        assert_eq!(open(b"pw", &blob).unwrap_err(), EnvelopeError::UnsupportedVersion);
    }

    #[test]
    fn empty_plaintext_round_trips() {
        let blob = seal(b"pw", b"", &FAST).unwrap();
        assert_eq!(blob.len(), HEADER_LEN + crate::aead::TAG_LEN);
        assert_eq!(open(b"pw", &blob).unwrap(), b"");
    }
}
