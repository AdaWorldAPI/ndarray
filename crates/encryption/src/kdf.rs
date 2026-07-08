//! Argon2id key derivation — password → 256-bit AEAD key.
//!
//! The parameters travel inside the sealed envelope header (see
//! [`crate::envelope`]) so old blobs stay openable after a cost bump.

use argon2::{Algorithm, Argon2, Params, Version};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Argon2id cost parameters. Stored verbatim (little-endian) in the
/// envelope header, so they are part of the authenticated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdfParams {
    /// Memory cost in KiB.
    pub m_cost_kib: u32,
    /// Iteration count (passes over memory).
    pub t_cost: u32,
    /// Parallelism (lanes).
    pub p_cost: u32,
}

impl KdfParams {
    /// Server-grade default: 64 MiB, 3 passes, 1 lane.
    pub const DEFAULT: KdfParams = KdfParams {
        m_cost_kib: 64 * 1024,
        t_cost: 3,
        p_cost: 1,
    };

    /// Interactive / browser-grade: 19 MiB, 2 passes, 1 lane
    /// (the OWASP first-recommended Argon2id configuration).
    /// Use when the derivation runs on every login inside wasm.
    pub const INTERACTIVE: KdfParams = KdfParams {
        m_cost_kib: 19 * 1024,
        t_cost: 2,
        p_cost: 1,
    };
}

impl Default for KdfParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A derived 256-bit key. Wiped from memory on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct DerivedKey(pub(crate) [u8; 32]);

impl DerivedKey {
    /// Borrow the raw key bytes (for handing to the AEAD).
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Key-derivation failure. Field-free on purpose — no secret material,
/// no parameter echo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdfError {
    /// The cost parameters are outside Argon2's accepted range.
    InvalidParams,
    /// The derivation itself failed (allocation, internal error).
    DerivationFailed,
}

impl core::fmt::Display for KdfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KdfError::InvalidParams => f.write_str("invalid Argon2id parameters"),
            KdfError::DerivationFailed => f.write_str("Argon2id derivation failed"),
        }
    }
}

impl std::error::Error for KdfError {}

/// Derive a 256-bit key from `password` and a 16-byte `salt` with
/// Argon2id (v1.3). Deterministic: same inputs → same key.
pub fn derive_key(password: &[u8], salt: &[u8; 16], params: &KdfParams) -> Result<DerivedKey, KdfError> {
    let argon_params =
        Params::new(params.m_cost_kib, params.t_cost, params.p_cost, Some(32)).map_err(|_| KdfError::InvalidParams)?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);
    let mut key = [0u8; 32];
    argon
        .hash_password_into(password, salt, &mut key)
        .map_err(|_| KdfError::DerivationFailed)?;
    Ok(DerivedKey(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small params so tests stay fast; correctness is parameter-independent.
    const TEST_PARAMS: KdfParams = KdfParams {
        m_cost_kib: 32,
        t_cost: 1,
        p_cost: 1,
    };

    #[test]
    fn deterministic_for_same_inputs() {
        let salt = [7u8; 16];
        let a = derive_key(b"correct horse", &salt, &TEST_PARAMS).unwrap();
        let b = derive_key(b"correct horse", &salt, &TEST_PARAMS).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn different_salt_different_key() {
        let a = derive_key(b"pw", &[1u8; 16], &TEST_PARAMS).unwrap();
        let b = derive_key(b"pw", &[2u8; 16], &TEST_PARAMS).unwrap();
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn different_password_different_key() {
        let salt = [9u8; 16];
        let a = derive_key(b"pw-a", &salt, &TEST_PARAMS).unwrap();
        let b = derive_key(b"pw-b", &salt, &TEST_PARAMS).unwrap();
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn rejects_zero_memory() {
        let bad = KdfParams {
            m_cost_kib: 0,
            t_cost: 1,
            p_cost: 1,
        };
        // No `unwrap_err()` here: DerivedKey deliberately has no Debug
        // impl (a key must never be printable).
        match derive_key(b"pw", &[0u8; 16], &bad) {
            Err(e) => assert_eq!(e, KdfError::InvalidParams),
            Ok(_) => panic!("zero-memory params must be rejected"),
        }
    }
}
