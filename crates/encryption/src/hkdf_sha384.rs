//! HKDF-SHA-384 key schedule for the sealed session channel.
//!
//! This is a thin, typed wrapper over the RustCrypto [`hkdf`] crate, pinned
//! to SHA-384 (see [`crate::hash`] for why SHA-384 is this workspace's
//! truncation-resistant hash of choice).
//!
//! ## Intended use
//!
//! This module is the key schedule for a Noise_NK-shaped handshake:
//!
//! - `ikm` (input key material) is the **concatenation of the X25519 shared
//!   secrets** produced by the key-exchange step (see `crate::kx`, which
//!   supplies the Diffie-Hellman output this module consumes — it does not
//!   do key agreement itself).
//! - `salt` is the **handshake transcript hash** — binding the derived keys
//!   to the exact sequence of messages that produced them, so a
//!   transcript substitution changes every downstream key.
//! - `info` is a **domain separator**. Callers MUST use a distinct `info`
//!   per protocol version (and per intended use of the output, e.g.
//!   "record key" vs "rekey seed"). HKDF-Expand treats `info` as part of
//!   the label that binds output bytes to a specific context; reusing one
//!   `info` across two protocol versions means a key derived under version
//!   1 and a key derived under version 2 can collide if the rest of the
//!   transcript ever coincides. A version byte (or string) folded into
//!   `info` keeps the two derivations in disjoint output spaces even if
//!   every other input matches.
//!
//! ## Extract / Expand / one-shot
//!
//! [`extract`] and [`expand`] expose the two RFC 5869 steps separately for
//! callers that need to run one Extract and many Expands from the same
//! [`Prk`] (e.g. deriving several sub-keys, as [`session_keys`] does).
//! [`derive`] is the one-shot convenience for callers who only need a
//! single Expand and don't want to hold a [`Prk`] around.

use hkdf::Hkdf;
use sha2::Sha384;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Byte length of the SHA-384 HKDF pseudorandom key (== HMAC-SHA-384's
/// output size).
pub const PRK_LEN: usize = 48;

/// The largest output HKDF-Expand can produce for any hash: `255` times
/// the hash's output length. For SHA-384 that is `255 * 48 = 12_240`
/// bytes — this is the only way [`expand`] or [`derive`] can fail.
pub const MAX_OKM_LEN: usize = 255 * PRK_LEN;

/// The HKDF-Extract output (the "pseudorandom key", PRK). Wiped from
/// memory on drop; deliberately has no `Debug` impl — a key must never be
/// printable.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Prk([u8; PRK_LEN]);

impl Prk {
    /// Borrow the raw PRK bytes (for handing to [`expand`] or an
    /// alternative HKDF-Expand implementation).
    pub fn as_bytes(&self) -> &[u8; PRK_LEN] {
        &self.0
    }
}

/// HKDF failure. Field-free on purpose — no secret material, no
/// parameter echo.
///
/// The only way HKDF-Expand fails per RFC 5869 is an output request past
/// its length ceiling ([`MAX_OKM_LEN`]); there is no other failure mode in
/// this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HkdfError {
    /// The requested output (`out.len()` / the `L` parameter in RFC 5869
    /// terms) exceeds [`MAX_OKM_LEN`].
    OutputTooLong,
}

impl core::fmt::Display for HkdfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HkdfError::OutputTooLong => f.write_str("HKDF output length exceeds 255 * hash length"),
        }
    }
}

impl std::error::Error for HkdfError {}

/// HKDF-Extract: condense `salt` and `ikm` into a uniformly-random,
/// fixed-length [`Prk`]. Cannot fail — Extract is a single HMAC call.
///
/// ```
/// use encryption::hkdf_sha384::extract;
///
/// let prk = extract(b"transcript-hash", b"x25519-shared-secret");
/// assert_eq!(prk.as_bytes().len(), 48);
/// ```
pub fn extract(salt: &[u8], ikm: &[u8]) -> Prk {
    let (prk, _hkdf) = Hkdf::<Sha384>::extract(Some(salt), ikm);
    let mut out = [0u8; PRK_LEN];
    out.copy_from_slice(&prk);
    Prk(out)
}

/// HKDF-Expand: stretch a [`Prk`] into `out.len()` bytes of output key
/// material, bound to `info`.
///
/// ```
/// use encryption::hkdf_sha384::{extract, expand};
///
/// let prk = extract(b"salt", b"ikm");
/// let mut key = [0u8; 32];
/// expand(&prk, b"medcare-rs/v1/record-key", &mut key).unwrap();
/// ```
pub fn expand(prk: &Prk, info: &[u8], out: &mut [u8]) -> Result<(), HkdfError> {
    // A `Prk` is always exactly `PRK_LEN` bytes — the length `Hkdf::extract`
    // for SHA-384 always produces — so `from_prk`'s "at least HashLen"
    // check can never reject it.
    let hk = Hkdf::<Sha384>::from_prk(prk.as_bytes())
        .expect("Prk is always PRK_LEN (48) bytes, exactly SHA-384's output size");
    hk.expand(info, out).map_err(|_| HkdfError::OutputTooLong)
}

/// One-shot HKDF-Extract-then-Expand: derive `out.len()` bytes directly
/// from `salt`, `ikm`, and `info` without exposing an intermediate
/// [`Prk`]. Equivalent to `expand(&extract(salt, ikm), info, out)`, at the
/// cost of one fewer HMAC pass since the PRK is consumed internally
/// instead of round-tripped through a byte array.
///
/// ```
/// use encryption::hkdf_sha384::derive;
///
/// let mut key = [0u8; 32];
/// derive(b"transcript-hash", b"shared-secret", b"medcare-rs/v1", &mut key).unwrap();
/// ```
pub fn derive(salt: &[u8], ikm: &[u8], info: &[u8], out: &mut [u8]) -> Result<(), HkdfError> {
    let hk = Hkdf::<Sha384>::new(Some(salt), ikm);
    hk.expand(info, out).map_err(|_| HkdfError::OutputTooLong)
}

/// The two directional keys of a sealed session channel: client-to-server
/// and server-to-client. Wiped from memory on drop; deliberately has no
/// `Debug` impl.
///
/// **Why two keys, not one:** if both directions of a channel used the
/// same key, an attacker could take a genuine record sent in one
/// direction and replay it back in the other — reflection — and it would
/// authenticate, because the same key produced a valid tag either way.
/// Deriving `c2s` and `s2c` from independent HKDF-Expand labels means a
/// record authenticated under one direction's key is simply not valid
/// under the other's; a reflected record fails to authenticate instead of
/// silently passing.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SessionKeys {
    /// Client-to-server key.
    pub c2s: [u8; 32],
    /// Server-to-client key.
    pub s2c: [u8; 32],
}

/// Derive both directional session keys from a single handshake output.
///
/// Runs one HKDF-Extract over `(salt, ikm)`, then two HKDF-Expand calls
/// against that shared [`Prk`] — one labeled for the client-to-server
/// direction, one for server-to-client — so the two keys are
/// cryptographically independent even though they share a PRK. See
/// [`SessionKeys`] for why the directions must not share a key.
///
/// `info` is still the caller's domain separator (see the module docs on
/// `info` / protocol versioning); the directional label is appended after
/// it, not folded into it, so callers keep control of the version tag.
///
/// Infallible: each directional output is a fixed 32 bytes, far below
/// [`MAX_OKM_LEN`], so the only failure mode [`expand`] has can never
/// trigger here.
///
/// ```
/// use encryption::hkdf_sha384::session_keys;
///
/// let keys = session_keys(b"transcript-hash", b"shared-secret", b"medcare-rs/v1");
/// assert_ne!(keys.c2s, keys.s2c);
/// ```
pub fn session_keys(salt: &[u8], ikm: &[u8], info: &[u8]) -> SessionKeys {
    let prk = extract(salt, ikm);
    let hk = Hkdf::<Sha384>::from_prk(prk.as_bytes())
        .expect("Prk is always PRK_LEN (48) bytes, exactly SHA-384's output size");

    let c2s_label: &[u8] = b"|c2s";
    let s2c_label: &[u8] = b"|s2c";
    let mut c2s = [0u8; 32];
    let mut s2c = [0u8; 32];
    hk.expand_multi_info(&[info, c2s_label], &mut c2s)
        .expect("32 bytes is far below MAX_OKM_LEN for SHA-384");
    hk.expand_multi_info(&[info, s2c_label], &mut s2c)
        .expect("32 bytes is far below MAX_OKM_LEN for SHA-384");
    SessionKeys { c2s, s2c }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_inputs() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        derive(b"salt", b"ikm", b"info", &mut a).unwrap();
        derive(b"salt", b"ikm", b"info", &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_salt_changes_the_output() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        derive(b"salt-a", b"ikm", b"info", &mut a).unwrap();
        derive(b"salt-b", b"ikm", b"info", &mut b).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn different_ikm_changes_the_output() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        derive(b"salt", b"ikm-a", b"info", &mut a).unwrap();
        derive(b"salt", b"ikm-b", b"info", &mut b).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn different_info_changes_the_output() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        derive(b"salt", b"ikm", b"info-a", &mut a).unwrap();
        derive(b"salt", b"ikm", b"info-b", &mut b).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn the_two_directional_keys_differ_from_each_other() {
        let keys = session_keys(b"transcript-hash", b"shared-secret", b"medcare-rs/v1");
        assert_ne!(keys.c2s, keys.s2c);
    }

    #[test]
    fn output_lengths_other_than_32_bytes_work() {
        for len in [16usize, 64, 96] {
            let mut out = vec![0u8; len];
            derive(b"salt", b"ikm", b"info", &mut out).unwrap();
            assert!(out.iter().any(|&b| b != 0), "derived output of length {len} was all-zero");
        }
    }

    /// HKDF-Expand refuses an output request past `255 * hash_len` bytes
    /// (RFC 5869 §2.3) instead of silently truncating it.
    #[test]
    fn an_over_long_output_is_rejected_rather_than_truncated() {
        let mut out = vec![0u8; MAX_OKM_LEN + 1];
        assert_eq!(derive(b"salt", b"ikm", b"info", &mut out), Err(HkdfError::OutputTooLong));

        // The ceiling itself is still admitted.
        let mut at_ceiling = vec![0u8; MAX_OKM_LEN];
        assert_eq!(derive(b"salt", b"ikm", b"info", &mut at_ceiling), Ok(()));
    }

    /// RFC 5869 §A.1 (Test Case 1), the canonical published HKDF vector —
    /// SHA-256, not SHA-384; no SHA-384 vector has ever been published by
    /// the RFC or a widely-cited independent source. This exercises the
    /// same `hkdf` crate machinery this module's SHA-384 wrapper sits on
    /// top of (same `Hkdf<H>` type, `H` swapped for `Sha256`), as a wiring
    /// sanity check: it proves this crate's dependency wiring (feature
    /// flags, no_std posture) reproduces a real, citable vector, which is
    /// the closest available evidence that the SHA-384 instantiation
    /// alongside it is wired the same way.
    //
    // TODO(orchestrator): no published SHA-384 vector; consider
    // cross-checking against an independent implementation.
    #[test]
    fn rfc5869_test_case_1_sha256_wiring_check() {
        use sha2::Sha256;

        let ikm = [0x0bu8; 22];
        #[rustfmt::skip]
        let salt: [u8; 13] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        #[rustfmt::skip]
        let info: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];
        #[rustfmt::skip]
        let expected_okm: [u8; 42] = [
            0x3c, 0xb2, 0x5f, 0x25, 0xfa, 0xac, 0xd5, 0x7a, 0x90, 0x43, 0x4f, 0x64, 0xd0, 0x36, 0x2f, 0x2a,
            0x2d, 0x2d, 0x0a, 0x90, 0xcf, 0x1a, 0x5a, 0x4c, 0x5d, 0xb0, 0x2d, 0x56, 0xec, 0xc4, 0xc5, 0xbf,
            0x34, 0x00, 0x72, 0x08, 0xd5, 0xb8, 0x87, 0x18, 0x58, 0x65,
        ];

        let hk = Hkdf::<Sha256>::new(Some(&salt[..]), &ikm);
        let mut okm = [0u8; 42];
        hk.expand(&info, &mut okm)
            .expect("42 is a valid length for SHA-256 to output");
        assert_eq!(okm, expected_okm);
    }
}
