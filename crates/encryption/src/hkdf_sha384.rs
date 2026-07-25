//! HMAC-SHA384 and HKDF-SHA384 — the key schedule for [`crate::channel`].
//!
//! Built directly on [`crate::hash::sha384`] rather than pulling the `hmac` /
//! `hkdf` crates. That is deliberate: those sit on the `digest 0.11` trait
//! generation while this crate's `sha2` is on `0.10`, so depending on them
//! today drags a second trait generation into the graph for two functions that
//! are twenty lines each. HMAC is RFC 2104 and HKDF is RFC 5869; both are
//! short, and both are pinned here by published test vectors (RFC 4231 for
//! HMAC-SHA384) rather than by trust.

use crate::hash::{sha384, DIGEST_LEN};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// SHA-384's block size, in bytes — the width HMAC pads keys to.
const BLOCK_LEN: usize = 128;

/// HMAC-SHA384 (RFC 2104): `H((K ^ opad) ‖ H((K ^ ipad) ‖ msg))`.
///
/// Keys longer than the block size are hashed first; shorter keys are
/// zero-padded. `msg` is passed as a slice of parts so callers can feed a
/// transcript without concatenating it into a temporary buffer.
#[must_use]
pub fn hmac_sha384(key: &[u8], parts: &[&[u8]]) -> [u8; DIGEST_LEN] {
    let mut padded = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        padded[..DIGEST_LEN].copy_from_slice(&sha384(key));
    } else {
        padded[..key.len()].copy_from_slice(key);
    }

    let mut inner = [0u8; BLOCK_LEN];
    let mut outer = [0u8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        inner[i] = padded[i] ^ 0x36;
        outer[i] = padded[i] ^ 0x5c;
    }
    padded.zeroize();

    // H((K ^ ipad) ‖ msg)
    let mut inner_input = Vec::with_capacity(BLOCK_LEN + parts.iter().map(|p| p.len()).sum::<usize>());
    inner_input.extend_from_slice(&inner);
    for p in parts {
        inner_input.extend_from_slice(p);
    }
    let inner_hash = sha384(&inner_input);
    inner_input.zeroize();
    inner.zeroize();

    // H((K ^ opad) ‖ inner)
    let mut outer_input = [0u8; BLOCK_LEN + DIGEST_LEN];
    outer_input[..BLOCK_LEN].copy_from_slice(&outer);
    outer_input[BLOCK_LEN..].copy_from_slice(&inner_hash);
    let out = sha384(&outer_input);
    outer_input.zeroize();
    outer.zeroize();
    out
}

/// An HKDF pseudo-random key. Wiped on drop; deliberately has no `Debug`.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Prk([u8; DIGEST_LEN]);

impl Prk {
    /// Borrow the raw PRK bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; DIGEST_LEN] {
        &self.0
    }
}

/// HKDF-Extract (RFC 5869 §2.2): `PRK = HMAC(salt, ikm)`.
///
/// The salt is the *public* transcript; the IKM is the Diffie-Hellman output,
/// which is uniform-ish but not uniform — extracting is what turns it into a
/// key, and skipping it is the classic mistake.
#[must_use]
pub fn extract(salt: &[u8], ikm: &[u8]) -> Prk {
    Prk(hmac_sha384(salt, &[ikm]))
}

/// HKDF-Expand (RFC 5869 §2.3) into a caller-provided buffer.
///
/// Returns `Err(())` for an output longer than `255 * 48` bytes, the
/// construction's ceiling.
pub fn expand(prk: &Prk, info: &[u8], out: &mut [u8]) -> Result<(), ExpandTooLong> {
    if out.len() > 255 * DIGEST_LEN {
        return Err(ExpandTooLong);
    }
    let mut t: [u8; DIGEST_LEN] = [0u8; DIGEST_LEN];
    let mut written = 0usize;
    let mut counter: u8 = 1;
    while written < out.len() {
        let block = if counter == 1 {
            hmac_sha384(prk.as_bytes(), &[info, &[counter]])
        } else {
            hmac_sha384(prk.as_bytes(), &[&t, info, &[counter]])
        };
        let take = core::cmp::min(DIGEST_LEN, out.len() - written);
        out[written..written + take].copy_from_slice(&block[..take]);
        t = block;
        written += take;
        counter = counter.wrapping_add(1);
    }
    t.zeroize();
    Ok(())
}

/// Requested output exceeded HKDF's `255 * HashLen` ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandTooLong;

impl core::fmt::Display for ExpandTooLong {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HKDF output exceeds 255 * HashLen")
    }
}

impl std::error::Error for ExpandTooLong {}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4231 test case 1 for HMAC-SHA-384. The published vector is the only
    /// thing that makes a hand-written HMAC trustworthy — without it this file
    /// is an assertion, not an implementation.
    #[test]
    fn hmac_sha384_matches_rfc4231_case_1() {
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let expected = hex("afd03944d84895626b0825f4ab46907f\
             15f9dadbe4101ec682aa034c7cebc59c\
             faea9ea9076ede7f4af152e8b2fa9cb6");
        assert_eq!(hmac_sha384(&key, &[data]).to_vec(), expected);
    }

    /// RFC 4231 case 2 — a short ASCII key, exercising the zero-pad path.
    #[test]
    fn hmac_sha384_matches_rfc4231_case_2() {
        let expected = hex("af45d2e376484031617f78d2b58a6b1b\
             9c7ef464f5a01b47e42ec3736322445e\
             8e2240ca5e69e2c78b3239ecfab21649");
        assert_eq!(hmac_sha384(b"Jefe", &[b"what do ya want for nothing?"]).to_vec(), expected);
    }

    /// RFC 4231 case 6 — a key LONGER than the 128-byte block, exercising the
    /// hash-the-key branch that a short-key-only test would leave unproven.
    #[test]
    fn hmac_sha384_matches_rfc4231_case_6_with_an_oversized_key() {
        let key = [0xaau8; 131];
        let expected = hex("4ece084485813e9088d2c63a041bc5b4\
             4f9ef1012a2b588f3cd11f05033ac4c6\
             0c2ef6ab4030fe8296248df163f44952");
        assert_eq!(hmac_sha384(&key, &[b"Test Using Larger Than Block-Size Key - Hash Key First"]).to_vec(), expected);
    }

    #[test]
    fn hkdf_is_deterministic_and_every_input_changes_the_output() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        expand(&extract(b"salt", b"ikm"), b"info", &mut a).unwrap();
        expand(&extract(b"salt", b"ikm"), b"info", &mut b).unwrap();
        assert_eq!(a, b);

        for (salt, ikm, info) in [
            (&b"salt2"[..], &b"ikm"[..], &b"info"[..]),
            (&b"salt"[..], &b"ikm2"[..], &b"info"[..]),
            (&b"salt"[..], &b"ikm"[..], &b"info2"[..]),
        ] {
            let mut c = [0u8; 32];
            expand(&extract(salt, ikm), info, &mut c).unwrap();
            assert_ne!(a, c);
        }
    }

    /// Outputs longer than one hash block must chain correctly, not repeat the
    /// first block — the bug a 32-byte-only test cannot see.
    #[test]
    fn hkdf_output_longer_than_one_block_does_not_repeat() {
        let mut long = [0u8; 96];
        expand(&extract(b"salt", b"ikm"), b"info", &mut long).unwrap();
        assert_ne!(&long[..48], &long[48..96]);

        // And a prefix of a long output equals a short output — the standard
        // HKDF property.
        let mut short = [0u8; 32];
        expand(&extract(b"salt", b"ikm"), b"info", &mut short).unwrap();
        assert_eq!(&long[..32], &short[..]);
    }

    #[test]
    fn an_over_long_expansion_is_refused() {
        let mut huge = vec![0u8; 255 * DIGEST_LEN + 1];
        assert_eq!(expand(&extract(b"s", b"i"), b"", &mut huge), Err(ExpandTooLong));
    }

    fn hex(s: &str) -> Vec<u8> {
        let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..clean.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).unwrap())
            .collect()
    }
}
