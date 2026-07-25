//! Signed content bundles — a distributable, verifiable unit of UI
//! projection (a view, a mask, a render adapter, or a recipe set).
//!
//! A bundle is minted once, offline, by a publisher's [`crate::sign::Keypair`]
//! and then handed to clients however content normally travels (a CDN, a
//! sync channel, a USB stick — the transport is not this module's concern).
//! The client's only obligation is: **refuse to render anything that does
//! not verify.** [`verify`] is the single gate between "bytes arrived" and
//! "bytes are trusted", and it is the only supported way to read a bundle's
//! body.
//!
//! ## Byte layout (fixed, little-endian, parsed by hand — no serde)
//!
//! ```text
//! offset  len  field
//!  0       4   magic       = b"ADAB"
//!  4       1   version     = 1
//!  5       1   kind        1 = view, 2 = mask, 3 = adapter, 4 = recipe-set
//!  6       2   flags       bit0 = revocable, bit1 = requires-session
//!  8       4   classid     u32, the concept this projection targets
//! 12       8   bundle_id   u64, monotonic per publisher
//! 20       8   issued_at   u64, unix seconds
//! 28      32   publisher   Ed25519 public key of the signer
//! 60       4   body_len    u32
//! 64       …   body
//!  …      64   signature   Ed25519 over bytes [0 .. 64 + body_len)
//! ```
//!
//! `HEADER_LEN` (64) is everything up to and including `body_len`; the
//! body follows immediately, and the signature is always the trailing 64
//! bytes of the blob.
//!
//! ## The four things that make this a security boundary, not a format
//!
//! 1. **The signature covers the header *and* the body.** Every header
//!    field — `kind`, `classid`, `version`, `flags`, all of it — is inside
//!    the signed range `[0 .. 64 + body_len)`. That means a `kind = Mask`
//!    header can never be spliced onto a `kind = View` body signed by the
//!    same publisher, and a `classid` cannot be swapped between two
//!    otherwise-legitimate bundles: any such splice changes the signed
//!    bytes, and [`verify`] rejects it. The header is not a label glued to
//!    the payload; it is part of what got signed.
//! 2. **`publisher` is carried in the blob, but it is not the trust
//!    anchor.** [`verify`] takes a caller-supplied `allowed_publishers`
//!    list and rejects any bundle whose embedded key is not on it —
//!    including a bundle that is perfectly, validly self-signed. Carrying
//!    the key in the blob exists so a rejection can say "unknown
//!    publisher" instead of failing opaquely; trust still comes entirely
//!    from the caller's allowlist, never from the blob asserting its own
//!    legitimacy. (As a consequence of point 1, an attacker also cannot
//!    forge the `publisher` field to impersonate an allowed key: the
//!    signature only validates under the key that actually produced it, so
//!    claiming a different `publisher` than the true signer just makes the
//!    signature check fail.)
//! 3. **Verification happens before any use of the body.** [`verify`]
//!    checks the signature and only *then* returns `&[u8]` for the body —
//!    there is no separate "peek at the body" accessor and no path that
//!    parses fields out of an unverified blob. Parse-then-verify (read the
//!    header, act on it, check the signature as an afterthought) is the
//!    bug class this API is shaped to make impossible: verification is not
//!    a step you can accidentally do second.
//! 4. **Why a byte layout and not signed JSON.** JSON has more than one
//!    valid serialization of the same logical document (key order,
//!    whitespace, numeric formatting, duplicate-key handling), so "the
//!    bytes that were signed" and "the bytes a second parser reads" can
//!    disagree — a documented class of signature-bypass bugs. A fixed
//!    little-endian layout has exactly one reading: byte 8 is always the
//!    low byte of `classid`, on every implementation, forever. There is no
//!    canonicalization step to get wrong because there is no alternate
//!    form to canonicalize from.
//!
//! ## Example
//!
//! ```
//! use encryption::bundle::{sign, verify, BundleFlags, BundleHeader, BundleKind};
//! use encryption::sign::Keypair;
//!
//! let publisher = Keypair::generate().unwrap();
//!
//! let header = BundleHeader {
//!     kind: BundleKind::View,
//!     flags: BundleFlags {
//!         revocable: true,
//!         requires_session: false,
//!     },
//!     classid: 0x00A0_0042,
//!     bundle_id: 7,
//!     issued_at: 1_753_000_000,
//!     publisher: [0u8; 32], // overwritten by `sign` with the real key
//! };
//!
//! let blob = sign(header, b"<view>...</view>", &publisher);
//!
//! let allowed = [publisher.public_key()];
//! let (verified, body) = verify(&blob, &allowed).unwrap();
//! assert_eq!(body, b"<view>...</view>");
//! assert_eq!(verified.classid, 0x00A0_0042);
//! assert_eq!(verified.kind, BundleKind::View);
//! ```

use crate::sign::{Keypair, PUBLIC_KEY_LEN, SIGNATURE_LEN};

/// Bundle magic: "ADAB" (Ada bundle).
pub const MAGIC: [u8; 4] = *b"ADAB";
/// Current bundle layout version.
pub const VERSION: u8 = 1;
/// Length of the fixed header preceding the body (through `body_len`).
pub const HEADER_LEN: usize = 64;

/// What kind of projection a bundle carries. Part of the signed header —
/// see module note #1: a `kind` cannot be swapped onto a different body
/// without breaking the signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleKind {
    /// A UI view.
    View = 1,
    /// A visibility/redaction mask.
    Mask = 2,
    /// A render adapter.
    Adapter = 3,
    /// A named set of recipes.
    RecipeSet = 4,
}

impl BundleKind {
    fn to_u8(self) -> u8 {
        self as u8
    }

    fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::View),
            2 => Some(Self::Mask),
            3 => Some(Self::Adapter),
            4 => Some(Self::RecipeSet),
            _ => None,
        }
    }
}

/// The two-bit flag field, decoded into named booleans. No external
/// bitflags dependency — this crate hand-rolls its layouts, flags
/// included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BundleFlags {
    /// The publisher may revoke this bundle after issue (advisory to the
    /// client; revocation itself is out of scope for this module).
    pub revocable: bool,
    /// The client must have an authenticated session before rendering
    /// this bundle's body.
    pub requires_session: bool,
}

impl BundleFlags {
    const REVOCABLE_BIT: u16 = 1 << 0;
    const REQUIRES_SESSION_BIT: u16 = 1 << 1;

    fn to_bits(self) -> u16 {
        (if self.revocable { Self::REVOCABLE_BIT } else { 0 })
            | (if self.requires_session {
                Self::REQUIRES_SESSION_BIT
            } else {
                0
            })
    }

    fn from_bits(bits: u16) -> Self {
        Self {
            revocable: bits & Self::REVOCABLE_BIT != 0,
            requires_session: bits & Self::REQUIRES_SESSION_BIT != 0,
        }
    }
}

/// The bundle header, typed. Every field here is inside the signed range —
/// see module note #1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundleHeader {
    /// What kind of projection this is.
    pub kind: BundleKind,
    /// Publisher-set behavior flags.
    pub flags: BundleFlags,
    /// The concept (classid) this projection targets.
    pub classid: u32,
    /// Monotonic id, scoped to the publisher.
    pub bundle_id: u64,
    /// Unix seconds at signing time.
    pub issued_at: u64,
    /// The signer's Ed25519 public key. Set by [`sign`] from the actual
    /// signing key — whatever value the caller puts here before calling
    /// [`sign`] is discarded, so it is not possible to mint a bundle that
    /// misrepresents its own signer (see module note #2).
    pub publisher: [u8; PUBLIC_KEY_LEN],
}

/// Why a bundle could not be minted or verified. Field-free — a rejection
/// never says which byte looked wrong, only which class of problem it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleError {
    /// The blob is not a bundle: too short, bad magic, bad `kind`, or its
    /// declared `body_len` does not match the buffer it was found in.
    Malformed,
    /// The blob's layout version is newer than this code understands.
    UnsupportedVersion,
    /// The embedded `publisher` key is not in the caller's allowlist.
    /// This includes a bundle that is validly self-signed — an allowlist
    /// hit is required regardless of how good the signature is.
    UnknownPublisher,
    /// The Ed25519 signature does not validate over the header+body.
    BadSignature,
}

impl core::fmt::Display for BundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BundleError::Malformed => f.write_str("not a well-formed bundle"),
            BundleError::UnsupportedVersion => f.write_str("unsupported bundle version"),
            BundleError::UnknownPublisher => f.write_str("publisher is not in the allowlist"),
            BundleError::BadSignature => f.write_str("signature does not validate"),
        }
    }
}

impl std::error::Error for BundleError {}

/// Sign `body` under `header`, producing a distributable bundle blob.
///
/// `header.publisher` is overwritten with `signing_key.public_key()` before
/// signing — the caller does not need to (and cannot usefully) set it
/// themselves; see the field doc on [`BundleHeader::publisher`].
///
/// # Panics
///
/// Panics if `body.len()` exceeds `u32::MAX`. This is a local encoding
/// precondition on the signer's own input, not a check against untrusted
/// data — nothing this crate distributes is anywhere near 4 GiB.
pub fn sign(mut header: BundleHeader, body: &[u8], signing_key: &Keypair) -> Vec<u8> {
    header.publisher = signing_key.public_key();
    let body_len: u32 = body
        .len()
        .try_into()
        .expect("bundle body exceeds u32::MAX bytes");

    let head = encode_header(&header, body_len);
    let mut signed_region = Vec::with_capacity(HEADER_LEN + body.len());
    signed_region.extend_from_slice(&head);
    signed_region.extend_from_slice(body);

    let signature = signing_key.sign(&signed_region);
    let mut blob = signed_region;
    blob.extend_from_slice(&signature);
    blob
}

/// Verify `blob` against `allowed_publishers`, returning the parsed header
/// and a borrowed slice of the body — never a copy.
///
/// The body is returned *only* if every check passes: well-formed layout,
/// a supported version, a `publisher` present in `allowed_publishers`, and
/// a valid Ed25519 signature over the header+body. There is no way to
/// reach the body bytes through this module without passing all four (see
/// module note #3).
pub fn verify<'b>(
    blob: &'b [u8], allowed_publishers: &[[u8; PUBLIC_KEY_LEN]],
) -> Result<(BundleHeader, &'b [u8]), BundleError> {
    if blob.len() < HEADER_LEN + SIGNATURE_LEN {
        return Err(BundleError::Malformed);
    }

    let (header, body_len) = decode_header(blob)?;
    let body_len = body_len as usize;

    // The length field is untrusted input: it is checked against the
    // buffer it actually arrived in before it is used for any slicing.
    // Trusting a declared length over the real buffer size is the classic
    // parser bug this guards against.
    let expected_len = HEADER_LEN
        .checked_add(body_len)
        .and_then(|v| v.checked_add(SIGNATURE_LEN))
        .ok_or(BundleError::Malformed)?;
    if blob.len() != expected_len {
        return Err(BundleError::Malformed);
    }

    // A publisher key is public material, so an early-out comparison leaks
    // nothing worth having; the signature check below is the one that must
    // stay constant-time, and ed25519-dalek owns that.
    if !allowed_publishers.contains(&header.publisher) {
        return Err(BundleError::UnknownPublisher);
    }

    let signed_region = &blob[..HEADER_LEN + body_len];
    let mut signature = [0u8; SIGNATURE_LEN];
    signature.copy_from_slice(&blob[HEADER_LEN + body_len..expected_len]);

    if !crate::sign::verify(&header.publisher, signed_region, &signature) {
        return Err(BundleError::BadSignature);
    }

    let body = &blob[HEADER_LEN..HEADER_LEN + body_len];
    Ok((header, body))
}

fn encode_header(header: &BundleHeader, body_len: u32) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = VERSION;
    h[5] = header.kind.to_u8();
    h[6..8].copy_from_slice(&header.flags.to_bits().to_le_bytes());
    h[8..12].copy_from_slice(&header.classid.to_le_bytes());
    h[12..20].copy_from_slice(&header.bundle_id.to_le_bytes());
    h[20..28].copy_from_slice(&header.issued_at.to_le_bytes());
    h[28..28 + PUBLIC_KEY_LEN].copy_from_slice(&header.publisher);
    h[60..64].copy_from_slice(&body_len.to_le_bytes());
    h
}

fn decode_header(blob: &[u8]) -> Result<(BundleHeader, u32), BundleError> {
    if blob.len() < HEADER_LEN || blob[0..4] != MAGIC {
        return Err(BundleError::Malformed);
    }
    if blob[4] != VERSION {
        return Err(BundleError::UnsupportedVersion);
    }
    let kind = BundleKind::from_u8(blob[5]).ok_or(BundleError::Malformed)?;
    let flags = BundleFlags::from_bits(u16::from_le_bytes([blob[6], blob[7]]));
    let classid = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    let bundle_id = u64::from_le_bytes(blob[12..20].try_into().unwrap());
    let issued_at = u64::from_le_bytes(blob[20..28].try_into().unwrap());
    let mut publisher = [0u8; PUBLIC_KEY_LEN];
    publisher.copy_from_slice(&blob[28..28 + PUBLIC_KEY_LEN]);
    let body_len = u32::from_le_bytes(blob[60..64].try_into().unwrap());

    Ok((
        BundleHeader {
            kind,
            flags,
            classid,
            bundle_id,
            issued_at,
            publisher,
        },
        body_len,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> BundleHeader {
        BundleHeader {
            kind: BundleKind::View,
            flags: BundleFlags {
                revocable: true,
                requires_session: false,
            },
            classid: 0x1234_5678,
            bundle_id: 99,
            issued_at: 1_753_000_000,
            publisher: [0u8; PUBLIC_KEY_LEN], // sign() overwrites this
        }
    }

    #[test]
    fn round_trip_returns_the_exact_body_bytes() {
        let kp = Keypair::generate().unwrap();
        let blob = sign(sample_header(), b"the projection body", &kp);
        let (header, body) = verify(&blob, &[kp.public_key()]).unwrap();
        assert_eq!(body, b"the projection body");
        assert_eq!(header.classid, 0x1234_5678);
        assert_eq!(header.bundle_id, 99);
        assert_eq!(header.kind, BundleKind::View);
        assert!(header.flags.revocable);
        assert!(!header.flags.requires_session);
        assert_eq!(header.publisher, kp.public_key());
    }

    #[test]
    fn a_mutated_body_byte_fails_to_verify() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"the projection body", &kp);
        let body_start = HEADER_LEN;
        blob[body_start] ^= 0x01;
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::BadSignature);
    }

    /// Finding #1, proven: the signature covers the header, so `classid`
    /// cannot be swapped onto a different, otherwise-validly-signed body.
    #[test]
    fn a_mutated_classid_bit_fails_to_verify() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"body", &kp);
        blob[8] ^= 0x01; // low byte of classid, inside the signed header
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::BadSignature);
    }

    /// Same finding, different field: flipping `kind` from `View` (1) to
    /// `Adapter` (3) is still a structurally valid header — the signature,
    /// not the parser, is what catches the swap.
    #[test]
    fn a_mutated_kind_byte_fails_to_verify() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"body", &kp);
        assert_eq!(blob[5], BundleKind::View.to_u8());
        blob[5] ^= 0b10; // View(1) -> Adapter(3), still a valid kind byte
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::BadSignature);
    }

    #[test]
    fn an_unlisted_publisher_is_rejected() {
        let kp = Keypair::generate().unwrap();
        let other = Keypair::generate().unwrap();
        let blob = sign(sample_header(), b"body", &kp);
        assert_eq!(verify(&blob, &[other.public_key()]).unwrap_err(), BundleError::UnknownPublisher);
    }

    #[test]
    fn an_empty_allowlist_rejects_everything() {
        let kp = Keypair::generate().unwrap();
        let blob = sign(sample_header(), b"body", &kp);
        assert_eq!(verify(&blob, &[]).unwrap_err(), BundleError::UnknownPublisher);
    }

    #[test]
    fn every_truncation_length_is_rejected_without_panicking() {
        let kp = Keypair::generate().unwrap();
        let blob = sign(sample_header(), b"the projection body", &kp);
        for len in 0..blob.len() {
            let truncated = &blob[..len];
            assert!(
                verify(truncated, &[kp.public_key()]).is_err(),
                "truncation to {len} bytes must be rejected, not accepted"
            );
        }
        // The full, untruncated blob is the one length that must succeed.
        assert!(verify(&blob, &[kp.public_key()]).is_ok());
    }

    /// The classic parser bug: trust the length field instead of the
    /// buffer. Here `body_len` still says what it said at signing time,
    /// but the buffer has grown past it.
    #[test]
    fn a_body_len_that_disagrees_with_the_buffer_is_rejected() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"body", &kp);
        blob.push(0xAA); // buffer is now longer than body_len promises
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::Malformed);
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"body", &kp);
        blob[0] = b'X';
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::Malformed);
    }

    #[test]
    fn wrong_version_is_rejected() {
        let kp = Keypair::generate().unwrap();
        let mut blob = sign(sample_header(), b"body", &kp);
        blob[4] = 99;
        assert_eq!(verify(&blob, &[kp.public_key()]).unwrap_err(), BundleError::UnsupportedVersion);
    }

    #[test]
    fn empty_body_round_trips() {
        let kp = Keypair::generate().unwrap();
        let blob = sign(sample_header(), b"", &kp);
        assert_eq!(blob.len(), HEADER_LEN + SIGNATURE_LEN);
        let (_, body) = verify(&blob, &[kp.public_key()]).unwrap();
        assert!(body.is_empty());
    }
}
