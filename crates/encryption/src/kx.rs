//! X25519 Diffie-Hellman key agreement — the shared secret behind the
//! sealed channel (`kdf.rs` derives the AEAD key from a password; this
//! module derives it from a key exchange instead).
//!
//! Two secret shapes, matching the upstream `x25519-dalek` distinction:
//!
//! - [`EphemeralSecret`] — fresh per exchange, consumed by
//!   [`EphemeralSecret::diffie_hellman`]. Cannot be reused: the method
//!   takes `self`, not `&self`, so the compiler refuses a second call.
//! - [`StaticSecret`] — long-lived, constructible from 32 bytes (e.g. a
//!   passphrase-derived key from [`crate::kdf`]), reusable across many
//!   exchanges via `&self`.
//!
//! ## Why `EphemeralSecret` is built on `x25519_dalek::StaticSecret`
//!
//! `x25519_dalek::EphemeralSecret` only offers `random_from_rng` (needs a
//! `rand_core::RngCore + CryptoRng`, and `rand_core` is not a dependency
//! of this crate — [`crate::fill_random`] is the only entropy chokepoint)
//! and `random()` (gated behind the `getrandom` feature, which this
//! crate's `x25519-dalek` dependency does not enable). Neither path is
//! reachable without adding a dependency this crate deliberately does
//! not carry. `x25519_dalek::StaticSecret` has no such restriction — it
//! implements `From<[u8; 32]>` — so [`EphemeralSecret`] wraps one,
//! filled from [`crate::fill_random`], and recovers the "used at most
//! once" guarantee at the type level instead: its
//! [`diffie_hellman`](EphemeralSecret::diffie_hellman) takes `self` by
//! value, so the secret is moved-from and unusable after the first call,
//! exactly as the upstream ephemeral type enforces it.
//!
//! ## Low-order peer keys
//!
//! X25519 does not validate that a peer's public key is a full-order
//! point. A small number of "low-order" byte strings — the all-zero
//! string among them — multiply out to the same shared secret
//! (the curve identity) no matter what secret key is on the other side.
//! An attacker able to substitute a peer's public key in transit can
//! hand both ends one of these points and force them onto a shared
//! secret the attacker already knows in advance, without either side
//! detecting anything wrong (they still agree with each other). This is
//! exactly the attack [`x25519_dalek::SharedSecret::was_contributory`]
//! exists to catch, and both [`EphemeralSecret::diffie_hellman`] and
//! [`StaticSecret::diffie_hellman`] call it on every agreement, rejecting
//! the result with [`KxError::LowOrderPeerKey`] instead of silently
//! handing back a secret an attacker already holds.
//!
//! ```
//! use encryption::kx::{EphemeralSecret, StaticSecret};
//!
//! // Alice is ephemeral (fresh per session); Bob is static (a long-lived
//! // identity key, e.g. loaded from storage via `StaticSecret::from_bytes`).
//! let alice = EphemeralSecret::generate().unwrap();
//! let bob = StaticSecret::generate().unwrap();
//!
//! let alice_public = alice.public_key();
//! let bob_public = bob.public_key();
//!
//! let alice_shared = alice.diffie_hellman(&bob_public).unwrap();
//! let bob_shared = bob.diffie_hellman(&alice_public).unwrap();
//!
//! assert_eq!(alice_shared.as_bytes(), bob_shared.as_bytes());
//! ```

use zeroize::Zeroize;

/// Byte length of an X25519 public key.
pub const PUBLIC_KEY_LEN: usize = 32;
/// Byte length of an X25519 shared secret.
pub const SHARED_SECRET_LEN: usize = 32;

/// Key-agreement failure. Field-free on purpose — no secret material,
/// no key echo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KxError {
    /// The platform CSPRNG was unavailable while generating a secret.
    Rng,
    /// The agreed-upon shared secret was the curve identity (all-zero) —
    /// the peer's public key was a low-order point. See the module docs'
    /// "Low-order peer keys" section for why this is rejected rather than
    /// silently returned.
    LowOrderPeerKey,
}

impl core::fmt::Display for KxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KxError::Rng => f.write_str("platform CSPRNG unavailable"),
            KxError::LowOrderPeerKey => f.write_str("peer public key is a low-order point; shared secret rejected"),
        }
    }
}

impl std::error::Error for KxError {}

/// Fill a fresh 32-byte scalar from the platform CSPRNG. The one place
/// this module touches entropy.
fn random_scalar_bytes() -> Result<[u8; 32], KxError> {
    let mut bytes = [0u8; 32];
    crate::fill_random(&mut bytes).map_err(|_| KxError::Rng)?;
    Ok(bytes)
}

/// A public X25519 key: the 32-byte Montgomery `u`-coordinate.
///
/// Not secret — safe to log, store, and send over the wire in the clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicKey(x25519_dalek::PublicKey);

impl PublicKey {
    /// Wrap a raw 32-byte public key (e.g. received from a peer).
    pub fn from_bytes(bytes: [u8; PUBLIC_KEY_LEN]) -> Self {
        PublicKey(x25519_dalek::PublicKey::from(bytes))
    }

    /// Borrow the raw 32 bytes (e.g. to send to a peer).
    pub fn as_bytes(&self) -> &[u8; PUBLIC_KEY_LEN] {
        self.0.as_bytes()
    }
}

/// The result of a Diffie-Hellman agreement.
///
/// Re-exported directly from `x25519-dalek`: it already zeroizes on drop
/// (this crate's `x25519-dalek` dependency enables the `zeroize`
/// feature) and deliberately has **no `Debug` impl** — the same choice
/// [`crate::kdf::DerivedKey`] makes. A shared secret must never be
/// printable; logging one is a key-disclosure bug waiting to happen.
pub use x25519_dalek::SharedSecret;

/// Reject a shared secret computed from a low-order peer public key.
/// Shared by both [`EphemeralSecret::diffie_hellman`] and
/// [`StaticSecret::diffie_hellman`].
fn checked(shared: SharedSecret) -> Result<SharedSecret, KxError> {
    if shared.was_contributory() {
        Ok(shared)
    } else {
        Err(KxError::LowOrderPeerKey)
    }
}

/// A fresh, single-use X25519 secret.
///
/// Generated from the platform CSPRNG; consumed by
/// [`diffie_hellman`](Self::diffie_hellman), which takes `self` so the
/// compiler refuses a second agreement with the same secret. See the
/// module docs for why this wraps `x25519_dalek::StaticSecret` rather
/// than `x25519_dalek::EphemeralSecret`.
pub struct EphemeralSecret(x25519_dalek::StaticSecret);

impl EphemeralSecret {
    /// Generate a fresh secret from the platform CSPRNG.
    pub fn generate() -> Result<Self, KxError> {
        let mut bytes = random_scalar_bytes()?;
        let secret = x25519_dalek::StaticSecret::from(bytes);
        bytes.zeroize();
        Ok(EphemeralSecret(secret))
    }

    /// The public key corresponding to this secret, to send to the peer.
    pub fn public_key(&self) -> PublicKey {
        PublicKey(x25519_dalek::PublicKey::from(&self.0))
    }

    /// Perform the agreement against a peer's public key, consuming this
    /// secret so it cannot be reused. Rejects a low-order `their_public`
    /// (see the module docs).
    pub fn diffie_hellman(self, their_public: &PublicKey) -> Result<SharedSecret, KxError> {
        checked(self.0.diffie_hellman(&their_public.0))
    }
}

/// A long-lived X25519 secret, reusable across many agreements.
///
/// Construct from the CSPRNG ([`StaticSecret::generate`]) or from 32
/// bytes the caller already owns ([`StaticSecret::from_bytes`] — e.g. a
/// key loaded from storage, or derived by [`crate::kdf`] from a
/// passphrase).
pub struct StaticSecret(x25519_dalek::StaticSecret);

impl StaticSecret {
    /// Generate a fresh secret from the platform CSPRNG.
    pub fn generate() -> Result<Self, KxError> {
        let mut bytes = random_scalar_bytes()?;
        let secret = x25519_dalek::StaticSecret::from(bytes);
        bytes.zeroize();
        Ok(StaticSecret(secret))
    }

    /// Load a secret from 32 bytes the caller already owns.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        StaticSecret(x25519_dalek::StaticSecret::from(bytes))
    }

    /// The public key corresponding to this secret.
    pub fn public_key(&self) -> PublicKey {
        PublicKey(x25519_dalek::PublicKey::from(&self.0))
    }

    /// Perform the agreement against a peer's public key. Does not
    /// consume `self` — the same secret may agree with many peers.
    /// Rejects a low-order `their_public` (see the module docs).
    pub fn diffie_hellman(&self, their_public: &PublicKey) -> Result<SharedSecret, KxError> {
        checked(self.0.diffie_hellman(&their_public.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 7748 §5.2 X25519 ladder test vector ("vectorset1"), copied
    /// verbatim from the vendored `x25519-dalek` 2.0.1 test suite
    /// (`tests/x25519_tests.rs::rfc7748_ladder_test1_vectorset1`) rather
    /// than transcribed from memory, so the bytes are known-good rather
    /// than guessed. Routed through this module's typed API
    /// (`StaticSecret::from_bytes` + `diffie_hellman`) rather than the
    /// bare `x25519()` function, since that typed path is what this
    /// module actually exposes and it calls the identical
    /// `mul_clamped` internally.
    #[test]
    fn rfc7748_ladder_test_vector() {
        let scalar: [u8; 32] = [
            0xa5, 0x46, 0xe3, 0x6b, 0xf0, 0x52, 0x7c, 0x9d, 0x3b, 0x16, 0x15, 0x4b, 0x82, 0x46, 0x5e, 0xdd, 0x62, 0x14,
            0x4c, 0x0a, 0xc1, 0xfc, 0x5a, 0x18, 0x50, 0x6a, 0x22, 0x44, 0xba, 0x44, 0x9a, 0xc4,
        ];
        let point: [u8; 32] = [
            0xe6, 0xdb, 0x68, 0x67, 0x58, 0x30, 0x30, 0xdb, 0x35, 0x94, 0xc1, 0xa4, 0x24, 0xb1, 0x5f, 0x7c, 0x72, 0x66,
            0x24, 0xec, 0x26, 0xb3, 0x35, 0x3b, 0x10, 0xa9, 0x03, 0xa6, 0xd0, 0xab, 0x1c, 0x4c,
        ];
        let expected: [u8; 32] = [
            0xc3, 0xda, 0x55, 0x37, 0x9d, 0xe9, 0xc6, 0x90, 0x8e, 0x94, 0xea, 0x4d, 0xf2, 0x8d, 0x08, 0x4f, 0x32, 0xec,
            0xcf, 0x03, 0x49, 0x1c, 0x71, 0xf7, 0x54, 0xb4, 0x07, 0x55, 0x77, 0xa2, 0x85, 0x52,
        ];

        let secret = StaticSecret::from_bytes(scalar);
        let peer = PublicKey::from_bytes(point);
        let shared = secret
            .diffie_hellman(&peer)
            .expect("known-good ladder vector must be contributory");
        assert_eq!(shared.as_bytes(), &expected);
    }

    #[test]
    fn alice_and_bob_derive_the_same_shared_secret() {
        let alice = EphemeralSecret::generate().unwrap();
        let bob = StaticSecret::generate().unwrap();

        let alice_public = alice.public_key();
        let bob_public = bob.public_key();

        let alice_shared = alice.diffie_hellman(&bob_public).unwrap();
        let bob_shared = bob.diffie_hellman(&alice_public).unwrap();

        assert_eq!(alice_shared.as_bytes(), bob_shared.as_bytes());
    }

    #[test]
    fn a_different_peer_key_gives_a_different_shared_secret() {
        let alice = StaticSecret::generate().unwrap();
        let bob = StaticSecret::generate().unwrap();
        let carol = StaticSecret::generate().unwrap();

        let shared_with_bob = alice.diffie_hellman(&bob.public_key()).unwrap();
        let shared_with_carol = alice.diffie_hellman(&carol.public_key()).unwrap();

        assert_ne!(shared_with_bob.as_bytes(), shared_with_carol.as_bytes());
    }

    /// An all-zero peer public key is a low-order point: X25519 would
    /// otherwise happily compute a shared secret from it, and that
    /// secret is the curve identity regardless of our own secret key —
    /// exactly the "Mallory replaces both public keys with zero" attack
    /// described in the module docs. This must be refused, not returned.
    #[test]
    fn an_all_zero_peer_public_key_is_rejected() {
        let alice = StaticSecret::generate().unwrap();
        let low_order_peer = PublicKey::from_bytes([0u8; 32]);

        match alice.diffie_hellman(&low_order_peer) {
            Err(e) => assert_eq!(e, KxError::LowOrderPeerKey),
            Ok(_) => panic!("a low-order (all-zero) peer public key must be rejected"),
        }
    }

    #[test]
    fn generated_secrets_differ_across_calls() {
        // The raw secret is never exposed for comparison (by design — see
        // `EphemeralSecret`/`StaticSecret`), so the CSPRNG-is-actually-used
        // check compares the public keys derived from two independently
        // generated secrets instead. Identical public keys would mean
        // identical secrets, which would mean the CSPRNG was not consulted.
        let a = EphemeralSecret::generate().unwrap();
        let b = EphemeralSecret::generate().unwrap();
        assert_ne!(a.public_key().as_bytes(), b.public_key().as_bytes());
    }
}
