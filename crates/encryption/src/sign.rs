//! Ed25519 signatures — licence stamping, audit-witness signing.
//!
//! Ed25519 is deterministic (no per-signature randomness to get wrong),
//! uses SHA-512 internally as the scheme mandates, and produces 32-byte
//! public keys with 64-byte signatures. Seeds come from `getrandom`
//! only, or from a caller-supplied 32-byte seed (e.g. one derived by
//! [`crate::kdf`] from a passphrase, for a deterministic identity).

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// Byte length of an Ed25519 public key.
pub const PUBLIC_KEY_LEN: usize = 32;
/// Byte length of an Ed25519 seed (private key).
pub const SEED_LEN: usize = 32;
/// Byte length of an Ed25519 signature.
pub const SIGNATURE_LEN: usize = 64;

/// Signing failure (currently only: no entropy for key generation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignError;

impl core::fmt::Display for SignError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("signing-key generation failed")
    }
}

impl std::error::Error for SignError {}

/// An Ed25519 keypair. The seed zeroizes on drop (via `ed25519-dalek`'s
/// `zeroize` feature).
pub struct Keypair {
    signing: SigningKey,
}

impl Keypair {
    /// Generate a fresh keypair from the platform CSPRNG.
    pub fn generate() -> Result<Self, SignError> {
        let mut seed = [0u8; SEED_LEN];
        crate::fill_random(&mut seed).map_err(|_| SignError)?;
        Ok(Self::from_seed(&seed))
    }

    /// Deterministic keypair from a 32-byte seed the caller owns.
    pub fn from_seed(seed: &[u8; SEED_LEN]) -> Self {
        Keypair {
            signing: SigningKey::from_bytes(seed),
        }
    }

    /// The 32-byte public half, safe to publish.
    pub fn public_key(&self) -> [u8; PUBLIC_KEY_LEN] {
        self.signing.verifying_key().to_bytes()
    }

    /// Sign `message`; Ed25519 signatures are deterministic per
    /// (seed, message).
    pub fn sign(&self, message: &[u8]) -> [u8; SIGNATURE_LEN] {
        self.signing.sign(message).to_bytes()
    }
}

/// Verify `signature` over `message` against a 32-byte public key.
/// Returns `false` for malformed keys/signatures as well as honest
/// mismatches — callers get one bit, nothing to oracle on.
pub fn verify(public_key: &[u8; PUBLIC_KEY_LEN], message: &[u8], signature: &[u8; SIGNATURE_LEN]) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let sig = Signature::from_bytes(signature);
    vk.verify(message, &sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_round_trip() {
        let kp = Keypair::generate().unwrap();
        let sig = kp.sign(b"licence: 42 seats");
        assert!(verify(&kp.public_key(), b"licence: 42 seats", &sig));
    }

    #[test]
    fn tampered_message_rejected() {
        let kp = Keypair::generate().unwrap();
        let sig = kp.sign(b"licence: 42 seats");
        assert!(!verify(&kp.public_key(), b"licence: 43 seats", &sig));
    }

    #[test]
    fn wrong_key_rejected() {
        let a = Keypair::generate().unwrap();
        let b = Keypair::generate().unwrap();
        let sig = a.sign(b"msg");
        assert!(!verify(&b.public_key(), b"msg", &sig));
    }

    #[test]
    fn deterministic_from_seed() {
        let seed = [0x11u8; SEED_LEN];
        let a = Keypair::from_seed(&seed);
        let b = Keypair::from_seed(&seed);
        assert_eq!(a.public_key(), b.public_key());
        assert_eq!(a.sign(b"m"), b.sign(b"m"));
    }

    #[test]
    fn rfc8032_test_vector_1() {
        // RFC 8032 §7.1 TEST 1: empty message.
        let seed = hex(b"9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let expected_pk = hex(b"d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
        let kp = Keypair::from_seed(&seed);
        assert_eq!(kp.public_key(), expected_pk);
        let sig = kp.sign(b"");
        assert!(verify(&kp.public_key(), b"", &sig));
    }

    fn hex<const N: usize>(s: &[u8]) -> [u8; N] {
        fn nibble(c: u8) -> u8 {
            match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => panic!("bad hex"),
            }
        }
        let mut out = [0u8; N];
        for (i, pair) in s.chunks(2).enumerate() {
            out[i] = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        out
    }
}
