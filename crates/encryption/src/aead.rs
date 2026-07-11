//! XChaCha20-Poly1305 seal/open with explicit 24-byte nonces.
//!
//! The extended (192-bit) nonce is the whole point: it is safe to draw
//! nonces at random from the CSPRNG for the lifetime of a key, so there
//! is no counter state to persist or synchronise between a browser tab
//! and a server. Callers normally use [`crate::envelope`], which manages
//! nonce generation and layout; this module is the raw primitive.
//!
//! ## Deliberately NOT hand-composed from `ndarray::simd::chacha20_keystream`
//!
//! `ndarray` now ships a hardware-accelerated ChaCha20 keystream
//! (`ndarray::simd::chacha20_keystream`, AVX-512 / wasm128, byte-parity-proven
//! against RustCrypto). It is tempting to swap this AEAD's keystream to that
//! primitive for speed — **do not.** XChaCha20-Poly1305 is not just a keystream:
//! it is HChaCha20 subkey derivation + the Poly1305 one-time-key + MAC framing +
//! the AAD/length encoding. Re-wiring only the keystream means re-implementing
//! that authenticated composition by hand, which is exactly the "roll your own
//! AEAD" footgun the stack forbids. The vetted RustCrypto `XChaCha20Poly1305`
//! construction stays here. The accelerated primitive is for *raw-stream* use
//! sites whose caller already owns a vetted MAC/framing — not this module.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};

/// Byte length of an XChaCha20-Poly1305 nonce.
pub const NONCE_LEN: usize = 24;
/// Byte length of the Poly1305 authentication tag appended to ciphertext.
pub const TAG_LEN: usize = 16;

/// AEAD failure. Deliberately carries nothing — a decryption failure
/// must be indistinguishable whether the key, nonce, ciphertext, or AAD
/// was wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AeadError;

impl core::fmt::Display for AeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("AEAD seal/open failed")
    }
}

impl std::error::Error for AeadError {}

/// Encrypt `plaintext` under `key`/`nonce`, authenticating `aad`
/// alongside it. Returns ciphertext with the 16-byte tag appended.
pub fn seal_with_key(
    key: &[u8; 32], nonce: &[u8; NONCE_LEN], aad: &[u8], plaintext: &[u8],
) -> Result<Vec<u8>, AeadError> {
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .encrypt(XNonce::from_slice(nonce), Payload { msg: plaintext, aad })
        .map_err(|_| AeadError)
}

/// Decrypt and authenticate. Fails if the key, nonce, AAD, or a single
/// bit of ciphertext is wrong.
pub fn open_with_key(
    key: &[u8; 32], nonce: &[u8; NONCE_LEN], aad: &[u8], ciphertext: &[u8],
) -> Result<Vec<u8>, AeadError> {
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .decrypt(XNonce::from_slice(nonce), Payload { msg: ciphertext, aad })
        .map_err(|_| AeadError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_with_aad() {
        let key = [0x42u8; 32];
        let nonce = [0x24u8; NONCE_LEN];
        let ct = seal_with_key(&key, &nonce, b"header", b"secret").unwrap();
        assert_eq!(ct.len(), b"secret".len() + TAG_LEN);
        let pt = open_with_key(&key, &nonce, b"header", &ct).unwrap();
        assert_eq!(pt, b"secret");
    }

    #[test]
    fn wrong_key_fails() {
        let ct = seal_with_key(&[1u8; 32], &[0u8; NONCE_LEN], b"", b"x").unwrap();
        assert!(open_with_key(&[2u8; 32], &[0u8; NONCE_LEN], b"", &ct).is_err());
    }

    #[test]
    fn wrong_aad_fails() {
        let key = [3u8; 32];
        let nonce = [4u8; NONCE_LEN];
        let ct = seal_with_key(&key, &nonce, b"aad-a", b"x").unwrap();
        assert!(open_with_key(&key, &nonce, b"aad-b", &ct).is_err());
    }

    #[test]
    fn bit_flip_fails() {
        let key = [5u8; 32];
        let nonce = [6u8; NONCE_LEN];
        let mut ct = seal_with_key(&key, &nonce, b"", b"payload").unwrap();
        ct[0] ^= 0x01;
        assert!(open_with_key(&key, &nonce, b"", &ct).is_err());
    }
}
