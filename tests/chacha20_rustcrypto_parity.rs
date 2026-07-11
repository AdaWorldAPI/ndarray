//! Cross-implementation trust gate for `ndarray::simd::chacha20_keystream`.
//!
//! The module KAT (`src/simd_crypto.rs`) pins the scalar reference to the single
//! RFC 8439 §2.3.2 vector, and `chacha20_keystream_dispatch_parity_scalar` pins
//! the AVX-512 backend to that scalar reference. This integration test closes the
//! remaining gap: it proves the *whole dispatcher* is byte-for-byte identical to
//! an **independent, vetted implementation** — RustCrypto's `chacha20::ChaCha20`
//! (the IETF 96-bit-nonce / 32-bit-counter variant, RFC 8439) — across a wide
//! space of random keys, nonces, counters, and lengths. This is the
//! "reference + KAT, then vectorize with parity" discipline extended to a second
//! oracle: no consumer (the `encryption` AEAD, a future XChaCha vault) should
//! draw keystream from the accelerated primitive until it is proven equivalent
//! to the trusted implementation everywhere, not just at one KAT point.
//!
//! ChaCha20 is ARX, so the accelerated path is constant-time by construction;
//! this test guards *correctness*, which is the property a hand-vectorization
//! can actually get wrong.

use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use chacha20::ChaCha20;
use ndarray::simd::{chacha20_block, chacha20_keystream, chacha20_state};

/// Deterministic SplitMix64 — a stand-in for a PRNG so the vectors are fixed and
/// reproducible without a `rand` dev-dependency (and without `Math.random`-style
/// nondeterminism that would make a failure impossible to bisect).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

/// The trusted oracle: `n_blocks` of RustCrypto ChaCha20 keystream starting at
/// block `counter` (produced by encrypting a zero buffer).
fn rustcrypto_keystream(key: &[u8; 32], counter: u32, nonce: &[u8; 12], n_blocks: usize) -> Vec<u8> {
    let mut cipher = ChaCha20::new_from_slices(key, nonce).expect("valid key/nonce lengths");
    // Seek to block `counter` (position is in bytes; each block is 64 bytes).
    cipher.seek(u64::from(counter) * 64);
    let mut buf = vec![0u8; n_blocks * 64];
    cipher.apply_keystream(&mut buf);
    buf
}

/// The primitive under test, flattened to a byte stream for comparison.
fn ndarray_keystream(key: &[u8; 32], counter: u32, nonce: &[u8; 12], n_blocks: usize) -> Vec<u8> {
    let mut blocks = vec![[0u8; 64]; n_blocks];
    chacha20_keystream(key, counter, nonce, &mut blocks);
    blocks.concat()
}

/// The dispatcher (AVX-512 backend when present, scalar otherwise) must equal the
/// RustCrypto oracle across a broad, deterministic sample of the parameter space.
#[test]
fn keystream_matches_rustcrypto_across_vectors() {
    let mut rng = SplitMix64(0x0DDB_1A5E_5EED_1234);

    // Lengths chosen to exercise: sub-stride, exactly one AVX-512 stride (16),
    // multi-stride, and every ragged-tail size in between.
    let block_counts = [1usize, 2, 7, 15, 16, 17, 31, 33, 40, 64];

    for &n in &block_counts {
        // A handful of independent (key, nonce, counter) draws per length.
        for _ in 0..8 {
            let mut key = [0u8; 32];
            let mut nonce = [0u8; 12];
            rng.fill(&mut key);
            rng.fill(&mut nonce);
            // Keep counter + n well inside u32 so the RustCrypto oracle (which
            // refuses to wrap its 32-bit counter) stays in its valid range; the
            // wrapping edge is covered separately by the scalar-parity unit test.
            let counter = (rng.next_u64() as u32) & 0x00FF_FFFF;

            let got = ndarray_keystream(&key, counter, &nonce, n);
            let want = rustcrypto_keystream(&key, counter, &nonce, n);
            assert_eq!(
                got, want,
                "keystream mismatch vs RustCrypto: n={n} blocks, counter={counter}, key[0]={}, nonce[0]={}",
                key[0], nonce[0]
            );
        }
    }
}

/// The single-block scalar primitive `chacha20_block` (fed via `chacha20_state`)
/// must equal RustCrypto's first block for the same inputs — pins the low-level
/// entry point, not just the batched dispatcher.
#[test]
fn single_block_matches_rustcrypto() {
    let mut rng = SplitMix64(0xC0FF_EE00_1357_9BDF);
    for _ in 0..64 {
        let mut key = [0u8; 32];
        let mut nonce = [0u8; 12];
        rng.fill(&mut key);
        rng.fill(&mut nonce);
        let counter = (rng.next_u64() as u32) & 0x00FF_FFFF;

        let got = chacha20_block(&chacha20_state(&key, counter, &nonce));
        let want = rustcrypto_keystream(&key, counter, &nonce, 1);
        assert_eq!(&got[..], &want[..], "single-block mismatch vs RustCrypto at counter={counter}");
    }
}

/// Sanity-check the oracle itself is the IETF RFC 8439 variant (not the 64-bit
/// legacy nonce cipher) by reproducing the §2.3.2 KAT through RustCrypto. If this
/// fails, the oracle is misconfigured and the parity tests above prove nothing.
#[test]
fn rustcrypto_oracle_is_rfc8439_ietf() {
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = i as u8;
    }
    let nonce: [u8; 12] = [0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00];
    let block = rustcrypto_keystream(&key, 1, &nonce, 1);

    let expected: [u8; 64] = [
        0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4, 0xc7, 0xd1,
        0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46,
        0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16,
        0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
    ];
    assert_eq!(&block[..], &expected[..], "RustCrypto oracle must reproduce RFC 8439 §2.3.2");
}
