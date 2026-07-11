//! Crypto ARX primitives for the `ndarray::simd` polyfill — the hardware
//! acceleration layer the Ada `encryption` crate (Argon2id / XChaCha20-Poly1305
//! / Ed25519 / SHA-384) draws its hot-path keystream from.
//!
//! # Why these live in `ndarray::simd`
//!
//! The workspace invariant (`AdaWorldAPI/lance-graph` `simd-savant`): **all SIMD
//! comes from `ndarray::simd` via the polyfill — `simd.rs` + `simd_ops.rs` >
//! `simd_{arch}.rs`**. The crypto keystream is a SIMD hot path like any other, so
//! its accelerated form belongs here (server AVX-512 / browser wasm128), not
//! re-hidden inside a per-consumer copy. The consumer-facing crypto surface
//! (`ogar-encryption` → the `encryption` crate) composes AEAD/KDF framing on top;
//! this module supplies the vectorizable core.
//!
//! # Why a SIMD ChaCha20 is *safe* to hand-vectorize (unlike, say, AES)
//!
//! ChaCha20 is an **ARX** cipher: every operation is a 32-bit `wrapping_add`,
//! `xor`, or fixed-distance `rotate_left`. There are **no secret-dependent
//! branches and no secret-dependent memory indices** (no S-boxes, no T-tables),
//! so the block function is **constant-time by construction** — a property a
//! straight-line SIMD implementation preserves automatically. That is exactly
//! why RFC 8439 chose ChaCha20 for constant-time software, and why an
//! `ndarray::simd` vectorization does not introduce a timing side channel the
//! scalar path lacked. (AES in software, by contrast, is NOT safe to hand-roll
//! this way — its table lookups are secret-indexed; that is deliberately out of
//! scope here, and browsers already expose hardware AES via WebCrypto.)
//!
//! # Backend ladder (W1a consumer contract)
//!
//! | Backend          | ChaCha20 block strategy                                   |
//! |------------------|----------------------------------------------------------|
//! | scalar (this rev)| the RFC 8439 reference double-round — the CORRECTNESS anchor |
//! | AVX-512 (server) | 4 states in parallel across the 4 columns/diagonals (next) |
//! | wasm128 (browser)| v128 `u32x4` lanes, one state, SIMD quarter-round (next)  |
//! | NEON (edge)      | `uint32x4_t` lanes (next)                                 |
//!
//! The scalar reference lands FIRST with the RFC 8439 §2.3.2 Known-Answer-Test:
//! every future vectorized backend is a drop-in that MUST reproduce this exact
//! keystream (parity test), so correctness is pinned before performance. This is
//! the "reference + KAT, then vectorize with parity" discipline the W1a contract
//! mandates for any new `ndarray::simd` primitive.

/// The ChaCha20 constants — the ASCII of `"expand 32-byte k"` as four
/// little-endian `u32` words (RFC 8439 §2.3).
const CHACHA20_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// One ChaCha20 quarter-round on four working words (RFC 8439 §2.1). Pure ARX:
/// `wrapping_add` / `xor` / `rotate_left` — no branch, no memory index, so it is
/// constant-time regardless of the (secret) word values.
#[inline(always)]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(7);
}

/// Compute one 64-byte ChaCha20 keystream block from a fully-populated 16-word
/// input `state` (RFC 8439 §2.3.1): 20 rounds (10 column + diagonal
/// double-rounds), then add the original input state, then serialize each word
/// little-endian.
///
/// `state` is the caller-assembled block state — words `0..4` the constants,
/// `4..12` the 256-bit key, word `12` the block counter, `13..16` the 96-bit
/// nonce. Building that state (and the XChaCha `HChaCha20` nonce extension) is
/// the caller's job; this is the pure, vectorizable block core.
///
/// **Constant-time:** straight-line ARX, no data-dependent control flow.
/// This scalar form is the reference every SIMD backend is parity-checked
/// against (see the module KAT).
#[must_use]
pub fn chacha20_block(state: &[u32; 16]) -> [u8; 64] {
    let mut w = *state;
    // 10 double-rounds = 20 rounds.
    for _ in 0..10 {
        // Column round.
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        // Diagonal round.
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for (i, word) in w.iter().enumerate() {
        let sum = word.wrapping_add(state[i]);
        out[i * 4..i * 4 + 4].copy_from_slice(&sum.to_le_bytes());
    }
    out
}

/// Assemble a ChaCha20 block state from a 256-bit `key`, a 32-bit block
/// `counter`, and a 96-bit `nonce` (RFC 8439 §2.3). Little-endian word packing.
/// Convenience for callers and for the KAT; the vectorized backends operate on
/// the assembled `[u32; 16]` from [`chacha20_block`].
#[must_use]
pub fn chacha20_state(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u32; 16] {
    let mut s = [0u32; 16];
    s[0..4].copy_from_slice(&CHACHA20_CONSTANTS);
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }
    s[12] = counter;
    for i in 0..3 {
        s[13 + i] = u32::from_le_bytes([nonce[i * 4], nonce[i * 4 + 1], nonce[i * 4 + 2], nonce[i * 4 + 3]]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 8439 §2.3.2 Known-Answer-Test: the canonical ChaCha20 block. Key =
    /// `00,01,…,1f`, block counter = 1, nonce = `00 00 00 09 00 00 00 4a 00 00
    /// 00 00`. This pins the scalar reference; every SIMD backend added later
    /// MUST reproduce this exact 64-byte keystream (the parity gate).
    #[test]
    fn chacha20_block_rfc8439_kat() {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let nonce: [u8; 12] = [0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00];
        let state = chacha20_state(&key, 1, &nonce);
        let block = chacha20_block(&state);

        // RFC 8439 §2.3.2 serialized keystream (64 bytes).
        let expected: [u8; 64] = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4, 0xc7, 0xd1,
            0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46,
            0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16,
            0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];
        assert_eq!(block, expected, "ChaCha20 block must match RFC 8439 §2.3.2");
    }

    /// The quarter-round test vector (RFC 8439 §2.1.1) — the ARX core in
    /// isolation, so a backend can be debugged at the round level.
    #[test]
    fn quarter_round_rfc8439_vector() {
        // §2.1.1: inputs a,b,c,d and expected outputs, placed in a 16-word state
        // at indices 0..4 so `quarter_round` operates on them.
        let mut s = [0u32; 16];
        s[0] = 0x1111_1111;
        s[1] = 0x0102_0304;
        s[2] = 0x9b8d_6f43;
        s[3] = 0x0123_4567;
        quarter_round(&mut s, 0, 1, 2, 3);
        assert_eq!(
            [s[0], s[1], s[2], s[3]],
            [0xea2a_92f4, 0xcb1c_f8ce, 0x4581_472e, 0x5881_c4bb],
            "quarter-round must match RFC 8439 §2.1.1"
        );
    }
}
