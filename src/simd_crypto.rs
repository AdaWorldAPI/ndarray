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
//! | scalar           | the RFC 8439 reference double-round — the CORRECTNESS anchor |
//! | AVX-512 (server) | 16 blocks in parallel, word-sliced `__m512i` lanes (DONE) |
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

/// Fill `out` with successive ChaCha20 keystream blocks: block `b` uses block
/// counter `counter.wrapping_add(b)`, key and nonce fixed (RFC 8439 CTR mode).
///
/// This is the accelerated entry point the `encryption` AEAD hot path draws its
/// keystream from. It runs the **AVX-512 backend in 16-block strides** when
/// `avx512f` is detected at runtime, and the scalar [`chacha20_block`] reference
/// for the ragged tail and on every non-x86 / non-AVX-512 target. Every backend
/// is a drop-in for the scalar reference — see the module KAT and the
/// `chacha20_keystream_avx512_parity` test that pins byte-for-byte equality.
///
/// **Constant-time:** all backends are straight-line ARX (add / xor / rotate),
/// no secret-dependent control flow or memory indexing. Backend equivalence is
/// pinned by `chacha20_keystream_dispatch_parity_scalar` (byte-for-byte vs the
/// scalar reference across two AVX-512 strides + a scalar tail).
pub fn chacha20_keystream(key: &[u8; 32], counter: u32, nonce: &[u8; 12], out: &mut [[u8; 64]]) {
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            while i + 16 <= out.len() {
                let state = chacha20_state(key, counter.wrapping_add(i as u32), nonce);
                // SAFETY: this arm is only reached after `is_x86_feature_detected!("avx512f")`
                // returned true, so every AVX-512F intrinsic in `chacha20_block16_avx512`
                // is supported on this CPU. The function reads only its `&state` argument
                // and returns an owned array — no raw pointers escape, no aliasing.
                let blocks = unsafe { chacha20_block16_avx512(&state) };
                out[i..i + 16].copy_from_slice(&blocks);
                i += 16;
            }
        }
    }

    // Scalar reference: the ragged tail, and the whole slice on non-AVX-512 targets.
    while i < out.len() {
        let state = chacha20_state(key, counter.wrapping_add(i as u32), nonce);
        out[i] = chacha20_block(&state);
        i += 1;
    }
}

/// AVX-512 backend: compute 16 consecutive ChaCha20 keystream blocks in parallel
/// (block `l` uses `state[12].wrapping_add(l)` as its counter, all other words
/// shared). Word-sliced ("vertical") layout — lane `l` of working vector `w`
/// holds word `w` of block `l` — so every round is pure SIMD add/xor/rotate with
/// no cross-lane shuffles; the only transpose is the final little-endian
/// serialization.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (guarded by
/// `is_x86_feature_detected!("avx512f")` at the single call site in
/// [`chacha20_keystream`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn chacha20_block16_avx512(state: &[u32; 16]) -> [[u8; 64]; 16] {
    use core::arch::x86_64::*;

    // One AVX-512 vertical quarter-round: same word indices as the scalar
    // `quarter_round`, applied across all 16 blocks at once. Pure ARX.
    macro_rules! qr512 {
        ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr) => {{
            $v[$a] = _mm512_add_epi32($v[$a], $v[$b]);
            $v[$d] = _mm512_rol_epi32::<16>(_mm512_xor_si512($v[$d], $v[$a]));
            $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
            $v[$b] = _mm512_rol_epi32::<12>(_mm512_xor_si512($v[$b], $v[$c]));
            $v[$a] = _mm512_add_epi32($v[$a], $v[$b]);
            $v[$d] = _mm512_rol_epi32::<8>(_mm512_xor_si512($v[$d], $v[$a]));
            $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
            $v[$b] = _mm512_rol_epi32::<7>(_mm512_xor_si512($v[$b], $v[$c]));
        }};
    }

    // Build the 16 initial working vectors. Every word is broadcast identically
    // across the 16 lanes EXCEPT the counter (word 12), which gets lane-index
    // 0..=15 added (u32 wrapping) — the 16 consecutive block counters.
    let lane_index = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mut orig = [_mm512_setzero_si512(); 16];
    for (w, o) in orig.iter_mut().enumerate() {
        *o = _mm512_set1_epi32(state[w] as i32);
    }
    orig[12] = _mm512_add_epi32(_mm512_set1_epi32(state[12] as i32), lane_index);

    let mut v = orig;
    // 10 double-rounds = 20 rounds, identical schedule to the scalar reference.
    for _ in 0..10 {
        qr512!(v, 0, 4, 8, 12);
        qr512!(v, 1, 5, 9, 13);
        qr512!(v, 2, 6, 10, 14);
        qr512!(v, 3, 7, 11, 15);
        qr512!(v, 0, 5, 10, 15);
        qr512!(v, 1, 6, 11, 12);
        qr512!(v, 2, 7, 8, 13);
        qr512!(v, 3, 4, 9, 14);
    }

    // Add the original input state back (RFC 8439 §2.3.1), then de-vectorize:
    // `words[w][l]` = final word `w` of block `l`.
    let mut words = [[0u32; 16]; 16];
    for (w, dst) in words.iter_mut().enumerate() {
        let summed = _mm512_add_epi32(v[w], orig[w]);
        let mut tmp = [0i32; 16];
        // SAFETY: `tmp` is a 16-element (64-byte) i32 array; `_mm512_storeu_si512`
        // writes exactly one 512-bit (64-byte) vector, unaligned, in bounds.
        _mm512_storeu_si512(tmp.as_mut_ptr().cast(), summed);
        for (l, slot) in dst.iter_mut().enumerate() {
            *slot = tmp[l] as u32;
        }
    }

    // Serialize each block little-endian: block `l`, word `w` → bytes 4w..4w+4.
    let mut out = [[0u8; 64]; 16];
    for (l, block) in out.iter_mut().enumerate() {
        for w in 0..16 {
            block[w * 4..w * 4 + 4].copy_from_slice(&words[w][l].to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RFC 8439 §2.3.2 canonical keystream block (key = `00,01,…,1f`, block
    /// counter = 1, nonce = `00 00 00 09 00 00 00 4a 00 00 00 00`). The scalar
    /// reference and every SIMD backend MUST reproduce this exact 64-byte block.
    const RFC8439_KAT_BLOCK: [u8; 64] = [
        0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4, 0xc7, 0xd1,
        0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46,
        0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16,
        0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
    ];

    /// The RFC 8439 §2.3.2 key material (key `00..1f`, counter 1, the §2.3.2 nonce).
    fn rfc8439_kat_inputs() -> ([u8; 32], u32, [u8; 12]) {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let nonce: [u8; 12] = [0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00];
        (key, 1, nonce)
    }

    /// RFC 8439 §2.3.2 Known-Answer-Test on the scalar reference block. This pins
    /// the scalar reference; every SIMD backend added later MUST reproduce this
    /// exact 64-byte keystream (the parity gate).
    #[test]
    fn chacha20_block_rfc8439_kat() {
        let (key, counter, nonce) = rfc8439_kat_inputs();
        let state = chacha20_state(&key, counter, &nonce);
        let block = chacha20_block(&state);
        assert_eq!(block, RFC8439_KAT_BLOCK, "ChaCha20 block must match RFC 8439 §2.3.2");
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

    /// The mandatory SIMD-crypto correctness gate: every block produced by the
    /// `chacha20_keystream` dispatcher (AVX-512 backend when `avx512f` is present,
    /// scalar tail always) MUST be byte-identical to the scalar
    /// [`chacha20_block`] reference for the same key/counter/nonce. `n = 40`
    /// spans two full 16-block AVX-512 strides plus an 8-block scalar tail, so
    /// both the vectorized body and the ragged-tail path are exercised. If this
    /// test is red, no SIMD crypto backend may be committed.
    #[test]
    fn chacha20_keystream_dispatch_parity_scalar() {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        let nonce: [u8; 12] = [0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        let base = 0x0102_0304u32;
        let n = 40;

        let mut ks = vec![[0u8; 64]; n];
        chacha20_keystream(&key, base, &nonce, &mut ks);

        for (b, got) in ks.iter().enumerate() {
            let state = chacha20_state(&key, base.wrapping_add(b as u32), &nonce);
            let want = chacha20_block(&state);
            assert_eq!(*got, want, "keystream block {b} must equal the scalar reference");
        }
    }

    /// The dispatcher must reproduce the RFC 8439 §2.3.2 KAT at block 0. When the
    /// host CPU has AVX-512F this drives the RFC vector THROUGH the AVX-512
    /// backend (16 blocks per stride, block 0 uses counter 1), directly pinning
    /// the vectorized path to the published answer — not just to the scalar path.
    #[test]
    fn chacha20_keystream_dispatch_matches_rfc8439_kat() {
        let (key, counter, nonce) = rfc8439_kat_inputs();
        // 16 blocks forces the AVX-512 stride when available; block 0 == the KAT.
        let mut ks = vec![[0u8; 64]; 16];
        chacha20_keystream(&key, counter, &nonce, &mut ks);
        assert_eq!(ks[0], RFC8439_KAT_BLOCK, "dispatcher block 0 must match RFC 8439 §2.3.2");
    }
}
