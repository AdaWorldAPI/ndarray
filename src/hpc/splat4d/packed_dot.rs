//! `splat4d::packed_dot` — G2 INT4×32 packed dot product (A4 worker, PR-X4).
//!
//! Three hardware-dispatch backends for thinking-style (32-dim INT4) and
//! qualia (16-dim INT4) cell signatures:
//!
//! | Backend | Primitive | Note |
//! |---|---|---|
//! | AVX-512 VNNI | `vpdpbusd` | i8×u8 → i32, 64 ops/inst → 2 inst for 32-dim INT4 |
//! | ARM NEON | `sdot` (UDOT) | i8.4·u8.4 → i32 → 8 inst for 32-dim |
//! | Scalar | 32-way unrolled | AMX BF16 tile is INT8 not INT4; fallback path |
//!
//! Precision class: **EXACT** (integer dot).
//!
//! # Bundle consumed: B-Pack-Dot
//!
//! `pack_int4x32 ∘ dot_i4x32_to_i32 ∘ dequant_f32`. Do NOT reach past this
//! bundle into raw intrinsics.
//!
//! # References
//!
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G2 — INT4×N packed dot"
//! - `pr-arithmetic-inventory.md` § L4-L8 G2

//// TODO(A4): implement three backends + parity test.
//// cross-backend parity test: 10K random (a, b) pairs, all three backends
//// must agree exactly (integer dot is exact).
//// Ref: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G2"
