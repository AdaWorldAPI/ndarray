//! WebAssembly SIMD128 backend — real `core::arch::wasm32` intrinsics.
//!
//! Mirrors the API of `simd_avx512` / `simd_avx2` / `simd_neon::aarch64_simd`
//! so consumer code reading `use crate::simd::F32x16` compiles and runs
//! uniformly on `wasm32` once the `simd128` target feature is enabled.
//!
//! # Dispatch
//!
//! `src/simd.rs` re-exports the native v128-backed hot-path types
//! (`F32x16` / `F64x8` + masks, `I8x16`) from `wasm32_simd` for
//! `target_arch = "wasm32"` **and** `target_feature = "simd128"`, and pulls
//! the long-tail integer / 256-bit-shaped types from the scalar fallback —
//! exactly the split `simd_neon` uses on aarch64 (native float kernels,
//! scalar for the rest). Without `simd128` (or on any other non-x86 / non-arm
//! target) the full scalar fallback is used and this module compiles empty.
//!
//! Build with simd128 enabled:
//! ```text
//! RUSTFLAGS='-C target-feature=+simd128' cargo build --target wasm32-unknown-unknown
//! # or: cargo build --target wasm32-unknown-unknown --config .cargo/config-wasm.toml
//! ```
//!
//! # Register model
//!
//! WASM SIMD128 has one 128-bit register type, `v128`, reinterpreted per op:
//!   * `f32x4` — 4 × f32   * `f64x2` — 2 × f64
//!   * `i8x16` — 16 × i8   * `i16x8` — 8 × i16
//!   * `i32x4` — 4 × i32   * `i64x2` — 2 × i64
//!
//! The 512-bit-shaped public types compose from four v128 registers, the
//! same 4-register pattern `simd_neon::aarch64_simd` uses on NEON:
//!   * `F32x16` = `[v128; 4]` (f32x4 interpretation)
//!   * `F64x8`  = `[v128; 4]` (f64x2 interpretation)
//!
//! # Semantic notes (cross-backend parity)
//!
//! * **FMA.** Base simd128 has no fused multiply-add. `mul_add` uses
//!   `mul`-then-`add` (two roundings) unless the `relaxed-simd` target
//!   feature is enabled, in which case `f32x4_relaxed_madd` is used. The
//!   unfused path differs from the fused scalar/NEON/AVX `mul_add` by ~1 ULP
//!   in the common case, but can diverge by more under catastrophic
//!   cancellation. Tests use a tolerance, matching the existing
//!   cross-backend tests.
//! * **reduce_sum.** Uses a balanced tree reduction (pairwise v128 adds)
//!   rather than the scalar fallback's sequential left-fold, so the result
//!   can differ in the last ULP — the same reduction-order tolerance the
//!   AVX/NEON backends already carry.
//! * **round().** Uses `f32x4_nearest` / `f64x2_nearest` (round-half-to-even),
//!   identical to NEON's `vrndnq_*`. The pure-scalar fallback uses
//!   `f32::round` (round-half-away-from-zero); the two differ only on exact
//!   `.5` ties, which the parity corpus avoids — the same situation that
//!   already exists between the scalar and NEON backends.
//! * **min/max NaN.** `f32x4_min` / `f32x4_max` follow IEEE minimum/maximum
//!   (NaN-propagating). The scalar fallback's `f32::min` returns the non-NaN
//!   operand. `simd_exp_f32` in `simd.rs` saves/restores NaN lanes around the
//!   clamp precisely because every SIMD backend treats NaN in min/max
//!   differently, so this divergence is already absorbed upstream.
//!
//! Reference for the intrinsic set: macerator's wasm32 backend
//! (wingertge/macerator) and the WebAssembly fixed-width SIMD proposal.

// Everything below is gated on BOTH the wasm32 architecture and the simd128
// target feature. When simd128 is off, the module is empty and the scalar
// fallback in `simd_scalar.rs` covers `wasm32` (see `simd.rs` dispatch).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32_simd {
    use core::arch::wasm32::*;
    use core::fmt;
    use core::ops::{Add, AddAssign, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    // Long-tail integer types come from the scalar fallback — they are not on
    // the perf-critical f32/byte path this module accelerates, and reusing the
    // scalar types keeps the `to_bits` / `from_bits` / `cast_i32` return types
    // identical to every other backend (same choice `simd_neon` makes).
    // `U32x16` is the exception: it carries the ARX vocabulary (Add/BitXor/
    // rotate_left) the ChaCha20 lane needs, so it is native here (`[U32x4; 4]`,
    // NEON-style) rather than the scalar fallback — see below.
    pub use crate::simd::scalar::{I32x16, U64x8};

    // ════════════════════════════════════════════════════════════════════
    // F32x16 — 16 × f32 backed by 4 × v128 (f32x4 interpretation)
    // ════════════════════════════════════════════════════════════════════

    /// 16×f32 backed by 4× WASM `v128` registers (f32x4 interpretation).
    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F32x16(pub [v128; 4]);

    impl F32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            let s = f32x4_splat(v);
            Self([s, s, s, s])
        }

        #[inline(always)]
        pub fn from_slice(s: &[f32]) -> Self {
            assert!(s.len() >= 16);
            // SAFETY: length checked >= 16; wasm v128 loads are alignment-free
            // and read 16 bytes (4 × f32) per load at offsets 0,4,8,12.
            unsafe {
                let p = s.as_ptr();
                Self([
                    v128_load(p as *const v128),
                    v128_load(p.add(4) as *const v128),
                    v128_load(p.add(8) as *const v128),
                    v128_load(p.add(12) as *const v128),
                ])
            }
        }

        #[inline(always)]
        pub fn from_array(a: [f32; 16]) -> Self {
            Self::from_slice(&a)
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut out = [0.0f32; 16];
            self.copy_to_slice(&mut out);
            out
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f32]) {
            assert!(s.len() >= 16);
            // SAFETY: length checked >= 16; each store writes 16 bytes.
            unsafe {
                let p = s.as_mut_ptr();
                v128_store(p as *mut v128, self.0[0]);
                v128_store(p.add(4) as *mut v128, self.0[1]);
                v128_store(p.add(8) as *mut v128, self.0[2]);
                v128_store(p.add(12) as *mut v128, self.0[3]);
            }
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f32 {
            let s = f32x4_add(f32x4_add(self.0[0], self.0[1]), f32x4_add(self.0[2], self.0[3]));
            f32x4_extract_lane::<0>(s)
                + f32x4_extract_lane::<1>(s)
                + f32x4_extract_lane::<2>(s)
                + f32x4_extract_lane::<3>(s)
        }

        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.to_array()
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.to_array()
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self([f32x4_abs(self.0[0]), f32x4_abs(self.0[1]), f32x4_abs(self.0[2]), f32x4_abs(self.0[3])])
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self([f32x4_sqrt(self.0[0]), f32x4_sqrt(self.0[1]), f32x4_sqrt(self.0[2]), f32x4_sqrt(self.0[3])])
        }

        /// Round to nearest, ties to even (`f32x4_nearest`) — matches NEON `vrndnq_f32`.
        #[inline(always)]
        pub fn round(self) -> Self {
            Self([
                f32x4_nearest(self.0[0]),
                f32x4_nearest(self.0[1]),
                f32x4_nearest(self.0[2]),
                f32x4_nearest(self.0[3]),
            ])
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            Self([f32x4_floor(self.0[0]), f32x4_floor(self.0[1]), f32x4_floor(self.0[2]), f32x4_floor(self.0[3])])
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            #[inline(always)]
            fn madd(a: v128, b: v128, c: v128) -> v128 {
                #[cfg(target_feature = "relaxed-simd")]
                {
                    f32x4_relaxed_madd(a, b, c)
                }
                #[cfg(not(target_feature = "relaxed-simd"))]
                {
                    f32x4_add(f32x4_mul(a, b), c)
                }
            }
            Self([
                madd(self.0[0], b.0[0], c.0[0]),
                madd(self.0[1], b.0[1], c.0[1]),
                madd(self.0[2], b.0[2], c.0[2]),
                madd(self.0[3], b.0[3], c.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            Self([
                f32x4_min(self.0[0], other.0[0]),
                f32x4_min(self.0[1], other.0[1]),
                f32x4_min(self.0[2], other.0[2]),
                f32x4_min(self.0[3], other.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            Self([
                f32x4_max(self.0[0], other.0[0]),
                f32x4_max(self.0[1], other.0[1]),
                f32x4_max(self.0[2], other.0[2]),
                f32x4_max(self.0[3], other.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        // Comparisons return a 16-bit mask (bit i set where the predicate holds).
        // Each f32x4 compare yields all-ones lanes; `i32x4_bitmask` extracts the
        // 4 sign bits, which are packed into the 16-bit result 4 lanes at a time.
        #[inline(always)]
        fn cmp_mask(a: [v128; 4], b: [v128; 4], op: fn(v128, v128) -> v128) -> F32Mask16 {
            let mut bits: u16 = 0;
            for k in 0..4 {
                bits |= (i32x4_bitmask(op(a[k], b[k])) as u16) << (4 * k);
            }
            F32Mask16(bits)
        }

        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_lt)
        }
        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_le)
        }
        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_gt)
        }
        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_ge)
        }
        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_eq)
        }
        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F32Mask16 {
            Self::cmp_mask(self.0, other.0, f32x4_ne)
        }

        #[inline(always)]
        pub fn to_bits(self) -> U32x16 {
            let a = self.to_array();
            let mut o = [0u32; 16];
            for i in 0..16 {
                o[i] = a[i].to_bits();
            }
            U32x16::from_array(o)
        }
        #[inline(always)]
        pub fn from_bits(bits: U32x16) -> Self {
            let b = bits.to_array();
            let mut o = [0.0f32; 16];
            for i in 0..16 {
                o[i] = f32::from_bits(b[i]);
            }
            Self::from_array(o)
        }
        #[inline(always)]
        pub fn cast_i32(self) -> I32x16 {
            let a = self.to_array();
            let mut o = [0i32; 16];
            for i in 0..16 {
                o[i] = a[i] as i32;
            }
            I32x16(o)
        }
    }

    impl Add for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, r: Self) -> Self {
            Self([
                f32x4_add(self.0[0], r.0[0]),
                f32x4_add(self.0[1], r.0[1]),
                f32x4_add(self.0[2], r.0[2]),
                f32x4_add(self.0[3], r.0[3]),
            ])
        }
    }
    impl Sub for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, r: Self) -> Self {
            Self([
                f32x4_sub(self.0[0], r.0[0]),
                f32x4_sub(self.0[1], r.0[1]),
                f32x4_sub(self.0[2], r.0[2]),
                f32x4_sub(self.0[3], r.0[3]),
            ])
        }
    }
    impl Mul for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, r: Self) -> Self {
            Self([
                f32x4_mul(self.0[0], r.0[0]),
                f32x4_mul(self.0[1], r.0[1]),
                f32x4_mul(self.0[2], r.0[2]),
                f32x4_mul(self.0[3], r.0[3]),
            ])
        }
    }
    impl Div for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, r: Self) -> Self {
            Self([
                f32x4_div(self.0[0], r.0[0]),
                f32x4_div(self.0[1], r.0[1]),
                f32x4_div(self.0[2], r.0[2]),
                f32x4_div(self.0[3], r.0[3]),
            ])
        }
    }
    impl AddAssign for F32x16 {
        #[inline(always)]
        fn add_assign(&mut self, r: Self) {
            *self = *self + r;
        }
    }
    impl SubAssign for F32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, r: Self) {
            *self = *self - r;
        }
    }
    impl MulAssign for F32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, r: Self) {
            *self = *self * r;
        }
    }
    impl DivAssign for F32x16 {
        #[inline(always)]
        fn div_assign(&mut self, r: Self) {
            *self = *self / r;
        }
    }
    impl Neg for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self([f32x4_neg(self.0[0]), f32x4_neg(self.0[1]), f32x4_neg(self.0[2]), f32x4_neg(self.0[3])])
        }
    }
    impl fmt::Debug for F32x16 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F32x16({:?})", self.to_array())
        }
    }
    impl PartialEq for F32x16 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }
    impl Default for F32x16 {
        fn default() -> Self {
            Self::splat(0.0)
        }
    }

    /// 16-lane comparison mask for `F32x16` (one bit per lane).
    #[derive(Copy, Clone, Debug)]
    pub struct F32Mask16(pub u16);
    impl F32Mask16 {
        #[inline(always)]
        pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
            let t = true_val.to_array();
            let f = false_val.to_array();
            let mut o = [0.0f32; 16];
            for i in 0..16 {
                o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] };
            }
            F32x16::from_array(o)
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // F64x8 — 8 × f64 backed by 4 × v128 (f64x2 interpretation)
    // ════════════════════════════════════════════════════════════════════

    /// 8×f64 backed by 4× WASM `v128` registers (f64x2 interpretation).
    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F64x8(pub [v128; 4]);

    impl F64x8 {
        pub const LANES: usize = 8;

        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            let s = f64x2_splat(v);
            Self([s, s, s, s])
        }

        #[inline(always)]
        pub fn from_slice(s: &[f64]) -> Self {
            assert!(s.len() >= 8);
            // SAFETY: length checked >= 8; each load reads 2 × f64 at offsets 0,2,4,6.
            unsafe {
                let p = s.as_ptr();
                Self([
                    v128_load(p as *const v128),
                    v128_load(p.add(2) as *const v128),
                    v128_load(p.add(4) as *const v128),
                    v128_load(p.add(6) as *const v128),
                ])
            }
        }

        #[inline(always)]
        pub fn from_array(a: [f64; 8]) -> Self {
            Self::from_slice(&a)
        }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut out = [0.0f64; 8];
            self.copy_to_slice(&mut out);
            out
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            // SAFETY: length checked >= 8; each store writes 2 × f64.
            unsafe {
                let p = s.as_mut_ptr();
                v128_store(p as *mut v128, self.0[0]);
                v128_store(p.add(2) as *mut v128, self.0[1]);
                v128_store(p.add(4) as *mut v128, self.0[2]);
                v128_store(p.add(6) as *mut v128, self.0[3]);
            }
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f64 {
            let s = f64x2_add(f64x2_add(self.0[0], self.0[1]), f64x2_add(self.0[2], self.0[3]));
            f64x2_extract_lane::<0>(s) + f64x2_extract_lane::<1>(s)
        }

        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.to_array()
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.to_array()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self([f64x2_abs(self.0[0]), f64x2_abs(self.0[1]), f64x2_abs(self.0[2]), f64x2_abs(self.0[3])])
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self([f64x2_sqrt(self.0[0]), f64x2_sqrt(self.0[1]), f64x2_sqrt(self.0[2]), f64x2_sqrt(self.0[3])])
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            Self([
                f64x2_nearest(self.0[0]),
                f64x2_nearest(self.0[1]),
                f64x2_nearest(self.0[2]),
                f64x2_nearest(self.0[3]),
            ])
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            Self([f64x2_floor(self.0[0]), f64x2_floor(self.0[1]), f64x2_floor(self.0[2]), f64x2_floor(self.0[3])])
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            #[inline(always)]
            fn madd(a: v128, b: v128, c: v128) -> v128 {
                #[cfg(target_feature = "relaxed-simd")]
                {
                    f64x2_relaxed_madd(a, b, c)
                }
                #[cfg(not(target_feature = "relaxed-simd"))]
                {
                    f64x2_add(f64x2_mul(a, b), c)
                }
            }
            Self([
                madd(self.0[0], b.0[0], c.0[0]),
                madd(self.0[1], b.0[1], c.0[1]),
                madd(self.0[2], b.0[2], c.0[2]),
                madd(self.0[3], b.0[3], c.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            Self([
                f64x2_min(self.0[0], other.0[0]),
                f64x2_min(self.0[1], other.0[1]),
                f64x2_min(self.0[2], other.0[2]),
                f64x2_min(self.0[3], other.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            Self([
                f64x2_max(self.0[0], other.0[0]),
                f64x2_max(self.0[1], other.0[1]),
                f64x2_max(self.0[2], other.0[2]),
                f64x2_max(self.0[3], other.0[3]),
            ])
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        // Each f64x2 compare yields 2 lanes; `i64x2_bitmask` extracts the 2 sign
        // bits, packed into the 8-bit result 2 lanes at a time.
        #[inline(always)]
        fn cmp_mask(a: [v128; 4], b: [v128; 4], op: fn(v128, v128) -> v128) -> F64Mask8 {
            let mut bits: u8 = 0;
            for k in 0..4 {
                bits |= (i64x2_bitmask(op(a[k], b[k])) as u8) << (2 * k);
            }
            F64Mask8(bits)
        }

        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_lt)
        }
        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_le)
        }
        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_gt)
        }
        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_ge)
        }
        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_eq)
        }
        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F64Mask8 {
            Self::cmp_mask(self.0, other.0, f64x2_ne)
        }

        #[inline(always)]
        pub fn to_bits(self) -> U64x8 {
            let a = self.to_array();
            let mut o = [0u64; 8];
            for i in 0..8 {
                o[i] = a[i].to_bits();
            }
            U64x8(o)
        }
        #[inline(always)]
        pub fn from_bits(bits: U64x8) -> Self {
            let mut o = [0.0f64; 8];
            for i in 0..8 {
                o[i] = f64::from_bits(bits.0[i]);
            }
            Self::from_array(o)
        }
    }

    impl Add for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, r: Self) -> Self {
            Self([
                f64x2_add(self.0[0], r.0[0]),
                f64x2_add(self.0[1], r.0[1]),
                f64x2_add(self.0[2], r.0[2]),
                f64x2_add(self.0[3], r.0[3]),
            ])
        }
    }
    impl Sub for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, r: Self) -> Self {
            Self([
                f64x2_sub(self.0[0], r.0[0]),
                f64x2_sub(self.0[1], r.0[1]),
                f64x2_sub(self.0[2], r.0[2]),
                f64x2_sub(self.0[3], r.0[3]),
            ])
        }
    }
    impl Mul for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, r: Self) -> Self {
            Self([
                f64x2_mul(self.0[0], r.0[0]),
                f64x2_mul(self.0[1], r.0[1]),
                f64x2_mul(self.0[2], r.0[2]),
                f64x2_mul(self.0[3], r.0[3]),
            ])
        }
    }
    impl Div for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, r: Self) -> Self {
            Self([
                f64x2_div(self.0[0], r.0[0]),
                f64x2_div(self.0[1], r.0[1]),
                f64x2_div(self.0[2], r.0[2]),
                f64x2_div(self.0[3], r.0[3]),
            ])
        }
    }
    impl AddAssign for F64x8 {
        #[inline(always)]
        fn add_assign(&mut self, r: Self) {
            *self = *self + r;
        }
    }
    impl SubAssign for F64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, r: Self) {
            *self = *self - r;
        }
    }
    impl MulAssign for F64x8 {
        #[inline(always)]
        fn mul_assign(&mut self, r: Self) {
            *self = *self * r;
        }
    }
    impl DivAssign for F64x8 {
        #[inline(always)]
        fn div_assign(&mut self, r: Self) {
            *self = *self / r;
        }
    }
    impl Neg for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self([f64x2_neg(self.0[0]), f64x2_neg(self.0[1]), f64x2_neg(self.0[2]), f64x2_neg(self.0[3])])
        }
    }
    impl fmt::Debug for F64x8 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F64x8({:?})", self.to_array())
        }
    }
    impl PartialEq for F64x8 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }
    impl Default for F64x8 {
        fn default() -> Self {
            Self::splat(0.0)
        }
    }

    /// 8-lane comparison mask for `F64x8` (one bit per lane).
    #[derive(Copy, Clone, Debug)]
    pub struct F64Mask8(pub u8);
    impl F64Mask8 {
        #[inline(always)]
        pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
            let t = true_val.to_array();
            let f = false_val.to_array();
            let mut o = [0.0f64; 8];
            for i in 0..8 {
                o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] };
            }
            F64x8::from_array(o)
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // I8x16 — 16 × i8 backed by one v128 (native byte lane)
    // ════════════════════════════════════════════════════════════════════

    /// 16×i8 backed by one WASM `v128` register.
    ///
    /// Exposes the union of the scalar and NEON `I8x16` method sets so consumer
    /// code is portable across every backend: arithmetic (`add`/`sub`),
    /// `min`/`max`, `cmp_gt`, plus the W1a primitives `from_i4_packed_u64`,
    /// `lane_i8`, and `saturating_abs`.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct I8x16(pub v128);

    impl I8x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn splat(v: i8) -> Self {
            Self(i8x16_splat(v))
        }

        #[inline(always)]
        pub fn zero() -> Self {
            Self(i8x16_splat(0))
        }

        #[inline(always)]
        pub fn from_slice(s: &[i8]) -> Self {
            assert!(s.len() >= 16);
            // SAFETY: length checked >= 16; one 16-byte load.
            Self(unsafe { v128_load(s.as_ptr() as *const v128) })
        }

        #[inline(always)]
        pub fn from_array(arr: [i8; 16]) -> Self {
            // SAFETY: a [i8; 16] is exactly 16 bytes; load reads all of it.
            Self(unsafe { v128_load(arr.as_ptr() as *const v128) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [i8; 16] {
            let mut arr = [0i8; 16];
            // SAFETY: store writes exactly 16 bytes into the 16-byte array.
            unsafe { v128_store(arr.as_mut_ptr() as *mut v128, self.0) };
            arr
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [i8]) {
            assert!(s.len() >= 16);
            // SAFETY: length checked >= 16; one 16-byte store.
            unsafe { v128_store(s.as_mut_ptr() as *mut v128, self.0) };
        }

        #[inline(always)]
        pub fn add(self, other: Self) -> Self {
            Self(i8x16_add(self.0, other.0))
        }
        #[inline(always)]
        pub fn sub(self, other: Self) -> Self {
            Self(i8x16_sub(self.0, other.0))
        }

        /// Lane-wise signed minimum.
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            // where self > other → other, else self
            Self(v128_bitselect(other.0, self.0, i8x16_gt(self.0, other.0)))
        }
        /// Lane-wise signed maximum.
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            // where self > other → self, else other
            Self(v128_bitselect(self.0, other.0, i8x16_gt(self.0, other.0)))
        }

        /// Compare-greater-than: returns a 16-bit mask. Bit i set where self[i] > other[i].
        #[inline(always)]
        pub fn cmp_gt(self, other: Self) -> u16 {
            i8x16_bitmask(i8x16_gt(self.0, other.0))
        }

        /// Unpack 16 signed i4 nibbles from a `u64` into 16 sign-extended `i8` lanes.
        ///
        /// `lane[i] = sign_extend_i4((packed >> (4*i)) & 0xf)`. Values `0x0..=0x7`
        /// map to `0..=7`; `0x8..=0xf` map to `-8..=-1`. Matches the scalar /
        /// AVX-512 `from_i4_packed_u64` byte-for-byte.
        #[inline(always)]
        pub fn from_i4_packed_u64(packed: u64) -> Self {
            let mut lanes = [0i8; 16];
            for i in 0..16 {
                let nibble = ((packed >> (4 * i)) & 0xf) as i8;
                lanes[i] = if nibble > 7 { nibble - 16 } else { nibble };
            }
            Self::from_array(lanes)
        }

        /// Extract lane `N` as an `i8`. `N` must be in `0..16`.
        #[inline(always)]
        pub fn lane_i8<const N: usize>(self) -> i8 {
            i8x16_extract_lane::<N>(self.0)
        }

        /// Lane-wise saturating absolute value. `saturating_abs(i8::MIN) == i8::MAX`.
        ///
        /// `i8x16_abs` leaves `i8::MIN` as `0x80` (the `-128` bit pattern); the
        /// unsigned clamp pulls only that lane down to `0x7f` (127), leaving every
        /// other lane (where `abs(x) < 0x80`) unchanged — the same VPABSB+VPMINUB
        /// mechanic the AVX-512 backend uses.
        #[inline(always)]
        pub fn saturating_abs(self) -> Self {
            let raw = i8x16_abs(self.0);
            let max = i8x16_splat(0x7f);
            // u8x16_gt(raw, 0x7f) is true only for the 0x80 lane → select 0x7f there.
            Self(v128_bitselect(max, raw, u8x16_gt(raw, max)))
        }
    }

    impl fmt::Debug for I8x16 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "I8x16({:?})", self.to_array())
        }
    }
    impl PartialEq for I8x16 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // U32x4 / U32x16 — u32 ARX lanes (ChaCha20 / BLAKE), NEON-style.
    //
    // `U32x4` is the dispatched native unit (one `v128`, u32x4 interpretation);
    // `U32x16` is the 16-wide polyfill `[U32x4; 4]`, fanning each op over the 4
    // sub-lanes — the same composition `simd_neon` uses (`[U32x4; 4]`, where
    // NEON's U32x4 wraps `uint32x4_t`). `U32x16`'s consumer API (`Add` /
    // `BitXor` / `rotate_left`) mirrors `simd_avx512::U32x16` exactly, so one
    // source (the ChaCha20 backend) compiles unchanged on every tier. The u32
    // ARX ops are all safe wasm intrinsics; only the v128 load/store are unsafe.
    // ════════════════════════════════════════════════════════════════════

    /// 4×u32 in one WASM `v128` (u32x4 interpretation) — the native ARX unit.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct U32x4(pub v128);

    impl U32x4 {
        pub const LANES: usize = 4;

        #[inline(always)]
        pub fn splat(v: u32) -> Self {
            Self(u32x4_splat(v))
        }

        #[inline(always)]
        pub fn from_array(a: [u32; 4]) -> Self {
            // SAFETY: a `[u32; 4]` is exactly 16 bytes; one unaligned v128 load.
            Self(unsafe { v128_load(a.as_ptr() as *const v128) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [u32; 4] {
            let mut a = [0u32; 4];
            // SAFETY: store writes exactly 16 bytes into the 16-byte array.
            unsafe { v128_store(a.as_mut_ptr() as *mut v128, self.0) };
            a
        }

        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(u32x4_add(self.0, o.0))
        }

        #[inline(always)]
        pub fn bitxor(self, o: Self) -> Self {
            Self(v128_xor(self.0, o.0))
        }

        /// Lane-wise left-rotate by `n` bits — the ARX rotate (matches
        /// `u32::rotate_left`). wasm has no rotate op, so this is the shift-or
        /// with the `n % 32 == 0` guard (`u32x4_shr` by 32 would over-shift);
        /// `u32x4_shl`/`u32x4_shr` already mask the count to `& 31`, so
        /// non-zero `n` is exact. Rotate amount is a public ARX constant.
        #[inline(always)]
        pub fn rotate_left(self, n: u32) -> Self {
            let n = n % 32;
            if n == 0 {
                return self;
            }
            Self(v128_or(u32x4_shl(self.0, n), u32x4_shr(self.0, 32 - n)))
        }
    }

    /// 16×u32 as `[U32x4; 4]` — the NEON-style 16-wide polyfill. Consumer API
    /// (`Add` / `BitXor` / `rotate_left`) matches `simd_avx512::U32x16`.
    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct U32x16(pub [U32x4; 4]);

    impl U32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn splat(v: u32) -> Self {
            Self([U32x4::splat(v); 4])
        }

        #[inline(always)]
        pub fn from_array(a: [u32; 16]) -> Self {
            Self([
                U32x4::from_array([a[0], a[1], a[2], a[3]]),
                U32x4::from_array([a[4], a[5], a[6], a[7]]),
                U32x4::from_array([a[8], a[9], a[10], a[11]]),
                U32x4::from_array([a[12], a[13], a[14], a[15]]),
            ])
        }

        #[inline(always)]
        pub fn to_array(self) -> [u32; 16] {
            let mut o = [0u32; 16];
            for i in 0..4 {
                o[i * 4..i * 4 + 4].copy_from_slice(&self.0[i].to_array());
            }
            o
        }

        /// Lane-wise left-rotate by `n` bits (ARX rotate), fanned over 4 lanes.
        #[inline(always)]
        pub fn rotate_left(self, n: u32) -> Self {
            Self([
                self.0[0].rotate_left(n),
                self.0[1].rotate_left(n),
                self.0[2].rotate_left(n),
                self.0[3].rotate_left(n),
            ])
        }
    }

    impl Add for U32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, r: Self) -> Self {
            Self([self.0[0].add(r.0[0]), self.0[1].add(r.0[1]), self.0[2].add(r.0[2]), self.0[3].add(r.0[3])])
        }
    }

    impl BitXor for U32x16 {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, r: Self) -> Self {
            Self([
                self.0[0].bitxor(r.0[0]),
                self.0[1].bitxor(r.0[1]),
                self.0[2].bitxor(r.0[2]),
                self.0[3].bitxor(r.0[3]),
            ])
        }
    }

    impl fmt::Debug for U32x16 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "U32x16({:?})", self.to_array())
        }
    }

    impl PartialEq for U32x16 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Lowercase aliases (consumer-API parity with the other backends)
    // ════════════════════════════════════════════════════════════════════

    #[allow(non_camel_case_types)]
    pub type f32x16 = F32x16;
    #[allow(non_camel_case_types)]
    pub type f64x8 = F64x8;
    #[allow(non_camel_case_types)]
    pub type i8x16 = I8x16;
    #[allow(non_camel_case_types)]
    pub type u32x16 = U32x16;

    // ════════════════════════════════════════════════════════════════════
    // Free hot-kernel functions — v128 counterparts to the NEON kernels in
    // `simd_neon.rs` (dot / Hamming / Base17 / codebook gather).
    // ════════════════════════════════════════════════════════════════════

    /// 4×f32 dot product via v128 multiply + horizontal sum.
    #[inline(always)]
    pub fn dot_f32x4_wasm(a: &[f32; 4], b: &[f32; 4]) -> f32 {
        // SAFETY: both arrays are exactly 16 bytes.
        let (va, vb) = unsafe { (v128_load(a.as_ptr() as *const v128), v128_load(b.as_ptr() as *const v128)) };
        let p = f32x4_mul(va, vb);
        f32x4_extract_lane::<0>(p)
            + f32x4_extract_lane::<1>(p)
            + f32x4_extract_lane::<2>(p)
            + f32x4_extract_lane::<3>(p)
    }

    /// Per-byte population count of a 16-byte register (`i8x16_popcnt`).
    #[inline(always)]
    pub fn popcount_u8x16_wasm(data: v128) -> v128 {
        i8x16_popcnt(data)
    }

    /// Horizontal sum of the 16 unsigned bytes of a v128 into a `u32`.
    #[inline(always)]
    fn hsum_u8x16(v: v128) -> u32 {
        let mut tmp = [0u8; 16];
        // SAFETY: 16-byte store into a 16-byte array.
        unsafe { v128_store(tmp.as_mut_ptr() as *mut v128, v) };
        tmp.iter().map(|&b| b as u32).sum()
    }

    /// Hamming distance of two 16-byte chunks: XOR → popcount → horizontal sum.
    #[inline(always)]
    pub fn hamming_u8x16_wasm(a: &[u8; 16], b: &[u8; 16]) -> u32 {
        // SAFETY: both arrays are exactly 16 bytes.
        let (va, vb) = unsafe { (v128_load(a.as_ptr() as *const v128), v128_load(b.as_ptr() as *const v128)) };
        hsum_u8x16(i8x16_popcnt(v128_xor(va, vb)))
    }

    /// Hamming distance of two 64-byte fingerprints via 4 × v128 XOR+popcount.
    ///
    /// This is the `Fingerprint<256>` / 512-bit-plane distance hot path: four
    /// 16-byte lanes XOR'd, popcounted (`i8x16_popcnt`), and reduced.
    #[inline(always)]
    pub fn hamming_u8x64_wasm(a: &[u8; 64], b: &[u8; 64]) -> u32 {
        let mut total = 0u32;
        for k in 0..4 {
            // SAFETY: offset 16*k stays within the 64-byte arrays for k in 0..4.
            let (va, vb) = unsafe {
                (v128_load(a.as_ptr().add(16 * k) as *const v128), v128_load(b.as_ptr().add(16 * k) as *const v128))
            };
            total += hsum_u8x16(i8x16_popcnt(v128_xor(va, vb)));
        }
        total
    }

    /// Base17 L1 distance: `Σ |a[i] - b[i]|` over 17 `i16` elements.
    ///
    /// Each 8-wide block is sign-extended `i16 → i32`
    /// (`i32x4_extend_{low,high}_i16x8`) **before** the subtract, so
    /// `|a[i] - b[i]|` is computed entirely in i32 and never wraps — bit-exact
    /// against the scalar reference for the full i16 input range (the 17th
    /// element is handled scalar). Differencing in i16 first (as NEON's
    /// `vabdq_s16` does) would wrap when `|a-b| > i16::MAX`; this path is the
    /// stricter i32-domain form, so it matches the scalar reference even on
    /// pathological large-magnitude inputs.
    #[inline(always)]
    pub fn base17_l1_wasm(a: &[i16; 17], b: &[i16; 17]) -> i32 {
        #[inline(always)]
        fn abs_diff_block(a: *const i16, b: *const i16) -> i32 {
            // SAFETY: caller guarantees 8 readable i16 (16 bytes) at each ptr.
            let (va, vb) = unsafe { (v128_load(a as *const v128), v128_load(b as *const v128)) };
            // Sign-extend the low and high 4×i16 halves to 4×i32, difference and
            // abs in i32 (max |a-b| = 65535 fits i32 — no overflow for any i16).
            let dlo = i32x4_sub(i32x4_extend_low_i16x8(va), i32x4_extend_low_i16x8(vb));
            let dhi = i32x4_sub(i32x4_extend_high_i16x8(va), i32x4_extend_high_i16x8(vb));
            let s = i32x4_add(i32x4_abs(dlo), i32x4_abs(dhi));
            i32x4_extract_lane::<0>(s)
                + i32x4_extract_lane::<1>(s)
                + i32x4_extract_lane::<2>(s)
                + i32x4_extract_lane::<3>(s)
        }
        // SAFETY: a/b have 17 elements; the two blocks read indices 0..8 and 8..16.
        let lo = abs_diff_block(a.as_ptr(), b.as_ptr());
        let hi = unsafe { abs_diff_block(a.as_ptr().add(8), b.as_ptr().add(8)) };
        lo + hi + (a[16] as i32 - b[16] as i32).abs()
    }

    /// Codebook gather: accumulate `indices.len()` centroids (each `dim`-wide,
    /// `dim` a multiple of 4) into `output` via v128 adds. v128 counterpart to
    /// `codebook_gather_f32x4_neon`.
    #[inline(always)]
    pub fn codebook_gather_f32x4_wasm(centroids: &[f32], indices: &[u8], dim: usize, output: &mut [f32]) {
        debug_assert!(dim % 4 == 0);
        debug_assert!(output.len() >= dim);
        let chunks = dim / 4;
        for c in 0..chunks {
            let mut acc = f32x4_splat(0.0);
            for &idx in indices {
                let off = idx as usize * dim + c * 4;
                // SAFETY: caller guarantees `idx*dim + dim <= centroids.len()`.
                let v = unsafe { v128_load(centroids.as_ptr().add(off) as *const v128) };
                acc = f32x4_add(acc, v);
            }
            // SAFETY: c*4 + 4 <= dim <= output.len().
            unsafe { v128_store(output.as_mut_ptr().add(c * 4) as *mut v128, acc) };
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // BF16 batch conversion — v128-accelerated (shift, no hardware BF16).
    // ════════════════════════════════════════════════════════════════════

    /// Batch BF16 (`u16` bits) → f32: `f32::from_bits((bits as u32) << 16)`,
    /// 4 lanes at a time via `i32x4` widening + shift. Tail is scalar.
    #[inline(always)]
    pub fn bf16_to_f32_batch_wasm(input: &[u16], output: &mut [f32]) {
        let n = input.len().min(output.len());
        let chunks = n / 4;
        for c in 0..chunks {
            let base = c * 4;
            // Widen 4 × u16 → 4 × u32, shift left 16, reinterpret as f32.
            let widened = i32x4(
                (input[base] as i32) << 16,
                (input[base + 1] as i32) << 16,
                (input[base + 2] as i32) << 16,
                (input[base + 3] as i32) << 16,
            );
            // SAFETY: writes 4 × f32 (16 bytes) at base; base+4 <= n <= output.len().
            unsafe { v128_store(output.as_mut_ptr().add(base) as *mut v128, widened) };
        }
        for i in (chunks * 4)..n {
            output[i] = f32::from_bits((input[i] as u32) << 16);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Tests — run under a wasm runtime (e.g. node) with simd128 enabled.
    // ════════════════════════════════════════════════════════════════════

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn f32x16_load_add_store() {
            let a: [f32; 16] = core::array::from_fn(|i| i as f32);
            let b: [f32; 16] = core::array::from_fn(|i| (i * 10) as f32);
            let c = F32x16::from_slice(&a) + F32x16::from_slice(&b);
            let out = c.to_array();
            for i in 0..16 {
                assert_eq!(out[i], (i + i * 10) as f32);
            }
        }

        #[test]
        fn f32x16_reduce_sum() {
            let v = F32x16::from_array(core::array::from_fn(|i| (i + 1) as f32));
            assert_eq!(v.reduce_sum(), 136.0); // 1..=16
        }

        #[test]
        fn f32x16_mul_add() {
            let r = F32x16::splat(2.0).mul_add(F32x16::splat(3.0), F32x16::splat(1.0));
            assert!((r.reduce_sum() - 112.0).abs() < 1e-4);
        }

        #[test]
        fn f32x16_mask_select() {
            let a = F32x16::from_array(core::array::from_fn(|i| (i + 1) as f32));
            let mask = a.simd_lt(F32x16::splat(8.5));
            let r = mask.select(F32x16::splat(1.0), F32x16::splat(0.0));
            assert!((r.reduce_sum() - 8.0).abs() < 1e-6);
        }

        #[test]
        fn f64x8_mul_add_reduce() {
            let r = F64x8::splat(2.0).mul_add(F64x8::splat(3.0), F64x8::splat(1.0));
            assert_eq!(r.reduce_sum(), 56.0);
        }

        #[test]
        fn i8x16_saturating_abs_min() {
            let r = I8x16::splat(i8::MIN).saturating_abs().to_array();
            assert!(r.iter().all(|&x| x == i8::MAX));
        }

        #[test]
        fn i8x16_from_i4_packed() {
            assert!(I8x16::from_i4_packed_u64(0)
                .to_array()
                .iter()
                .all(|&x| x == 0));
            assert!(I8x16::from_i4_packed_u64(u64::MAX)
                .to_array()
                .iter()
                .all(|&x| x == -1));
        }

        #[test]
        fn hamming_kernels() {
            let a = [0xFFu8; 64];
            let b = [0x00u8; 64];
            assert_eq!(hamming_u8x64_wasm(&a, &b), 512);
            let a16 = [0xFFu8; 16];
            let b16 = [0x0Fu8; 16];
            assert_eq!(hamming_u8x16_wasm(&a16, &b16), 64);
        }

        #[test]
        fn base17_l1_matches_scalar() {
            let a: [i16; 17] = core::array::from_fn(|i| i as i16);
            let b: [i16; 17] = core::array::from_fn(|i| (16 - i) as i16);
            let want: i32 = (0..17).map(|i| (a[i] as i32 - b[i] as i32).abs()).sum();
            assert_eq!(base17_l1_wasm(&a, &b), want);
        }

        #[test]
        fn base17_l1_no_i16_overflow() {
            // Large opposite-sign deltas: |a-b| = 60000 > i16::MAX. The i32-domain
            // path must not wrap (an i16-domain abs-diff would).
            let a: [i16; 17] = core::array::from_fn(|i| if i % 2 == 0 { 30000 } else { -30000 });
            let b: [i16; 17] = core::array::from_fn(|i| if i % 2 == 0 { -30000 } else { 30000 });
            let want: i32 = (0..17).map(|i| (a[i] as i32 - b[i] as i32).abs()).sum();
            assert_eq!(base17_l1_wasm(&a, &b), want);
            assert_eq!(want, 17 * 60000);
        }
    }
}
