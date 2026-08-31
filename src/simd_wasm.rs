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

    /// BLAKE3 `hash_many` shuffle surface — the transpose network's four unpacks
    /// and two half-concatenations, at DEGREE 16.
    ///
    /// **Why these live on `U32x16` and not on a half-width type.** BLAKE3's
    /// AVX2 backend is DEGREE 8 over `__m256i`. Two of those fit in one
    /// `U32x16`, and every operation in `hash_many` is either lane-wise
    /// (add / xor / rotate) or confined *within* a 128- or 256-bit lane — which
    /// is exactly what the methods below preserve. The two 8-lane groups
    /// therefore never interact, and the algorithm runs on both at once as
    /// DEGREE 16 with no cross-talk. A `U32x8` is neither needed nor wanted
    /// (operator ruling, 2026-07-28: a half-width type standing in for the lane
    /// the substrate actually uses is an absolute no-go).
    ///
    /// Semantics are x86's, applied to **each 256-bit half independently**:
    /// `interleave_*_u32` / `interleave_*_u64` reproduce
    /// `_mm256_unpack{lo,hi}_epi{32,64}` within each 128-bit quad, and
    /// `concat_{lo,hi}_halves` reproduce
    /// `_mm256_permute2x128_si256(_, _, 0x20 / 0x31)` within each 256-bit half.
    /// The per-lane structure is load-bearing: BLAKE3's transpose is defined in
    /// terms of it, so a "helpful" whole-vector interleave would compute a
    /// different permutation and produce wrong hashes with no compile error.
    ///
    /// Every body is a plain index loop. The codegen oracle
    /// (`.claude/knowledge/simd-codegen-oracle/`) measured this exact shape:
    /// fixed two-source permutations written as index loops compile to real
    /// packed shuffles, and a transpose composed from them emits
    /// `vpunpcklqdq` / `vpermq` / `vinserti128`. No `unsafe`, no `core::arch`,
    /// no intrinsic override earned. See
    /// `.claude/knowledge/blake3-on-ndarray-simd.md`.
    impl U32x16 {
        /// Set difference: `self & !other`, lane-wise — one `v128.andnot` per
        /// 128-bit part (the wasm intrinsic's argument order already matches
        /// this crate's direction). Reference doc: `simd_avx512::U32x16::andnot`.
        ///
        /// # Examples
        ///
        /// ```
        /// use ndarray::simd::U32x16;
        /// let a = U32x16::splat(0b1100);
        /// let b = U32x16::splat(0b1010);
        /// assert_eq!(a.andnot(b).to_array()[0], 0b0100); // a & !b
        /// ```
        #[inline(always)]
        pub fn andnot(self, other: Self) -> Self {
            Self([
                U32x4(v128_andnot(self.0[0].0, other.0[0].0)),
                U32x4(v128_andnot(self.0[1].0, other.0[1].0)),
                U32x4(v128_andnot(self.0[2].0, other.0[2].0)),
                U32x4(v128_andnot(self.0[3].0, other.0[3].0)),
            ])
        }

        /// Any 3-input boolean function of `self`, `b` and `c` — Intel's
        /// VPTERNLOG convention, matched exactly by every backend. Named
        /// immediates: `crate::simd::ternlog`. Only `0..=255` is legal,
        /// enforced at compile time on every backend. Composed per 128-bit
        /// part from `v128` bit ops; the minterm branches fold at compile
        /// time, so only the truth table's terms survive.
        ///
        /// # Examples
        ///
        /// ```
        /// use ndarray::simd::{ternlog, U32x16};
        /// let (a, b, c) = (U32x16::splat(0b1100), U32x16::splat(0b1010), U32x16::splat(0b1001));
        /// let maj = a.ternlog::<{ ternlog::MAJ3 }>(b, c); // two-of-three majority
        /// assert_eq!(maj.to_array()[0], 0b1000);
        /// ```
        #[inline(always)]
        pub fn ternlog<const IMM: i32>(self, b: Self, c: Self) -> Self {
            const { assert!(IMM >= 0 && IMM <= 255, "ternlog IMM is an 8-bit truth table") }
            let mut parts = [self.0[0].0; 4];
            for p in 0..4 {
                let (x, y, z) = (self.0[p].0, b.0[p].0, c.0[p].0);
                let mut r = v128_xor(x, x); // zero
                if IMM & 0x01 != 0 {
                    r = v128_or(r, v128_and(v128_not(x), v128_andnot(v128_not(y), z)));
                }
                if IMM & 0x02 != 0 {
                    r = v128_or(r, v128_and(v128_not(x), v128_andnot(z, y)));
                }
                if IMM & 0x04 != 0 {
                    r = v128_or(r, v128_and(v128_not(x), v128_andnot(y, z)));
                }
                if IMM & 0x08 != 0 {
                    r = v128_or(r, v128_and(v128_not(x), v128_and(y, z)));
                }
                if IMM & 0x10 != 0 {
                    r = v128_or(r, v128_and(x, v128_andnot(v128_not(y), z)));
                }
                if IMM & 0x20 != 0 {
                    r = v128_or(r, v128_and(x, v128_andnot(z, y)));
                }
                if IMM & 0x40 != 0 {
                    r = v128_or(r, v128_and(x, v128_andnot(y, z)));
                }
                if IMM & 0x80 != 0 {
                    r = v128_or(r, v128_and(x, v128_and(y, z)));
                }
                parts[p] = r;
            }
            Self([U32x4(parts[0]), U32x4(parts[1]), U32x4(parts[2]), U32x4(parts[3])])
        }

        /// `_mm256_unpacklo_epi32` per 256-bit half: within each 128-bit quad,
        /// interleave the low two `u32` of each operand.
        #[inline(always)]
        pub fn interleave_lo_u32(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for q in 0..4 {
                let i = 4 * q;
                o[i] = a[i];
                o[i + 1] = b[i];
                o[i + 2] = a[i + 1];
                o[i + 3] = b[i + 1];
            }
            Self::from_array(o)
        }

        /// `_mm256_unpackhi_epi32` per 256-bit half: within each 128-bit quad,
        /// interleave the high two `u32` of each operand.
        #[inline(always)]
        pub fn interleave_hi_u32(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for q in 0..4 {
                let i = 4 * q;
                o[i] = a[i + 2];
                o[i + 1] = b[i + 2];
                o[i + 2] = a[i + 3];
                o[i + 3] = b[i + 3];
            }
            Self::from_array(o)
        }

        /// `_mm256_unpacklo_epi64` per 256-bit half: within each 128-bit quad,
        /// the low `u64` of each operand.
        #[inline(always)]
        pub fn interleave_lo_u64(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for q in 0..4 {
                let i = 4 * q;
                o[i] = a[i];
                o[i + 1] = a[i + 1];
                o[i + 2] = b[i];
                o[i + 3] = b[i + 1];
            }
            Self::from_array(o)
        }

        /// `_mm256_unpackhi_epi64` per 256-bit half: within each 128-bit quad,
        /// the high `u64` of each operand.
        #[inline(always)]
        pub fn interleave_hi_u64(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for q in 0..4 {
                let i = 4 * q;
                o[i] = a[i + 2];
                o[i + 1] = a[i + 3];
                o[i + 2] = b[i + 2];
                o[i + 3] = b[i + 3];
            }
            Self::from_array(o)
        }

        /// `_mm256_permute2x128_si256(a, b, 0x20)` per 256-bit half: the low
        /// 128-bit lane of each operand, concatenated.
        ///
        /// Only the two immediates BLAKE3's transpose uses are exposed, as named
        /// methods rather than a generic `const IMM` permute: the remaining
        /// immediates would have no caller and no parity test.
        #[inline(always)]
        pub fn concat_lo_halves(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for h in 0..2 {
                let i = 8 * h;
                o[i..i + 4].copy_from_slice(&a[i..i + 4]);
                o[i + 4..i + 8].copy_from_slice(&b[i..i + 4]);
            }
            Self::from_array(o)
        }

        /// `_mm256_permute2x128_si256(a, b, 0x31)` per 256-bit half: the high
        /// 128-bit lane of each operand, concatenated.
        #[inline(always)]
        pub fn concat_hi_halves(self, other: Self) -> Self {
            let (a, b) = (self.to_array(), other.to_array());
            let mut o = [0u32; 16];
            for h in 0..2 {
                let i = 8 * h;
                o[i..i + 4].copy_from_slice(&a[i + 4..i + 8]);
                o[i + 4..i + 8].copy_from_slice(&b[i + 4..i + 8]);
            }
            Self::from_array(o)
        }

        /// One butterfly exchange at block granularity `G` elements — the general
        /// form of the whole unpack / lane-exchange family, parameterized by
        /// granularity instead of one method per width.
        ///
        /// `G = 1` is the 32-bit unpack, `G = 2` the 64-bit unpack, `G = 4` the
        /// 128-bit lane exchange, and `G = 8` the 256-bit half exchange that a
        /// 512-bit lane additionally needs and for which no AVX2 intrinsic exists.
        /// `G` is a const parameter, so every shuffle pattern is compile-time
        /// constant — the same property a hand-written intrinsic has, and the
        /// precondition for LLVM to select a shuffle rather than an indexed copy.
        ///
        /// Four stages over this (`G` = 1, 2, 4, 8, pairing row `r` with `r | G`)
        /// compose a complete 16x16 transpose. Measured as
        /// `transpose_16x16_composed`: **79 packed / 0 scalar-lane-arith**, 19 of
        /// them real shuffles. The same transpose written as one monolithic index
        /// loop measures **0 packed** — 1088 bytes of stack and a 256-iteration
        /// scalar copy. The spelling is the entire difference. See
        /// `.claude/knowledge/blake3-on-ndarray-simd.md`.
        ///
        /// No intrinsic override is earned: the generic form does not fail.
        #[inline(always)]
        pub fn exchange<const G: usize>(self, other: Self) -> (Self, Self) {
            let (l, h) = (self.to_array(), other.to_array());
            let mut nl = [0u32; 16];
            let mut nh = [0u32; 16];
            for c in 0..16 {
                nl[c] = if c & G == 0 { l[c] } else { h[c ^ G] };
                nh[c] = if c & G != 0 { h[c] } else { l[c ^ G] };
            }
            (Self::from_array(nl), Self::from_array(nh))
        }
    }

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

        /// Lane-wise equality as a packed 16-bit bitmask.
        ///
        /// Bit `i` of the result is set iff `self.lane(i) == other.lane(i)`.
        /// Bit order is **LSB-first**: lane `0` occupies bit `0`. Same
        /// convention as `I32x16::cmpge_zero_mask` / `I32x16::gt_bitmask`
        /// (which on wasm32 come from the scalar tier).
        ///
        /// Edge cases: equality is exact bitwise comparison over the full
        /// 32-bit range, so `u32::MAX` and `0` behave like any other value —
        /// no saturation, wrapping, or signedness question arises.
        ///
        /// Plain index loop over the `[U32x4; 4]` fan-out; LLVM lowers this
        /// shape to `i32x4.eq` plus a narrowing extraction under `simd128`.
        #[inline(always)]
        pub fn eq_bitmask(self, other: Self) -> u16 {
            let (a, b) = (self.to_array(), other.to_array());
            let mut mask = 0u16;
            for i in 0..16 {
                if a[i] == b[i] {
                    mask |= 1 << i;
                }
            }
            mask
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
    // INT8 dot / GEMM — i8×i8→i32 via i16-widen + i32-pairwise-reduce.
    //
    // wasm32 SIMD128 has no VNNI-style unsigned×signed dot-product
    // instruction (no `VPDPBUSD` analogue — see `hpc::amx_matmul`'s AMX /
    // AVX-512-VNNI / AVX-VNNI tiers on x86_64, which all shift i8 → u8 to
    // feed exactly such an instruction), so this is the wasm32+simd128
    // tier of `matmul_i8_to_i32`'s dispatch ladder: signed i8×i8 products
    // widen to i16 via `i16x8_extmul_{low,high}_i8x16` (both operands are
    // read as SIGNED `i8x16` — confirmed against the `core::arch::wasm32`
    // source, which implements these via a sign-extending
    // `simd_cast::<i8x8, i16x8>`, distinct from the separate `_u8x16`
    // unsigned entry points), then each i16x8 half reduces to i32x4 via
    // `i32x4_extadd_pairwise_i16x8` (signed adjacent-pair sum).
    // ════════════════════════════════════════════════════════════════════

    /// i8×i8→i32 dot product over the first `k` elements of `a`/`b`.
    ///
    /// **Exactness.** `i8::MIN * i8::MIN = (-128) * (-128) = 16384` is the
    /// largest possible `|product|`; it fits `i16` (max `32767`) with
    /// headroom, so `i16x8_extmul_{low,high}_i8x16` never overflows. The
    /// pairwise sum of two such products maxes at `32768`, still well
    /// inside `i32`, so `i32x4_extadd_pairwise_i16x8` never overflows
    /// either. Accumulating the per-chunk `i32x4` partials (`i32x4_add`)
    /// and horizontal-reducing at the end is therefore bit-identical, for
    /// any `k`, to the plain scalar reference `Σ a[i] as i32 * b[i] as
    /// i32` — up to the same `i32`-accumulator overflow ceiling the
    /// scalar form already has (roughly `k > i32::MAX / 16384 ≈
    /// 131_072`, unreachable for any realistic GEMM dimension).
    ///
    /// **No sign-shift bias trick needed.** Unlike the AMX / AVX-512-VNNI
    /// / AVX-VNNI tiers in `hpc::amx_matmul::matmul_i8_to_i32` (which
    /// shift i8 → u8 by `+128` to feed an unsigned×signed instruction,
    /// then subtract `128 · colsum(B)` to undo the bias), signed×signed
    /// multiply is native to this i16-widening path, so no bias
    /// correction is required here.
    ///
    /// `a` and `b` must have length `>= k`; only the first `k` elements
    /// of each are read. The tail (`k % 16 != 0`) is a plain scalar
    /// accumulation over the remaining `< 16` elements, using the same
    /// `i32` accumulation as the SIMD path.
    #[inline(always)]
    pub fn dot_i8_to_i32_wasm(a: &[i8], b: &[i8], k: usize) -> i32 {
        debug_assert!(a.len() >= k && b.len() >= k);
        let chunks = k / 16;
        let tail = k % 16;

        let mut acc = i32x4_splat(0);
        for c in 0..chunks {
            let off = c * 16;
            // SAFETY: `off + 16 <= chunks * 16 <= k <= a.len()` (and
            // likewise for `b`), so both loads read exactly 16 in-bounds
            // bytes.
            let (va, vb) = unsafe {
                (v128_load(a.as_ptr().add(off) as *const v128), v128_load(b.as_ptr().add(off) as *const v128))
            };
            let lo = i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(va, vb));
            let hi = i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(va, vb));
            acc = i32x4_add(acc, i32x4_add(lo, hi));
        }

        let mut sum = i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc);

        for t in 0..tail {
            let idx = chunks * 16 + t;
            sum += a[idx] as i32 * b[idx] as i32;
        }
        sum
    }

    /// i8×i8 → i32 GEMM: `c[i*n+j] = Σ_p a[i*k+p] * b[p*n+j]`.
    ///
    /// Row-major `a` (M×K), row-major `b` (K×N), row-major `c` (M×N,
    /// fully OVERWRITTEN — not accumulated). `b`'s "columns" are strided
    /// (stride `n`) in its row-major layout, which the contiguous v128
    /// loads in [`dot_i8_to_i32_wasm`] can't walk directly, so this
    /// transposes `b` into an `n × k` column-major scratch buffer ONCE (a
    /// plain `O(k·n)` data reshuffle — not a change to the dot-product
    /// kernel's intrinsic sequence itself) and then calls
    /// `dot_i8_to_i32_wasm` once per output cell, matching row-major `a`
    /// against the now-contiguous transposed column of `b`.
    ///
    /// Bit-identical to the plain scalar `i8×i8→i32` triple-loop
    /// reference for any `m`/`n`/`k` (integer addition is associative /
    /// commutative, so summing the `k` products in a different order —
    /// per-cell dot product here vs. the scalar reference's `ikj`
    /// accumulation order — cannot change the exact result, only the
    /// memory-access pattern).
    #[inline(always)]
    pub fn matmul_i8_to_i32_wasm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
        debug_assert!(a.len() >= m * k);
        debug_assert!(b.len() >= k * n);
        debug_assert!(c.len() >= m * n);

        let mut b_t = vec![0i8; n * k];
        for p in 0..k {
            for j in 0..n {
                b_t[j * k + p] = b[p * n + j];
            }
        }
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            for j in 0..n {
                let b_col = &b_t[j * k..j * k + k];
                c[i * n + j] = dot_i8_to_i32_wasm(a_row, b_col, k);
            }
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

        // ── SplitMix64 — deterministic PRNG shared by the INT8 GEMM parity
        // tests below (self-contained, no external dep). ──────────────────
        struct SplitMix64(u64);
        impl SplitMix64 {
            fn next_u64(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            fn next_i8(&mut self) -> i8 {
                (self.next_u64() & 0xFF) as u8 as i8
            }
        }

        /// Direct parity test of the dot-product primitive (not routed
        /// through the GEMM/transpose wrapper) across `k` values that hit:
        /// zero, sub-chunk, exactly one chunk, multiple chunks, and
        /// multiple chunks plus a tail.
        #[test]
        fn dot_i8_to_i32_wasm_matches_scalar_various_k() {
            for &k in &[0usize, 1, 3, 15, 16, 17, 31, 32, 33, 100] {
                let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0 ^ (k as u64));
                let a: Vec<i8> = (0..k).map(|_| rng.next_i8()).collect();
                let b: Vec<i8> = (0..k).map(|_| rng.next_i8()).collect();
                let expected: i32 = (0..k).map(|i| a[i] as i32 * b[i] as i32).sum();
                assert_eq!(dot_i8_to_i32_wasm(&a, &b, k), expected, "mismatch at k={k}");
            }
        }

        /// W1a-mandatory parity test: `matmul_i8_to_i32_wasm` (the wasm32
        /// SIMD128 arm feeding `matmul_i8_to_i32`'s dispatch ladder) vs a
        /// scalar `i8×i8→i32` GEMM reference, across several shapes
        /// including K not a multiple of 16. The scalar reference here
        /// replicates `hpc::amx_matmul::matmul_i8_to_i32`'s tier-4 scalar
        /// fallback verbatim (that module is `#[cfg(target_arch =
        /// "x86_64")]`-gated and therefore unreachable from a wasm32 test
        /// binary, so the reference is inlined rather than called into —
        /// same math: `c[i,j] = Σ_p a[i,p] as i32 * b[p,j] as i32`, plain
        /// `i32` accumulation, no sign-shift bias).
        #[test]
        fn wasm_matmul_i8_matches_scalar() {
            fn scalar_reference(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
                let mut c = vec![0i32; m * n];
                for i in 0..m {
                    for p in 0..k {
                        let av = a[i * k + p] as i32;
                        for j in 0..n {
                            c[i * n + j] += av * b[p * n + j] as i32;
                        }
                    }
                }
                c
            }

            // Shapes deliberately include: exact multiples of 16, K not a
            // multiple of 16 (tail exercised), K < 16 (tail-only, zero
            // full chunks), and non-square M/N so row/col mixups show up.
            let shapes: &[(usize, usize, usize)] =
                &[(1, 1, 1), (4, 3, 5), (2, 2, 16), (3, 5, 17), (8, 4, 31), (5, 6, 64), (7, 1, 33), (1, 9, 100)];

            let mut rng = SplitMix64(0xC0FF_EE12_3456_78);
            for &(m, n, k) in shapes {
                let mut a: Vec<i8> = (0..m * k).map(|_| rng.next_i8()).collect();
                let mut b: Vec<i8> = (0..k * n).map(|_| rng.next_i8()).collect();
                // Force the i8::MIN/i8::MAX extremes in explicitly: the
                // exactness argument in `dot_i8_to_i32_wasm`'s doc comment
                // hinges on i8::MIN * i8::MIN being the largest-magnitude
                // product, so it must appear in the corpus rather than
                // rely on the PRNG landing on it.
                if !a.is_empty() {
                    a[0] = i8::MIN;
                }
                if !b.is_empty() {
                    b[0] = i8::MIN;
                }
                if a.len() > 1 {
                    a[1] = i8::MAX;
                }
                if b.len() > 1 {
                    b[1] = i8::MAX;
                }

                let expected = scalar_reference(&a, &b, m, n, k);
                let mut actual = vec![0i32; m * n];
                matmul_i8_to_i32_wasm(&a, &b, &mut actual, m, n, k);

                assert_eq!(actual, expected, "mismatch at shape m={m} n={n} k={k}");
            }
        }
    }
}
