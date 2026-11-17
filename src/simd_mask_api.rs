//! Shared mask API for SIMD compare results stored as a packed bitmask
//! (bit `i` = lane `i`): AVX-512 `__mmask8` / `__mmask16`, and the `u8` /
//! `u16` masks of the scalar, NEON and wasm backends.
//!
//! Masks stored as full-width lane vectors (the x86 256-bit `F32Mask8` /
//! `F64Mask4`, and the AVX2 512-bit pairs built from them) implement the same
//! API by hand in their own backend file, so that consumer code reads
//! identically on every arm:
//!
//! | method | meaning |
//! |---|---|
//! | `from_bitmask(bits)` | lane `i` set iff bit `i` is set; bits at or above `LANES` are ignored |
//! | `any()` / `all()` / `none()` | some / every / no lane set |
//! | `a & b`, `a \| b`, `a ^ b`, `!a` | lane-wise logic; `!` only flips the `LANES` live bits |

/// Implements `from_bitmask`, `any`, `all`, `none` and the bit operators for a
/// mask newtype whose field `.0` is a packed bitmask of type `$prim`.
///
/// `$vec` is the vector type whose comparisons produce this mask; it is used
/// only for the doc examples.
#[allow(unused_macros)] // not every arm instantiates it (e.g. nightly-simd off x86)
macro_rules! impl_bitmask_mask_api {
    ($mask:ident, $prim:ty, $lanes:expr, $vec:ident) => {
        impl $mask {
            /// The live bits of this mask: one per lane, LSB-first.
            const LIVE_BITS: $prim = ((1u64 << $lanes) - 1) as $prim;

            #[doc = concat!(
                "Builds a mask from a packed bitmask: lane `i` is set iff bit `i` is set.\n",
                "Bits at or above `", stringify!($lanes), "` are ignored.\n\n",
                "```\n",
                "use ndarray::simd::{", stringify!($mask), ", ", stringify!($vec), "};\n",
                "let m = ", stringify!($mask), "::from_bitmask(0b101);\n",
                "assert_eq!(m.to_bitmask(), 0b101);\n",
                "let v = m.select(", stringify!($vec), "::splat(1.0), ", stringify!($vec), "::splat(0.0)).to_array();\n",
                "assert_eq!((v[0], v[1], v[2]), (1.0, 0.0, 1.0));\n",
                "```"
            )]
            #[inline(always)]
            pub fn from_bitmask(bits: $prim) -> Self {
                Self(bits & Self::LIVE_BITS)
            }

            #[doc = concat!(
                "`true` if at least one lane is set.\n\n",
                "```\n",
                "use ndarray::simd::", stringify!($vec), ";\n",
                "let x = ", stringify!($vec), "::splat(3.0);\n",
                "assert!(x.simd_lt(", stringify!($vec), "::splat(4.0)).any());\n",
                "assert!(!x.simd_lt(", stringify!($vec), "::splat(2.0)).any());\n",
                "```"
            )]
            #[inline(always)]
            pub fn any(self) -> bool {
                (self.0 & Self::LIVE_BITS) != 0
            }

            #[doc = concat!(
                "`true` if every lane is set.\n\n",
                "```\n",
                "use ndarray::simd::", stringify!($vec), ";\n",
                "let x = ", stringify!($vec), "::splat(3.0);\n",
                "assert!(x.simd_le(", stringify!($vec), "::splat(3.0)).all());\n",
                "```"
            )]
            #[inline(always)]
            pub fn all(self) -> bool {
                (self.0 & Self::LIVE_BITS) == Self::LIVE_BITS
            }

            #[doc = concat!(
                "`true` if no lane is set — the early-exit test of a masked loop.\n\n",
                "```\n",
                "use ndarray::simd::", stringify!($vec), ";\n",
                "let x = ", stringify!($vec), "::splat(5.0);\n",
                "assert!(x.simd_lt(", stringify!($vec), "::splat(1.0)).none());\n",
                "```"
            )]
            #[inline(always)]
            pub fn none(self) -> bool {
                !self.any()
            }
        }

        impl core::ops::BitAnd for $mask {
            type Output = Self;
            /// Lane-wise AND: set where both masks are set.
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl core::ops::BitOr for $mask {
            type Output = Self;
            /// Lane-wise OR: set where either mask is set.
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl core::ops::BitXor for $mask {
            type Output = Self;
            /// Lane-wise XOR: set where exactly one mask is set.
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl core::ops::Not for $mask {
            type Output = Self;
            /// Lane-wise NOT over the live lanes only.
            #[inline(always)]
            fn not(self) -> Self {
                Self(!self.0 & Self::LIVE_BITS)
            }
        }
    };
}
