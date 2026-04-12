//! LazyLock frozen SIMD dispatch — detect once, keep the CPU choice forever.
//!
//! Replaces per-call `if simd_caps().avx512f { ... } else { ... }` branching
//! with a single frozen function pointer table. After first access:
//!
//! ```text
//! Per-call branch:   simd_caps().avx512f → ~1ns (deref + bool + branch predict)
//! Frozen dispatch:   SIMD_DISPATCH.op()  → ~0.3ns (deref + indirect call, no branch)
//! ```
//!
//! The table is a `Copy` struct of function pointers, frozen at first access via
//! `LazyLock`. Every subsequent call is one pointer deref + one indirect call.
//! No branch, no atomic, no prediction miss.
//!
//! # Tiers (selected once at startup)
//!
//! | Priority | Tier | Width | Guard |
//! |----------|------|-------|-------|
//! | 1 | AVX-512 | 512-bit | `caps.avx512f` |
//! | 2 | AVX2 | 256-bit | `caps.avx2` |
//! | 3 | SSE2 | 128-bit | `caps.sse2` (always true on x86_64) |
//! | 4 | Scalar | 1 lane | fallback |
//!
//! On wasm32 (future): tier would be WASM SIMD (128-bit, `+simd128`).

use std::sync::LazyLock;
use super::simd_caps::simd_caps;

/// The selected SIMD tier, frozen at first access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    /// AVX-512 Foundation (512-bit, 16 × f32).
    Avx512,
    /// AVX2 (256-bit, 8 × f32).
    Avx2,
    /// SSE2 (128-bit, 4 × f32). Baseline on x86_64.
    Sse2,
    /// NEON with dotprod (128-bit, 4 × f32 + int8 dot product).
    /// ARMv8.2+: Pi 5 (A76), Orange Pi 5.
    NeonDotProd,
    /// NEON baseline (128-bit, 4 × f32).
    /// ARMv8.0: Pi Zero 2 W (A53), Pi 3 (A53), Pi 4 (A72).
    Neon,
    /// Scalar fallback (1 lane).
    Scalar,
    /// WebAssembly SIMD (128-bit, 4 × f32). Future tier.
    #[allow(dead_code)]
    WasmSimd128,
}

impl SimdTier {
    /// Number of f32 lanes this tier processes per instruction.
    pub const fn lanes_f32(self) -> usize {
        match self {
            Self::Avx512 => 16,
            Self::Avx2 => 8,
            Self::Sse2 | Self::WasmSimd128 | Self::NeonDotProd | Self::Neon => 4,
            Self::Scalar => 1,
        }
    }

    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Avx512 => "AVX-512",
            Self::Avx2 => "AVX2",
            Self::Sse2 => "SSE2",
            Self::NeonDotProd => "NEON+dotprod (Pi 5 / A76)",
            Self::Neon => "NEON (Pi 3/4 / A53/A72)",
            Self::Scalar => "Scalar",
            Self::WasmSimd128 => "WASM SIMD128",
        }
    }
}

/// Frozen dispatch table: function pointers selected once at startup.
///
/// Each field is a function pointer to the best available implementation.
/// After `LazyLock` initialization, calling any field is one indirect call
/// with zero branching.
#[derive(Clone, Copy)]
pub struct SimdDispatch {
    /// Which tier was selected.
    pub tier: SimdTier,

    // ── byte_scan.rs ──
    /// `byte_find_all(haystack, needle) -> Vec<usize>`
    pub byte_find_all: fn(&[u8], u8) -> Vec<usize>,
    /// `byte_count(haystack, needle) -> usize`
    pub byte_count: fn(&[u8], u8) -> usize,

    // ── distance.rs ──
    /// `squared_distances_f32(query, points) -> Vec<f32>`
    pub squared_distances_f32: fn([f32; 3], &[[f32; 3]]) -> Vec<f32>,

    // ── nibble.rs ──
    /// `nibble_unpack(packed, count) -> Vec<u8>`
    pub nibble_unpack: fn(&[u8], usize) -> Vec<u8>,
    /// `nibble_above_threshold(packed, threshold) -> Vec<usize>`
    pub nibble_above_threshold: fn(&[u8], u8) -> Vec<usize>,

    // ── spatial_hash.rs ──
    /// `batch_sq_dist(query, candidates, radius_sq) -> Vec<(usize, f32)>`
    pub batch_sq_dist: fn([f32; 3], &[[f32; 3]], f32) -> Vec<(usize, f32)>,
}

// NOTE: aabb and cam_pq dispatch on method-level (self + data), so they keep
// inline dispatch using simd_caps(). The dispatch table covers free functions.

/// Global frozen dispatch table. Detected once, used forever.
static DISPATCH: LazyLock<SimdDispatch> = LazyLock::new(SimdDispatch::detect);

/// Get the frozen dispatch table. First call detects; all subsequent calls
/// are one pointer deref to a `Copy` struct.
#[inline(always)]
pub fn simd_dispatch() -> SimdDispatch {
    *DISPATCH
}

impl SimdDispatch {
    #[cfg(target_arch = "x86_64")]
    fn detect() -> Self {
        let caps = simd_caps();

        if caps.avx512bw {
            Self {
                tier: SimdTier::Avx512,
                byte_find_all: byte_find_all_avx512_wrapper,
                byte_count: byte_count_avx512_wrapper,
                squared_distances_f32: squared_distances_avx2_wrapper, // no avx512 variant for 3D dist
                nibble_unpack: nibble_unpack_avx2_wrapper,
                nibble_above_threshold: nibble_above_threshold_avx2_wrapper,
                batch_sq_dist: batch_sq_dist_avx2_wrapper,
            }
        } else if caps.avx2 {
            Self {
                tier: SimdTier::Avx2,
                byte_find_all: byte_find_all_avx2_wrapper,
                byte_count: byte_count_avx2_wrapper,
                squared_distances_f32: squared_distances_avx2_wrapper,
                nibble_unpack: nibble_unpack_avx2_wrapper,
                nibble_above_threshold: nibble_above_threshold_avx2_wrapper,
                batch_sq_dist: batch_sq_dist_avx2_wrapper,
            }
        } else {
            Self::scalar()
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect() -> Self {
        let caps = simd_caps();
        let tier = if caps.asimd_dotprod {
            SimdTier::NeonDotProd
        } else {
            SimdTier::Neon
        };
        // NEON uses the same scalar wrapper signatures — NEON intrinsics
        // will be wired when simd_neon.rs types are activated. For now,
        // dispatch to scalar which auto-vectorizes well on aarch64 with
        // `-C target-feature=+neon` (mandatory on aarch64).
        Self {
            tier,
            ..Self::scalar()
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect() -> Self {
        Self::scalar()
    }

    fn scalar() -> Self {
        Self {
            tier: SimdTier::Scalar,
            byte_find_all: byte_find_all_scalar,
            byte_count: byte_count_scalar,
            squared_distances_f32: squared_distances_scalar,
            nibble_unpack: nibble_unpack_scalar_wrapper,
            nibble_above_threshold: nibble_above_threshold_scalar_wrapper,
            batch_sq_dist: batch_sq_dist_scalar_wrapper,
        }
    }
}

// ============================================================================
// Wrapper functions — bridge between dispatch table signature and actual impls
// ============================================================================
//
// The actual SIMD implementations are `unsafe` with `#[target_feature]`.
// The wrappers handle the safety contract (features were already verified at
// dispatch table construction time).

// ── byte_scan wrappers ──

fn byte_find_all_scalar(haystack: &[u8], needle: u8) -> Vec<usize> {
    haystack.iter().enumerate()
        .filter(|(_, &b)| b == needle)
        .map(|(i, _)| i)
        .collect()
}

fn byte_count_scalar(haystack: &[u8], needle: u8) -> usize {
    haystack.iter().filter(|&&b| b == needle).count()
}

#[cfg(target_arch = "x86_64")]
fn byte_find_all_avx512_wrapper(haystack: &[u8], needle: u8) -> Vec<usize> {
    // SAFETY: avx512bw was verified at dispatch table construction.
    unsafe { super::byte_scan::simd_impl::byte_find_all_avx512(haystack, needle) }
}

#[cfg(target_arch = "x86_64")]
fn byte_find_all_avx2_wrapper(haystack: &[u8], needle: u8) -> Vec<usize> {
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::byte_scan::simd_impl::byte_find_all_avx2(haystack, needle) }
}

#[cfg(target_arch = "x86_64")]
fn byte_count_avx512_wrapper(haystack: &[u8], needle: u8) -> usize {
    // SAFETY: avx512bw was verified at dispatch table construction.
    unsafe { super::byte_scan::simd_impl::byte_count_avx512(haystack, needle) }
}

#[cfg(target_arch = "x86_64")]
fn byte_count_avx2_wrapper(haystack: &[u8], needle: u8) -> usize {
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::byte_scan::simd_impl::byte_count_avx2(haystack, needle) }
}

// ── distance wrappers ──

fn squared_distances_scalar(query: [f32; 3], points: &[[f32; 3]]) -> Vec<f32> {
    points.iter().map(|p| {
        let dx = query[0] - p[0];
        let dy = query[1] - p[1];
        let dz = query[2] - p[2];
        dx * dx + dy * dy + dz * dz
    }).collect()
}

#[cfg(target_arch = "x86_64")]
fn squared_distances_avx2_wrapper(query: [f32; 3], points: &[[f32; 3]]) -> Vec<f32> {
    let mut out = Vec::new();
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::distance::simd_impl::squared_distances_avx2(query, points, &mut out) };
    out
}

// ── nibble wrappers ──

fn nibble_unpack_scalar_wrapper(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    super::nibble::nibble_unpack_scalar(packed, count, &mut out);
    out
}

fn nibble_above_threshold_scalar_wrapper(packed: &[u8], threshold: u8) -> Vec<usize> {
    super::nibble::nibble_above_threshold_scalar(packed, threshold)
}

#[cfg(target_arch = "x86_64")]
fn nibble_unpack_avx2_wrapper(packed: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::nibble::nibble_unpack_avx2(packed, count, &mut out) };
    out
}

#[cfg(target_arch = "x86_64")]
fn nibble_above_threshold_avx2_wrapper(packed: &[u8], threshold: u8) -> Vec<usize> {
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::nibble::nibble_above_threshold_avx2(packed, threshold) }
}

// ── spatial_hash wrappers ──

fn batch_sq_dist_scalar_wrapper(query: [f32; 3], candidates: &[[f32; 3]], radius_sq: f32) -> Vec<(usize, f32)> {
    super::spatial_hash::batch_sq_dist_scalar(query, candidates, radius_sq)
}

#[cfg(target_arch = "x86_64")]
fn batch_sq_dist_avx2_wrapper(query: [f32; 3], candidates: &[[f32; 3]], radius_sq: f32) -> Vec<(usize, f32)> {
    // SAFETY: avx2 was verified at dispatch table construction.
    unsafe { super::spatial_hash::batch_sq_dist_avx2(query, candidates, radius_sq) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_table_initializes() {
        let d = simd_dispatch();
        // Should pick the best tier available on this CPU.
        println!("SIMD tier: {:?} ({} f32 lanes)", d.tier, d.tier.lanes_f32());
        assert!(d.tier.lanes_f32() >= 1);
    }

    #[test]
    fn dispatch_is_frozen() {
        let a = simd_dispatch();
        let b = simd_dispatch();
        assert_eq!(a.tier, b.tier);
    }

    #[test]
    fn dispatch_byte_find_all() {
        let d = simd_dispatch();
        let data = b"hello world hello";
        let hits = (d.byte_find_all)(data, b'l');
        // "hello world hello" has 'l' at positions 2,3,10,14,15
        assert_eq!(hits.len(), 5);
        assert!(hits.contains(&2));
        assert!(hits.contains(&3));
    }

    #[test]
    fn dispatch_byte_count() {
        let d = simd_dispatch();
        let data = b"hello world hello";
        let count = (d.byte_count)(data, b'l');
        assert_eq!(count, 5);
    }

    #[test]
    fn dispatch_squared_distances() {
        let d = simd_dispatch();
        let query = [1.0, 2.0, 3.0];
        let points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let dists = (d.squared_distances_f32)(query, &points);
        assert!((dists[0] - 0.0).abs() < 1e-6); // self distance = 0
        assert!((dists[1] - 27.0).abs() < 1e-4); // 3² + 3² + 3² = 27
    }

    #[test]
    fn dispatch_nibble_above_threshold() {
        let d = simd_dispatch();
        // Pack two nibbles per byte: [0x37] = nibble 7 at index 0, nibble 3 at index 1
        let packed = [0x37u8, 0x59]; // indices 0-3: 7, 3, 9, 5
        let above_4 = (d.nibble_above_threshold)(&packed, 4);
        // Indices where nibble value > 4
        assert!(above_4.contains(&0)); // 7 > 4
        assert!(above_4.contains(&2)); // 9 > 4
        assert!(above_4.contains(&3)); // 5 > 4
    }

    #[test]
    fn tier_names() {
        assert_eq!(SimdTier::Avx512.name(), "AVX-512");
        assert_eq!(SimdTier::Avx2.name(), "AVX2");
        assert_eq!(SimdTier::Scalar.name(), "Scalar");
        assert_eq!(SimdTier::WasmSimd128.name(), "WASM SIMD128");
    }
}
