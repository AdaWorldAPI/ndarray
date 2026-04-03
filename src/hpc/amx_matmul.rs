//! AMX tile-based matrix multiplication via inline asm (stable Rust 1.94).
//!
//! TDPBUSD: 16×16 tile of u8×i8 → i32 = 256 MACs per instruction.
//! For the ThinkingEngine: builds the 4096² distance table from codebook centroids.
//!
//! Hardware confirmed: AMX-TILE + AMX-INT8 + AMX-BF16 (Sapphire Rapids+).
//! OS enabled: kernel 6.18.5, XCR0 bits 17+18 set.
//! Rust intrinsics: NIGHTLY ONLY (issue #126622).
//! This module: STABLE via inline asm!().
//!
//! Tile registers: 8 tiles, each 16 rows × 64 bytes = 1 KB.
//! For u8: 16×64 = 1024 values per tile.
//! For i32: 16×16 = 256 values per tile (result).
//!
//! One TDPBUSD: C[16×16 i32] += A[16×64 u8] × B[64×16 i8] = 16384 MACs.
//! Compared to VPDPBUSD (64 MACs): 256× more per instruction.

use std::arch::asm;

/// Check if AMX is available AND OS-enabled.
pub fn amx_available() -> bool {
    crate::simd_amx::amx_available()
}

/// AMX tile configuration (64 bytes, must be 64-byte aligned).
#[repr(C, align(64))]
pub struct TileConfig {
    pub data: [u8; 64],
}

impl TileConfig {
    /// Configure for TDPBUSD: C[16×16 i32] += A[16×k u8] × B[k×16 i8].
    ///
    /// Tiles:
    ///   tmm0 = C (result): 16 rows × 64 bytes (16×16 i32)
    ///   tmm1 = A (left):   16 rows × 64 bytes (16×64 u8)
    ///   tmm2 = B (right):  16 rows × 64 bytes (transposed: 64×16 → 16×64)
    pub fn for_dpbusd(k_bytes: u16) -> Self {
        let mut cfg = TileConfig { data: [0u8; 64] };
        cfg.data[0] = 1; // palette 1

        // Tile 0 (C): 16 rows × 64 bytes (16 × i32 per row = 64 bytes)
        cfg.data[16] = 16;
        cfg.data[48] = 64;

        // Tile 1 (A): 16 rows × k_bytes (capped at 64)
        cfg.data[17] = 16;
        cfg.data[50] = k_bytes.min(64) as u8;

        // Tile 2 (B): k_bytes/4 rows × 64 bytes (transposed layout)
        cfg.data[18] = (k_bytes.min(64) / 4) as u8;
        cfg.data[52] = 64;

        cfg
    }
}

/// Load tile configuration via inline asm.
///
/// # Safety
/// Config must be valid and 64-byte aligned.
#[inline]
pub unsafe fn tile_loadconfig(config: &TileConfig) {
    asm!(
        "ldtilecfg [{cfg}]",
        cfg = in(reg) config.data.as_ptr(),
        options(nostack),
    );
}

/// Zero a tile register.
///
/// # Safety
/// Tiles must be configured first via tile_loadconfig.
#[inline]
pub unsafe fn tile_zero(tile: u8) {
    match tile {
        0 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem)),
        1 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc8", options(nostack, nomem)),
        2 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd0", options(nostack, nomem)),
        3 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd8", options(nostack, nomem)),
        _ => {} // tiles 4-7: add when needed
    }
}

/// Release all tile registers.
///
/// # Safety
/// Must be called when done with tile operations.
#[inline]
pub unsafe fn tile_release() {
    asm!(".byte 0xc4, 0xe2, 0x78, 0x49, 0xc0", options(nostack, nomem));
}

/// Load tile from memory.
///
/// # Safety
/// Pointer must be valid, stride must match tile config.
#[inline]
pub unsafe fn tile_load(tile: u8, ptr: *const u8, stride: usize) {
    match tile {
        // TILELOADD tmm0, [ptr + stride*row]
        // Encoding: VEX.128.F2.0F38.W0 4B /r with memory operand
        1 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x0c, 0x08",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        2 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x14, 0x08",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        _ => {}
    }
}

/// Store tile to memory.
///
/// # Safety
/// Pointer must be valid and writable, stride must match.
#[inline]
pub unsafe fn tile_store(tile: u8, ptr: *mut u8, stride: usize) {
    match tile {
        // TILESTORED [ptr + stride*row], tmm0
        0 => asm!(
            ".byte 0xc4, 0xe2, 0x7a, 0x4b, 0x04, 0x08",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        _ => {}
    }
}

/// TDPBUSD: C += A(u8) × B(i8) → i32.
/// tmm0 += tmm1 × tmm2.
///
/// 16×16 output, 64 products per element = 16384 MACs in ONE instruction.
///
/// # Safety
/// Tiles must be loaded with valid data.
#[inline]
pub unsafe fn tile_dpbusd() {
    // TDPBUSD tmm0, tmm1, tmm2
    // VEX.128.F2.0F38.W0 5E C8+reg
    asm!(".byte 0xc4, 0xe2, 0x73, 0x5e, 0xc1", options(nostack, nomem));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_creation() {
        let cfg = TileConfig::for_dpbusd(64);
        assert_eq!(cfg.data[0], 1);      // palette
        assert_eq!(cfg.data[16], 16);     // tile 0 rows
        assert_eq!(cfg.data[48], 64);     // tile 0 colbytes
    }

    #[test]
    fn test_tile_zero_and_release() {
        if !amx_available() {
            eprintln!("AMX not available, skipping");
            return;
        }
        unsafe {
            // Minimal config: just tile 0, 1 row × 4 bytes
            let mut cfg = TileConfig { data: [0u8; 64] };
            cfg.data[0] = 1;     // palette 1
            cfg.data[16] = 1;    // tile 0: 1 row
            cfg.data[48] = 4;    // tile 0: 4 colbytes

            tile_loadconfig(&cfg);
            // TILEZERO tmm0
            asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem));
            // TILERELEASE
            asm!(".byte 0xc4, 0xe2, 0x78, 0x49, 0xc0", options(nostack, nomem));
        }
        eprintln!("AMX tile_zero + tile_release: OK on stable Rust");
    }
}
