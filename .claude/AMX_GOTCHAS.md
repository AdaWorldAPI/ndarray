# AMX Gotchas — Resolved on Stable Rust 1.94

> Updated: 2026-04-03
> CPU: Sapphire Rapids (AMX-TILE + AMX-INT8 + AMX-BF16 confirmed)
> Kernel: 6.18.5 (XCR0 bits 17+18 enabled)

---

## Status

AMX works on **stable Rust 1.94** via `asm!()`. No nightly needed.

```
LDTILECFG:   ✓  (load tile configuration)
TILEZERO:    ✓  (zero a tile register)
TILERELEASE: ✓  (release tiles)
TDPBUSD:     ✓  (u8×i8 tile dot product, 256 MACs/instruction)
```

---

## Gotcha 1: Rust intrinsics are NIGHTLY ONLY

```rust
// This DOES NOT compile on stable:
use std::arch::x86_64::_tile_loadconfig;  // error: unstable feature x86_amx_intrinsics
```

**Fix**: Use `asm!()` (stable since Rust 1.59):
```rust
asm!("ldtilecfg [{}]", in(reg) config.data.as_ptr(), options(nostack));
```

Tracking issue: https://github.com/rust-lang/rust/issues/126622

---

## Gotcha 2: Tile config MUST be 64-byte aligned

```rust
// This SEGFAULTS:
let config = [0u8; 64];  // stack-allocated, no alignment guarantee

// This WORKS:
#[repr(C, align(64))]
struct TileConfig { data: [u8; 64] }
let config = TileConfig { data: [0u8; 64] };
```

LDTILECFG reads 64 bytes from the pointer. If not 64-byte aligned,
the CPU raises #GP (general protection fault) → SIGSEGV.

---

## Gotcha 3: rbx is LLVM-reserved

```rust
// This DOES NOT compile:
asm!("cpuid", out("ebx") ebx, ...);  // error: rbx is used internally by LLVM

// This WORKS:
let result = core::arch::x86_64::__cpuid_count(7, 0);  // stable, handles rbx internally
```

For CPUID leaf 7 (AMX detection): use `__cpuid_count()`, not inline asm.

---

## Gotcha 4: OS must enable AMX via XSETBV + process must request permission

AMX tiles are large (8 KB of state). Two levels of OS enablement required:

1. **Kernel enables tile state in XCR0** (bits 17+18). Linux 5.19+ does this.
2. **Process requests XCOMP_PERM** via `prctl(ARCH_REQ_XCOMP_PERM, 18)`.
   Without this, LDTILECFG will SIGILL even if XCR0 bits are set.

**Detection (stable)**:
```rust
// Step 1: CPUID — does CPU support AMX?
let cpuid = core::arch::x86_64::__cpuid_count(7, 0);
let amx_tile = (cpuid.edx >> 24) & 1;
let amx_int8 = (cpuid.edx >> 25) & 1;

// Step 2: OSXSAVE — does OS support XSAVE?
let cpuid_01 = core::arch::x86_64::__cpuid(1);
let osxsave = (cpuid_01.ecx >> 27) & 1;

// Step 3: _xgetbv(0) — did OS ACTUALLY enable tile state?
// ⚠ Do NOT use __cpuid_count(0xD, 0) — that reports what CPU SUPPORTS,
//   not what the OS ENABLED. _xgetbv(0) reads the actual XCR0 register.
let xcr0: u64 = unsafe { core::arch::x86_64::_xgetbv(0) };
let tilecfg  = (xcr0 >> 17) & 1;  // bit 17 = XTILECFG
let tiledata = (xcr0 >> 18) & 1;  // bit 18 = XTILEDATA

// Step 4: prctl — request tile permission for this process
// SYS_prctl = 157, ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18
// Returns 0 on success, -errno on failure. Idempotent.
```

**Previous bug**: `__cpuid_count(0xD, 0)` reports XSAVE state component bitmap
(what the CPU *supports*), NOT the actual XCR0 value (what the OS *enabled*).
On hypervisors that advertise AMX in CPUID but don't enable tile state,
the old check returned `true` → SIGILL on LDTILECFG.

---

## Gotcha 5: TILEZERO/TILERELEASE need manual byte encoding

The Rust assembler on some toolchains doesn't know AMX mnemonics.
Use raw instruction bytes:

```rust
// TILEZERO tmm0
asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem));

// TILEZERO tmm1
asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc8", options(nostack, nomem));

// TILEZERO tmm2
asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd0", options(nostack, nomem));

// TILEZERO tmm3
asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd8", options(nostack, nomem));

// TILERELEASE
asm!(".byte 0xc4, 0xe2, 0x78, 0x49, 0xc0", options(nostack, nomem));

// TDPBUSD tmm0, tmm1, tmm2 (C += A × B)
asm!(".byte 0xc4, 0xe2, 0x73, 0x5e, 0xc1", options(nostack, nomem));
```

Note: LDTILECFG works as a mnemonic:
```rust
asm!("ldtilecfg [{}]", in(reg) ptr, options(nostack));
```

---

## Gotcha 6: Tile config field layout is not obvious

The 64-byte tile config structure:
```
Byte  0:     palette (must be 1)
Bytes 1-15:  reserved (zero)
Bytes 16-23: rows per tile (tile 0 at byte 16, tile 1 at byte 17, ...)
Bytes 24-47: reserved (zero)
Bytes 48-63: colbytes per tile (tile 0 at [48..49] as u16 LE, tile 1 at [50..51], ...)
```

For TDPBUSD (u8×i8 → i32):
- Tile 0 (C result): rows=16, colbytes=64 (16 × i32 = 64 bytes per row)
- Tile 1 (A input):  rows=16, colbytes=64 (16 × 64 u8)
- Tile 2 (B input):  rows=16, colbytes=64 (transposed for column access)

**IMPORTANT**: colbytes is a u16 at byte offset 48+2*tile_id (little-endian).
For values ≤ 64, only the low byte matters.

---

## Gotcha 7: TILEZERO with wrong config = SEGFAULT

If you configure tile 0 as 16 rows × 64 colbytes but then TILEZERO tmm0,
it works. But if the config doesn't match what the hardware expects (e.g.,
palette=0 or all zeros), TILEZERO will SEGFAULT.

**Fix**: Always start with the minimal working config:
```rust
cfg.data[0] = 1;      // palette 1 (MUST be 1, not 0)
cfg.data[16] = 1;     // at least 1 row
cfg.data[48] = 4;     // at least 4 colbytes (1 × i32)
```

Then expand to full 16×64 after verifying the minimal config works.

---

## Gotcha 8: is_x86_feature_detected!("amx-tile") is NIGHTLY ONLY

```rust
// DOES NOT compile on stable:
is_x86_feature_detected!("amx-tile")  // error: unstable x86_amx_intrinsics

// WORKS on stable:
fn amx_available() -> bool {
    let cpuid = core::arch::x86_64::__cpuid_count(7, 0);
    let amx_tile = (cpuid.edx >> 24) & 1;
    let amx_int8 = (cpuid.edx >> 25) & 1;
    amx_tile == 1 && amx_int8 == 1
}
```

Use `__cpuid_count` (stable) for detection, not `is_x86_feature_detected!`.

---

## Hardware Tiers (this session)

```
Tier   Feature         MACs/instr  Detection (stable)                CPU
────   ───────         ──────────  ──────────────────                ───
3      AMX             256         __cpuid_count(7,0).edx bit 24     Sapphire Rapids+
2      avx512vnni      64          is_x86_feature_detected!          Cascade Lake+, Zen 4+
1      avxvnniint8     32          is_x86_feature_detected!          Arrow Lake (NUC 14)
0      scalar          1           always                            any
```

Also detectable but not yet kernelized:
- `avxvnniint16`: i16×i16 dot product (VPDPWSSD)
- `amx-bf16`: TDPBF16PS (BF16 tile matmul, for calibration)

---

## Files

```
ndarray/src/simd_amx.rs          — AMX detection + VNNI/VNNI2 kernels + quantize
ndarray/src/hpc/amx_matmul.rs    — AMX tile ops via inline asm (TDPBUSD)
ndarray/crates/burn/src/ops/matmul.rs — 4-tier dispatch in distance table builder
```

---

## What AMX Enables

```
Distance table build (4096² = 16M dot products):
  AMX:       ~20 min  (all models combined)
  avx512vnni: ~1:20h
  avxvnniint8: ~2:40h (NUC 14)
  scalar:    ~24-48h

ThinkingEngine MatVec (per cycle):
  AMX:       ~44 μs   (L1 table fits in 4 tile registers)
  avx512vnni: ~175 μs
  avxvnniint8: ~350 μs
  scalar:    ~5 ms
```
