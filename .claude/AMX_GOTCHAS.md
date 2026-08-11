# AMX Gotchas — Troubleshooting Playbook (Stable Rust 1.94)

> Updated: 2026-06-14 (corrected — the 2026-04-03 version shipped three of the
> bugs below: syscall 157, `TDPBUSD = …73…C1`, and the swapped TILECFG layout).
> Canonical reference: `.claude/knowledge/amx-enablement-and-kernel.md`.
> Owning agent: `.claude/agents/amx-savant.md`.
> Verified on: Emerald Rapids (CPUID model 0xCF), kernel 6.18.5. The fixes are
> ISA-level and apply equally to Sapphire Rapids (0x8F) and Granite Rapids.

This file is the *how-to-debug* companion. Each gotcha lists its **fault
signature** so you can map a crash to a cause without a debugger. The
instruction-bisector that produced these is `examples/amx_probe.rs` — run it
FIRST (it prints a flushed line before each tile op, so the last line names the
faulting instruction, then it checks correctness across shapes).

---

## Status (verified by actual execution, not by a skipped test)

```
LDTILECFG    ✓   TILEZERO ✓   TILELOADD ✓   TILESTORED ✓   TILERELEASE ✓
TDPBUSD      ✓   (u8×i8 → i32, bit-exact vs scalar)
TDPBF16PS    ✓   (bf16×bf16 → f32, within BF16 tolerance)
amx_available() = true on Emerald Rapids (cached LazyLock)
int8 2048³ = 169.7 GMAC/s, 600× scalar, single-thread
```

> ⚠ The previous "✓" marks were never executed: every AMX test early-returns
> `if !amx_available() { return; }`, and detection always returned false
> (Gotcha 4). Treat any "tile asm tested" claim as UNVERIFIED until you confirm
> `amx_available()` was `true` when the test ran. See Gotcha 9.

---

## Fault-signature → cause (the fast index)

| You see… | Almost certainly… | Go to |
|---|---|---|
| `amx_available()==false` on a Xeon you *know* has AMX | arch_prctl syscall number | Gotcha 4 |
| SIGSEGV at the very first tile op (`LDTILECFG`) | TILECFG rows/colsb swapped, or not 64B-aligned | Gotcha 6, 2 |
| SIGSEGV at `TILELOADD`/`TILESTORED` | SIB base/index swapped (stride deref'd as ptr) | Gotcha 10 |
| SIGILL at `TDPBUSD`/`TDPBF16PS` | ModRM aliases two tile operands (same-tile #UD) | Gotcha 11 |
| runs fine, `correct=false` | operand index/sign convention mirrored | Gotcha 12 |
| compile error `unstable x86_amx_intrinsics` | used nightly intrinsics | Gotcha 1, 8 |
| compile error `rbx is used internally by LLVM` | inline-asm CPUID | Gotcha 3 |
| exact when idle, silently wrong under CPU load (VM) | tile state lost across host vCPU switch | Gotcha 14 |

---

## Gotcha 1: Rust `_tile_*` intrinsics are NIGHTLY ONLY

```rust
use std::arch::x86_64::_tile_loadconfig;  // error: unstable feature x86_amx_intrinsics
```
**Fix**: inline `asm!` (stable since 1.59). LDTILECFG works as a mnemonic; the
tile ops need raw `.byte` (Gotcha 5). Tracking: rust-lang/rust#126622.

---

## Gotcha 2: Tile config MUST be 64-byte aligned

```rust
#[repr(C, align(64))]
struct TileConfig { data: [u8; 64] }
```
LDTILECFG reads 64 bytes; an unaligned pointer raises `#GP` → SIGSEGV.

---

## Gotcha 3: `rbx` is LLVM-reserved — don't inline-asm CPUID

Use `core::arch::x86_64::__cpuid_count(7, 0)` (stable, handles rbx). Inline
`asm!("cpuid", out("ebx") …)` fails to compile.

---

## Gotcha 4: enablement needs `arch_prctl` — syscall **158**, not `prctl` 157  ⚑ THE BIG ONE

AMX `XTILEDATA` is a *dynamically-enabled* XSTATE feature (Linux 5.16+). A
process must request permission before any tile op or the first one faults
(XFD `#NM`):

```
arch_prctl(ARCH_REQ_XCOMP_PERM /*0x1023*/, XFEATURE_XTILEDATA /*18*/)
```

`ARCH_REQ_XCOMP_PERM` is an **arch_prctl** op → **syscall 158**. Issuing it on
**prctl (157)** returns `-EINVAL`, so detection's gate 4 always failed and
`amx_available()` returned `false` on EVERY AMX host. **This file's previous
version literally documented `SYS_prctl = 157`** — that is where the bug came
from. Always 158.

Fault signature: `amx_available()==false` while `cpu_model().has_amx()==true`.

---

## Gotcha 5: tile ops need raw byte encoding (LDTILECFG is the exception)

See the authoritative table in the knowledge doc. The correct sequences:

```rust
// TILEZERO tmm0 / tmm1 / tmm2 / tmm3
asm!(".byte 0xc4,0xe2,0x7b,0x49,0xc0", options(nostack,nomem)); // tmm0
asm!(".byte 0xc4,0xe2,0x7b,0x49,0xc8", options(nostack,nomem)); // tmm1
asm!(".byte 0xc4,0xe2,0x7b,0x49,0xd0", options(nostack,nomem)); // tmm2
// TILERELEASE
asm!(".byte 0xc4,0xe2,0x78,0x49,0xc0", options(nostack,nomem));
// TILELOADD tmmN,[rcx+rax]  (SIB 0x01 = base=rcx,index=rax)
asm!(".byte 0xc4,0xe2,0x7b,0x4b,0x04,0x01", in("rcx") ptr, in("rax") stride, options(nostack)); // tmm0
// TILESTORED [rcx+rax],tmm0
asm!(".byte 0xc4,0xe2,0x7a,0x4b,0x04,0x01", in("rcx") ptr, in("rax") stride, options(nostack));
// TDPBUSD  tmm0,tmm1,tmm2   (u8 in rm/tmm2, i8 in vvvv/tmm1 — see Gotcha 12)
asm!(".byte 0xc4,0xe2,0x71,0x5e,0xc2", options(nostack,nomem));
// TDPBF16PS tmm0,tmm1,tmm2
asm!(".byte 0xc4,0xe2,0x72,0x5c,0xc2", options(nostack,nomem));
```

> ✗ The previous version listed `TDPBUSD … 0x73 … 0xc1` — `0x73` is TDPBSSD
> (wrong sign variant) and `0xc1` aliases tmm1 with itself (Gotcha 11).

---

## Gotcha 6: TILECFG field layout — colsb and rows are NOT where you'd guess  ⚑

Correct XTILECFG (Intel SDM):

```
byte 0      palette (=1)
byte 1      start_row (=0)
bytes 2-15  reserved (0)
bytes 16-47 colsb[t] : 16 × u16   →  colsb[t] at offset 16 + 2*t   (≤ 64)
bytes 48-63 rows[t]  : 16 × u8    →  rows[t]  at offset 48 + t      (≤ 16)
```

The previous version said "rows 16-23, colbytes 48-63" — **swapped**. With the
swap you get `colsb[0]=0x1010=4112` and `rows[0]=64`, both out of range, so
**LDTILECFG `#GP`-faults → SIGSEGV** the instant the AMX path runs. For the
16×16 int8/bf16 tile, every tile is 16 rows × 64 colbytes.

Fault signature: SIGSEGV at the first `LDTILECFG`.

---

## Gotcha 7: TILEZERO/LDTILECFG with palette=0 SEGFAULTs

Always `cfg.data[0] = 1`. Start from a minimal valid tile (1 row × 4 colbytes:
`data[16]=4; data[48]=1`) to confirm the config path before scaling to 16×64.

---

## Gotcha 8: `is_x86_feature_detected!("amx-tile")` is NIGHTLY ONLY

Use `__cpuid_count(7,0).edx` bits 24 (TILE) + 25 (INT8), then XGETBV(0) bits
17/18, then the arch_prctl (Gotcha 4). All stable. See `simd_amx::detect_amx`.

---

## Gotcha 9: "tests pass" can mean "tests skipped"

Every AMX test guards with `if !amx_available() { return; }`. While detection
was broken (Gotcha 4), 100% of them early-returned green without running a
single tile instruction. **A skipped test is not a passing test.** Validate AMX
with `examples/amx_probe` (unconditional) on real AMX silicon, and require a
`correct=`/parity assertion, not just "didn't crash."

---

## Gotcha 10: TILELOADD/TILESTORED SIB byte — base vs index

`TILELOADD tmm,[rcx+rax]` with regs bound `in("rcx") ptr, in("rax") stride`
needs SIB `0x01` = (scale=1, index=rax, base=rcx). The previous code used SIB
`0x08` = (index=rcx, base=rax), i.e. base/index swapped, so the tile engine
used the **stride value (~64) as the start address** → SIGSEGV. For TILELOADD
the *base* register is the data pointer and the *index* register is the row
stride in bytes.

Fault signature: SIGSEGV at the first `TILELOADD`.

---

## Gotcha 11: the three tile operands MUST be distinct registers

`TDPBUSD`/`TDPBF16PS` raise `#UD` (→ SIGILL) if any two of (dst, src1, src2)
name the same tile. ModRM `0xC1` = rm=tmm1, and `VEX.vvvv` was also tmm1 →
src1==src2 → same-tile `#UD`. Use ModRM `0xC2` (dst=tmm0, vvvv=tmm1, rm=tmm2).

Fault signature: SIGILL at the first `TDPBUSD`/`TDPBF16PS`, AFTER LDTILECFG and
the loads succeed.

---

## Gotcha 12: the operand index/sign convention is mirrored from the SDM  ⚑

Measured on EMR (selector probe + 4-opcode sign sweep — see the knowledge doc):

- `dst[m][n] = Σ_k tmm2(ModRM.rm)[m][k] · tmm1(VEX.vvvv)[k][n]` — plain **M×K**
  goes in **tmm2/rm**, VNNI **K×N** goes in **tmm1/vvvv** (mirror of the naive
  SDM operand order).
- For `TDPBUSD` (0x71): **rm = unsigned, vvvv = signed**.

So the kernel loads `A(u8)→tmm2`, `B_vnni(i8)→tmm1`. Get this wrong and it
runs cleanly but every value is wrong (often a suspiciously *clean* wrong, like
`total/16` for constant inputs — that uniformity is the tell). Isolate it with
the selector probe (`A[0][s]=1` → `C[0][:]` should equal `B[s][:]`).

Fault signature: no crash, `correct=false`.

---

## Gotcha 13: cache detection in a `LazyLock` (don't re-syscall per call)

`amx_available()` runs CPUID + XGETBV + arch_prctl. Calling it per matmul is
wasteful (and the arch_prctl, though idempotent, is a syscall). Cache it:

```rust
static AMX_AVAILABLE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(detect_amx);
pub fn amx_available() -> bool { *AMX_AVAILABLE }
```

All four gates are non-blocking (no I/O, no lock, no spin) so the init can't
stall. The arch_prctl grant is process-wide + inherited by all threads, so
once is correct even under rayon. `cpu_model()` is cached the same way.

---

## Gotcha 14: on oversubscribed VMs, tile state is silently corrupted under host CPU contention  ⚑

Observed 2026-07-02 on this remote VM (4 vCPU, EMR-class Xeon, guest kernel
6.18.5) by `examples/onebrc_cascade_probe.rs`, reproduced on demand:

```
idle:                       413/413 stations bit-exact (10M and 100M rows)
4 busy-loop competitors:    89/413, 152/413 exact — whole rows LOST, no fault
probe pinned to core 0,
load pinned to cores 1-3:   124/413 exact — pinning does NOT mitigate
idle control right after:   413/413 exact again
```

Signature: **no crash, no SIGSEGV/SIGILL — results are silently wrong, and
only under load.** An AVX-512 path in the same process, same run, stays
bit-exact, isolating the corruption to TMM tile state (the tmm0 accumulator
loses in-flight partial sums). Because guest-side pinning doesn't help, the
suspected mechanism is the **host** hypervisor's vCPU context switch failing
to save/restore guest `XTILEDATA` when the host multiplexes oversubscribed
pCPUs (idle guests keep their vCPUs resident → no corruption; loaded guests
get switched → corruption). Guest-side `arch_prctl` permission (Gotcha 4) is
correctly granted — this is a layer below the guest kernel.

Consequences:

- **Never certify AMX numerics from a shared/oversubscribed VM.** Bare metal
  or a dedicated-CPU instance only. A "PASS on my cloud box" is worthless
  under this gotcha unless the box was provably idle.
- **Extend Gotcha 9's discipline**: a parity test for a tile kernel must ALSO
  run under deliberate CPU contention (a few busy loops are enough — see the
  reproduction above). Exact-when-idle is necessary, not sufficient.
- **Keep tile residency short.** Long accumulation loops that live in tmm
  across many iterations (the 16×16×K pattern holds tmm0 for K/32 iterations)
  maximize the exposure window. Draining accumulators to memory more often
  shrinks it but does NOT close it — treat it as harm reduction, not a fix.
- Production dispatch on virtualized hosts should either avoid AMX or pair it
  with a checksum/parity channel (e.g. a redundant ones-row whose expected
  value is known — the onebrc probe's count row doubles as exactly that).

Fault signature: `correct=true` in every quiet test, sporadic wrong results
in production under load, AVX-512 siblings unaffected.

---

## Gotcha 15: a guest can DENY AMX outright — environment gate, NOT a code/ISA bug

> Cross-check breadcrumb (2026-06-20). Buried mid-doc on purpose (away from the
> head/tail a skim touches) so NO session concludes "AMX is broken" from one
> `amx_available()==false`. ISA fixes 1-13 above are correct and verified.

Some virtualised guests refuse AMX even when every ISA fix above is right. To
tell "AMX off by provisioning" from "our code is wrong", probe all FOUR gates
directly (not just CPUID) — each a few lines of stable asm (Gotcha 4/5 has them):

- `__cpuid_count(7,0).edx` b24/b25/b22 (TILE/INT8/BF16) — hypervisor may MASK to
  0 (a generic model name like `Xeon @ 2.80GHz` is the tell).
- `_xgetbv(0)` XCR0 b17/b18 (TILECFG/TILEDATA) — if **0**, the OS never enabled
  tile XSTATE; AMX cannot run, full stop.
- `syscall(SYS_arch_prctl /* 158 */, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA=18)`
  — `-95 / -EOPNOTSUPP`
  means the kernel REFUSES the grant; **no byte-call can force it**.
- `syscall(SYS_arch_prctl /* 158 */, ARCH_GET_XCOMP_PERM, &mask)` — XTILEDATA (b18)
  must be in mask.

Observed once on one provisioning (kernel 6.18.5 guest): CPUID=0 **and** XCR0
b17/b18=0 **and** REQ=`-EOPNOTSUPP` → genuinely unavailable. `detect_amx()`
correctly returns `false`; consumers run the AVX-512 / `F32x16` fallback (correct,
NaN-clean). The SAME binary prints `[AMX TDPBF16PS]` the moment it lands on a
guest where `arch_prctl` returns 0 and XCR0 bits are set (e.g. the Emerald Rapids
host this file's header was verified on). **`false` here ≠ broken — it's an
environment gate above the code.** Full write-up + a ~25-line four-gate probe:
lance-graph `.claude/board/EPIPHANIES.md` → `E-DOMINO-SOA-ORCHESTRATION-GREEN`.
---

## Hardware tiers

```
Tier  Feature      MACs/instr  Detect (stable)                   CPU
3     AMX-TILE     16384       __cpuid_count(7,0).edx bit24+25   SPR / EMR / GNR (NOT Sierra Forest)
2     avx512vnni   64          is_x86_feature_detected!          Cascade Lake+, Zen 4+
1     avxvnniint8  32          is_x86_feature_detected!          Arrow / Meteor Lake
0     scalar       1           always                            any
```

`cpu_model()` returns `SierraForest` for model 0xAF — E-core silicon with NO
AMX, so `has_amx()` is false there even though it's a recent Xeon.

---

## Files

```
src/simd_amx.rs                 — detection (CPUID+XGETBV+arch_prctl), CpuModel, LazyLock
src/hpc/amx_matmul.rs           — tile primitives + TileConfig + public matmul_{i8_to_i32,bf16_to_f32,f32}
src/hpc/int8_tile_gemm.rs       — fast int8 driver (LDTILECFG hoisted) + 16×16 kernel
src/hpc/bf16_tile_gemm.rs       — bf16 sibling
examples/amx_probe.rs           — instruction bisector + correctness validator (run FIRST)
examples/amx_gemm_bench.rs      — throughput + independent correctness check
```
