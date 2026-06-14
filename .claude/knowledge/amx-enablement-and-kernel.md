# AMX Enablement & Tile-Kernel Reference

> READ BY: amx-savant, savant-architect, sentinel-qa, simd-savant
> Status: AMX ENABLED + bit-exact + fast on Emerald Rapids (2026-06-14).
> Supersedes the buggy claims in the original `.claude/AMX_GOTCHAS.md`
> (that doc has been corrected; this file is the canonical reference, the
> gotchas file is the troubleshooting playbook, `amx-savant` is the agent).

This is the "teach to fish" file: not just *how* to turn AMX on, but *why*
it was off, how every caveat manifests, and how to troubleshoot each one.

---

## TL;DR — current truth

AMX (Intel Advanced Matrix Extensions) runs on **stable Rust 1.94** via inline
`asm!` byte-encodings (the Rust `_tile_*` intrinsics are nightly-only, issue
#126622). As of 2026-06-14 it is **enabled, bit-exact, and fast** here:

| Surface | State |
|---|---|
| `ndarray::simd::amx_available()` | `true` on Emerald Rapids (cached `LazyLock`) |
| `ndarray::simd::cpu_model()` | `EmeraldRapids` (CPUID model 0xCF) |
| `matmul_i8_to_i32` (TDPBUSD) | bit-exact vs scalar, all shapes |
| `matmul_f32` / `matmul_bf16_to_f32` (TDPBF16PS) | within BF16 tol (rel-err ~0.004) |
| int8 GEMM 2048³, single-thread, `target-cpu=native` | **169.7 GMAC/s (339 GOP/s), 600× scalar** |

Files:
- `src/simd_amx.rs` — detection (CPUID + XGETBV + arch_prctl), `CpuModel`, `LazyLock` cache.
- `src/hpc/amx_matmul.rs` — tile primitives (`tile_loadconfig`/`tile_load`/`tile_store`/`tile_dpbusd`/`tile_dpbf16ps`/`tile_zero`/`tile_release`/`vnni_pack_*`/`TileConfig`).
- `src/hpc/int8_tile_gemm.rs` — `int8_gemm_amx_tiled` (the fast driver) + `int8_tile_gemm_16x16`.
- `src/hpc/bf16_tile_gemm.rs` — bf16 sibling.
- `examples/amx_probe.rs` — the validator / instruction-bisector (run this FIRST when debugging).
- `examples/amx_gemm_bench.rs` — throughput.

---

## 1. The one-line enablement bug — the "special way to enable it"

Linux 5.16+ makes AMX `XTILEDATA` a **dynamically-enabled XSTATE feature**:
a process must *request permission* before any tile op, or the first tile
instruction faults (XFD `#NM`). The request is:

```
arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)   // 0x1023, 18
```

`ARCH_REQ_XCOMP_PERM` (0x1023) is an **`arch_prctl`** op → **syscall 158**.
The code issued it on **`prctl` → syscall 157**, which rejects option 0x1023
with `-EINVAL`. So gate 4 of detection *always failed* → `amx_available()`
returned `false` on **every AMX-capable host**, and the AMX path was dead code
that never ran. Fix: `157 → 158`. That single digit is the whole "AMX is
available 50-79% of the time but needs a special way to enable it."

Why ~50-79%: Claude's fleet is heterogeneous — a container lands on AMX
silicon (SPR/EMR/GNR) only some of the time. On those hosts the gate-4 bug
made AMX look absent; on non-AMX hosts gate 1 (CPUID) correctly returns false.

---

## 2. SPR vs EMR — *not* the cause

The original `AMX_GOTCHAS.md` header says "CPU: Sapphire Rapids … AMX
confirmed." That confirmation was hollow: every AMX unit test early-returns
`if !amx_available() { return; }`, and `amx_available()` was always false
(the 157 bug), so **the tile asm had literally never executed on SPR either.**
The "✓ TDPBUSD works" checkmarks were CPUID detection + aspiration, not a run.

Consequence: the five bugs below are **ISA / encoding bugs, identical on SPR,
EMR, and GNR**. EMR was simply the host where gate 4 got fixed first, so it was
the first host to actually *execute* the tile path and expose them. The fixes
are silicon-independent; they apply equally to Sapphire Rapids and Granite
Rapids. The operand convention (§4) is a property of the VEX encoding, not the
microarchitecture, so it holds across all AMX CPUs.

`cpu_model()` exists so a run can *say* which silicon it landed on and tell
"no AMX silicon" apart from "AMX present but not OS-enabled" — but no code path
should branch kernel correctness on SPR-vs-EMR. They are the same ISA here.

---

## 3. The five bugs, by fault signature (the troubleshooting spine)

Each AMX bug has a *distinct* crash/■ signature. Memorize the mapping —
it's how you bisect without a debugger (the `amx_probe` example prints a
flushed line before each instruction so the LAST line names the fault):

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `amx_available()==false` on AMX silicon | `arch_prctl` on syscall **157** not **158** → `-EINVAL` | use 158 |
| 2 | **SIGSEGV** at `LDTILECFG` (first tile op) | `TileConfig` rows/colsb regions **swapped** → colsb=4112, rows=64 → `#GP` | colsb u16 @16+2t, rows u8 @48+t |
| 3 | **SIGSEGV** at `TILELOADD`/`TILESTORED` | SIB `0x08` = `[base=rax,index=rcx]` but regs bound rcx=ptr,rax=stride → derefs stride(~64) as base | SIB `0x01` = `[base=rcx,index=rax]` |
| 4 | **SIGILL** at `TDPBUSD`/`TDPBF16PS` | ModRM `0xC1` ⇒ rm=tmm1 == vvvv=tmm1 → two sources alias → same-tile `#UD` | ModRM `0xC2` (rm=tmm2, distinct) |
| 5 | runs, **wrong values** (`correct=false`) | operand index+sign convention mirrored from naive SDM reading | load plain M×K→tmm2, VNNI K×N→tmm1; TDPBUSD `0x71` |

The order matters: each fix exposes the next signature (SIGSEGV → SIGSEGV →
SIGILL → wrong-values → correct). If you fix #2 and still SIGSEGV, you're on
#3; if you clear both segfaults and hit SIGILL, you're on #4.

---

## 4. The empirical operand convention (the "alien magic")

**The AMX tile-op operand mapping on this silicon is the mirror of the naive
Intel-SDM reading, on BOTH axes.** Verified by driving the tile primitives with
a selector A (`A[0][s]=1`) and a marked B, then sweeping all four `TDPB**D`
opcodes against sign-sensitive constant inputs:

- **INDEX**: `dst[m][n] = Σ_k  tmm2(ModRM.rm)[m][k] · tmm1(VEX.vvvv)[k][n]`.
  The **plain M×K** operand goes in **tmm2 (rm)**; the **VNNI-packed K×N**
  operand goes in **tmm1 (vvvv)**. (Naive SDM order would say the opposite.)
- **SIGN** (for opcode `0x71`): **tmm2/rm = UNSIGNED, tmm1/vvvv = SIGNED.**

Sign sweep (loads: B_vnni→tmm1, A→tmm2; A=200=u8 200/i8 -56, B=3 / A=3, B=200):

| byte2 | mnemonic (pp) | A(rm) | B(vvvv) |
|---|---|---|---|
| 0x70 | TDPBUUD (NP) | unsigned | unsigned |
| **0x71** | **TDPBUSD (66)** | **unsigned** | **signed** ← `u8×i8` |
| 0x72 | TDPBSUD (F3) | signed | unsigned |
| 0x73 | TDPBSSD (F2) | signed | signed |

`0x70` and `0x73` (both-same-sign) match the SDM directly, which confirms the
opcode→pp map is right and only the src1/src2 ↔ vvvv/rm *position* is mirrored.

Kernel consequence: `int8_tile_gemm::amx_path` loads `A(u8)→tmm2`, `B_vnni(i8)
→tmm1`, executes `tile_dpbusd` (0x71). The `matmul_i8_to_i32` caller keeps its
`A+128→u8` shift and `−128·colsum(B)` bias unchanged — correct because A(rm)
is the unsigned operand. bf16 has no sign split, so the index swap alone fixes
TDPBF16PS.

---

## 5. Validated byte-code table (authoritative — measured on EMR)

> There is **no** W3C/intrinsics export with these AMX encodings — the
> `.claude` "w3c" files are semantic-web ontologies (SKOS/PROV-O/FIBO). The
> authority is the Intel SDM opcode map + the empirical sweep above. The
> following are confirmed correct in-tree:

```
LDTILECFG [mem]              : "ldtilecfg [{}]"  (mnemonic; assembler encodes it)
TILEZERO  tmmN              : C4 E2 7B 49 (C0 | N<<3)     # tmm0=C0 tmm1=C8 tmm2=D0 tmm3=D8
TILERELEASE                 : C4 E2 78 49 C0
TILELOADD tmmN,[rcx+rax]    : C4 E2 7B 4B (04 | N<<3) 01  # SIB 01 = base=rcx,index=rax,scale1
TILESTORED [rcx+rax],tmm0   : C4 E2 7A 4B 04 01
TDPBUSD   tmm0,tmm1,tmm2    : C4 E2 71 5E C2              # pp=66 ; dst=tmm0,vvvv=tmm1,rm=tmm2
TDPBF16PS tmm0,tmm1,tmm2    : C4 E2 72 5C C2              # pp=F3 opcode 5C
```

VEX byte2 = `W(1) . vvvv(4) . L(1) . pp(2)`; `vvvv` is the 1's-complement of
the register (tmm1 → 1110). ModRM `0xC2` = mod=11, reg=000(tmm0), rm=010(tmm2).
The two earlier WRONG encodings that shipped from the SPR-era gotchas doc were
`TDPBUSD = C4 E2 73 5E C1` (0x73=TDPBSSD wrong variant; C1 aliases tmm1).

### TILECFG (XTILECFG) 64-byte layout — the corrected version

```
byte 0      : palette (MUST be 1)
byte 1      : start_row (0)
bytes 2-15  : reserved (0)
bytes 16-47 : colsb[t] — 16 × u16, colsb[t] @ (16 + 2*t)   # bytes-per-row, ≤ 64
bytes 48-63 : rows[t]  — 16 × u8,  rows[t]  @ (48 + t)      # rows,          ≤ 16
```

The SPR-era doc had rows and colsb **swapped** ("rows 16-23, colbytes 48-63"),
which is bug #2. For the 16×16 int8/bf16 tile, all three tiles are 16 rows ×
64 colbytes.

---

## 6. Detection API (cached, CPU-aware)

```rust
use ndarray::simd::{amx_available, cpu_model, amx_report, CpuModel};

amx_available()  // bool, cached once via LazyLock (the 4 gates of §1)
cpu_model()      // CpuModel::{SapphireRapids,EmeraldRapids,GraniteRapids,SierraForest,OtherX86,NonX86}
cpu_model().has_amx()   // true for SPR/EMR/GNR; false for Sierra Forest (E-core)
amx_report()     // e.g. "AMX [Emerald Rapids expects_amx=true]: TILE=true INT8=true BF16=true available=true"
```

Why `LazyLock`: the four gates (CPUID, XGETBV, one `arch_prctl`) are all
non-blocking — no I/O, no lock contention, no spin — so the init cannot stall;
it runs once on first call and every later call is a cached load. The
`arch_prctl` grant is **process-wide and inherited by all threads**, so
requesting it exactly once is correct even under a rayon consumer. Diagnostic
value: `cpu_model().has_amx() == true && amx_available() == false` means the
silicon has AMX but the OS/hypervisor hasn't enabled it (XCR0 clear, or — until
the 157→158 fix — the permission request failed). That split is the single most
useful troubleshooting signal.

---

## 7. Performance — what made it fast

Correct ≠ fast. The first correct version was **14.8 GMAC/s** (~0.7% of peak)
because `int8_gemm_amx_tiled` called the 16×16 kernel per output tile, which ran
`LDTILECFG` (a **serializing** instruction) + `TILERELEASE` and re-VNNI-packed
B **on every tile** (256 `LDTILECFG`s for a 256² output). The fast driver:

1. `LDTILECFG` **once** up front, `TILERELEASE` **once** at the end.
2. VNNI-pack each B column band **once per j-tile** (reused across all row tiles).
3. `TILEZERO` the C accumulator and `TILESTORED` the 16×16 result **straight
   into its strided slot** in C (row pitch n·4 bytes) — no scratch + copy.

Result: **14.8 → 169.7 GMAC/s** (11.5×), still correct=true.

**Then 2×2 register blocking** (`int8_gemm_amx_tiled_rb` + `tile_dpbusd_2x2`):
4 C accumulators (tmm0-3) fed by 2 A tiles (tmm4-5) + 2 B tiles (tmm6-7), so
each A/B tile load serves TWO products — half the tile loads per MAC, the right
lever for this **memory-bandwidth-bound** kernel. The loop order matters: a
first cut pre-packed ALL of B (~4 MB at 2048²) and thrashed cache, *regressing*
large shapes (1024³ 156→125). The BLIS-style fix — OUTER over 32-col panels,
pack only that panel's 2 B bands (L2-resident) and reuse across all row-blocks,
INNER over 32-row blocks — also halves A's DRAM re-reads (32-col vs 16-col
panels). Single-thread result, all correct=true:

```
        serial    rb(2×2)
256³    65.7      80.8  (+23%)
512³   124.9     132.0  (+6%)
1024³  155.9     170.2  (+9%)
2048³  169.7     197.7  (+16%)   ← 395 GOP/s
```

**Rayon over row-tiles** (`int8_gemm_amx_tiled_par`, `feature = "rayon"`): this
kernel is bandwidth-bound, so 4-core scaling is sublinear — 2048³ → 237.5 GMAC/s
(~1.4×) — and it REGRESSES small/medium (thread + B-prepack overhead), so it's
gated to `m·n·k ≥ 2e9`. Many-core servers gain more.

Dispatch (in `int8_gemm_amx_tiled`): huge + rayon → `_par` (16×16, shared
pre-packed B); else m,n≥32 → `_rb` (2×2); else `_serial` (16×16); m or n < 32
strips fall to the 16×16 path inside `_rb`.

**Remaining headroom (with a caution).** "rayon-over-rb" — fanning the rb
row-panels across the pool instead of the 16×16 kernel — is the obvious combine,
BUT a first attempt (each rayon task calls `_rb` on a 64-row band) was REVERTED:
it ran SLOWER than rb-single (155 vs 197 GMAC/s at 2048³ — each task re-VNNI-packs
B, an O(K·N) duplicate ×num_tasks) AND `correct=false` appeared at 1024³/2048³
while 256³/512³ stayed correct (an AMX-tiles-under-rayon-at-scale issue not yet
diagnosed — single-thread `_rb` is bit-exact at every size, so it's specific to
the threaded 8-tile path). Do NOT reship that shape without (a) a SHARED pre-pack
of B (as `_par` 16×16 already does) and (b) a probe that reproduces the
large-size correctness failure under rayon and explains it. The safe wins are
banked: rb-single (197) is the default, 16×16-rayon (237) the huge case. The
bigger lever is full BLIS Mc/Nc/Kc cache blocking.

---

## 8. Modus operandi when AMX "doesn't work"

1. `amx_report()` first. `expects_amx` false → not AMX silicon (or non-Intel /
   masked); stop, use the VNNI/scalar fallback. `expects_amx` true but
   `available=false` → enablement, not silicon: check kernel ≥5.16, XCR0 bits
   17/18 (`XGETBV(0)`), and that gate 4 uses syscall **158**.
2. If `available=true` but a GEMM crashes/misbehaves, run `examples/amx_probe`
   (it bisects instruction-by-instruction and then checks correctness across
   shapes). Match the fault to the §3 table.
3. Never trust "tile asm tested" claims that sit behind an
   `if !amx_available() { return; }` guard — confirm the guard was *true* when
   the test ran (i.e. on real AMX silicon with detection fixed).
4. Validate with `amx_probe` (correctness) **and** `amx_gemm_bench`
   (throughput + an independent `correct=` check) before believing numbers.

---

## References

- `.claude/AMX_GOTCHAS.md` — the per-caveat troubleshooting playbook.
- `.claude/agents/amx-savant.md` — the agent that owns this surface.
- `.claude/knowledge/hardware_map.md`, `agnostic-surface-cpu-matrix.md` — CPU tiers.
- Intel SDM Vol 2 (LDTILECFG / TILELOADD / TDPBUSD / TDPBF16PS), Vol 1 §13 (XSAVE/XFD).
- Linux `Documentation/arch/x86/xstate.rst` (ARCH_REQ_XCOMP_PERM, dynamic XSTATE).
