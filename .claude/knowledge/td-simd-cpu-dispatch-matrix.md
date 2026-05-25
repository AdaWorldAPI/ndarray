# Per-CPU SIMD Dispatch Matrix

> **Companion to:** `td-simd-tier-audit.md` (the debt inventory) and `td-simd-integration-plan.md` (the `SimdProfile` architecture). This document is the **authoritative feature table** that the `SimdProfile::detect()` function in Phase 3 of the integration plan will be implemented against. Every cell is sourced from official spec or recognized secondary source; nothing is verified on hardware yet.

## Status legend

Per cell:

- **DOC** — feature presence stated in official Intel/AMD documentation, AVX-512 Wikipedia table, or WikiChip microarchitecture page. Considered safe to dispatch but UNVERIFIED on physical silicon by this project.
- **TEST** — feature verified by running an instruction on real hardware of this CPU model and observing correct behavior. Promoted from DOC after first hardware test.
- **✗** — feature explicitly absent per source.
- **(N/A)** — feature category doesn't apply to this architecture (e.g. AMX on aarch64).

**Every cell in the table below is currently `DOC`.** No hardware verification has been performed for this project. The matrix is safe to dispatch from at the silicon-spec level; promotion to TEST happens as each profile gets first hardware verification.

## Provenance

Sources used to populate this matrix (per CPU below the table):

- Intel Architecture Instruction Set Extensions Programming Reference (319433-024)
- Intel ARK product specifications pages
- AMD EPYC 9004 / 9005 series data sheets
- Wikipedia AVX-512 article (cross-referenced against primary sources)
- WikiChip microarchitecture pages (cross-referenced against primary sources)
- AVX-512 BF16 / VNNI / VBMI extension articles on WikiChip
- Hot Chips 2020 (Ice Lake-SP), Hot Chips 2023 (Zen 4 / Genoa) presentations
- ServeTheHome / Phoronix / AnandTech hardware reviews for tested-in-the-wild confirmation

---

## Master matrix — x86_64

Rows ordered by silicon generation. Columns grouped: baseline → AVX-512 sub-features → AMX → AVX-VNNI (non-512). Every cell is `DOC` unless marked otherwise.

| CPU profile | F | CD | VL | DQ | BW | IFMA | VBMI | VBMI2 | VNNI | BF16 | FP16 | VPOPCNTDQ | BITALG | GFNI | VAES | VPCLMUL | VP2INT | AMX-TILE | AMX-INT8 | AMX-BF16 | AMX-FP16 | AVX-VNNI | AVX-VNNI-INT8 | AVX-IFMA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SkylakeX** | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **CascadeLake** | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **CooperLake** | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **IceLakeSp** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **TigerLakeU** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **SapphireRapids** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | DOC | DOC | DOC | ✗ | DOC | ✗ | ✗ |
| **EmeraldRapids** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | DOC | DOC | DOC | ✗ | DOC | ✗ | ✗ |
| **GraniteRapids** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ |
| **Zen4** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | DOC | ✗ | ✗ |
| **Zen5** | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | DOC | ✗ | ✗ |
| **ArrowLake** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | DOC | DOC | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | DOC | DOC | DOC |
| **HaswellAvx2** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

Reads: every `DOC` cell means "documented in an official spec, dispatch is safe; will be promoted to TEST after first hardware verification of this profile by this project."

### Microarchitecture notes (alphabetical by codename)

#### CascadeLake (Intel Xeon Scalable 2nd gen, 14nm, 2019)

Codename: Cascade Lake-SP / -AP. ARK lineup: Xeon Platinum 8200 series, Gold 6200 / 5200 series.

- AVX-512: **F, CD, VL, DQ, BW, VNNI** — VNNI was *added* in Cascade Lake on top of the Skylake-X baseline; this is the headline DL Boost feature for inference.
- No IFMA, no VBMI, no BF16, no FP16, no AMX, no VPOPCNTDQ, no BITALG, no GFNI/VAES/VPCLMULQDQ in the 512-bit form.
- Two 512-bit FMA units (per high-bin SKUs) on Port 0 + Port 5; Port 1 supplies an AVX-512 add only (no second FMA). VNNI shares logic with FMA at Ports 0/1.
- Source: WikiChip Cascade Lake page; Microway "Detailed Specifications of the Cascade Lake SP" review; Intel ARK; Wikipedia AVX-512 table.

#### CooperLake (Intel Xeon Scalable "3rd gen" 4S/8S, 14nm refresh, 2020)

Codename: Cooper Lake-SP. ARK lineup: Xeon Platinum 8380H / 8380HL / 8376H / 8376HL / 8354H / 8353H, 4 or 8-socket only. Branded as "3rd gen Xeon Scalable" alongside Ice Lake-SP, but a different microarchitecture (Skylake-derived 14nm vs Ice Lake's Sunny Cove 10nm).

- AVX-512: **F, CD, VL, DQ, BW, VNNI, BF16** — *first* CPU with `avx512_bf16` (`VDPBF16PS`, `VCVTNE2PS2BF16`, `VCVTNEPS2BF16`).
- No IFMA, no VBMI, no FP16, no AMX, no VPOPCNTDQ/BITALG/GFNI/VAES/VPCLMULQDQ.
- Critical for our matrix: Cooper Lake has BF16 *without* VBMI. ICX has VBMI *without* BF16. This is the precise reason the `Tier::Avx512` collapse in audit TD-T12 is wrong — these are two different runtime dispatches that must be distinguished.
- Source: WikiChip Cooper Lake page; Intel Spec Update doc 634897; Intel Deep Learning Boost article on bfloat16.

#### EmeraldRapids (Intel Xeon Scalable 5th gen, Intel 7, 2023)

Codename: Emerald Rapids. ARK lineup: Xeon Platinum 8500 / 8600 series.

- ISA identical to Sapphire Rapids — same Raptor Cove P-core, refined process node, more L3, no new SIMD/AMX bits.
- **Dispatch handled by the SapphireRapids profile** (no separate `EmeraldRapids` variant needed in `SimdProfile` — the runtime kernels are byte-identical to SPR).
- Source: Wikipedia Emerald Rapids; Phoronix EMR review.

#### GraniteRapids (Intel Xeon 6, Redwood Cove P-core, Intel 3, 2024)

Codename: Granite Rapids-AP / -SP. ARK lineup: Xeon 6900P series (128 P-cores), Xeon 6700P, Xeon 6500P.

- AVX-512: same superset as Sapphire Rapids (F + CD + VL + DQ + BW + IFMA + VBMI + VBMI2 + VNNI + BF16 + FP16 + VPOPCNTDQ + BITALG + GFNI + VAES + VPCLMULQDQ).
- AMX: SPR superset PLUS **AMX-FP16** (`TDPFP16PS` — FP16 tile dot product). Future GNR-D may also ship AMX-COMPLEX (TDPFP16PS-style for complex multiply).
- Capable of 1024 BF16/FP16 + 2048 INT8 FLOPS per core per cycle via AMX (vs 1024 INT8 on SPR).
- Detection: AMX-FP16 lives at CPUID.07H.1H:EAX bit 21 — separate from the AMX-INT8/BF16 bits at leaf 7,0. The current `simd_caps.rs::detect()` does NOT check leaf 7,1 — needs to be added before GNR dispatch lands (per integration plan risk #4).
- Source: Intel Xeon 6 Granite Rapids product brief PDF; Tom's Hardware Granite Rapids launch article; Phoronix "Massive AI Performance Benefit With AMX On Intel Xeon 6"; Wikipedia Granite Rapids.

#### IceLakeSp (Intel Xeon Scalable "3rd gen" 1S/2S, 10nm, 2021)

Codename: Ice Lake-SP. ARK lineup: Xeon Platinum 8300 series, Gold 6300 / 5300 series. Sunny Cove P-core. **Different microarchitecture from Cooper Lake despite both being branded "3rd gen Xeon Scalable."**

- AVX-512: **F, CD, VL, DQ, BW, IFMA, VBMI, VBMI2, VNNI, VPOPCNTDQ, BITALG, GFNI, VAES, VPCLMULQDQ.**
- No BF16, no FP16, no AMX. No VP2INTERSECT (that's Tiger Lake mobile only).
- Source: WikiChip Fuse "Intel Launches 3rd Gen Ice Lake Xeon Scalable"; Hot Chips 2020 ICX-SP presentation; Microway "Detailed Specifications of the Ice Lake SP."

#### SapphireRapids (Intel Xeon Scalable 4th gen, Intel 7, 2023)

Codename: Sapphire Rapids-SP (-MCC, -XCC) and Sapphire Rapids HBM (Xeon Max). ARK lineup: Xeon Platinum 8400 series, Xeon Max 9400 series. Golden Cove P-core.

- AVX-512: **F, CD, VL, DQ, BW, IFMA, VBMI, VBMI2, VNNI, BF16, FP16, VPOPCNTDQ, BITALG, GFNI, VAES, VPCLMULQDQ.** Adds FP16 and the SPR-class AMX on top of ICX's superset.
- AMX: **AMX-TILE, AMX-INT8, AMX-BF16.** No AMX-FP16 (GNR only).
- AVX-VNNI (256-bit non-AVX-512 VNNI on ymm registers): present alongside AVX-512-VNNI. Same VPDPBUSD opcode, different EVEX/VEX encoding.
- Sapphire Rapids HBM (Xeon Max) has **identical ISA** to standard SPR — only memory differs (64 GB HBM2e on-package). One `SimdProfile::SapphireRapids` covers both. Caps differ on max sockets (Max scales to 2S, standard to 8S) and max cores (Max caps at 56, standard at 60) but that doesn't affect dispatch.
- Source: Wikipedia Sapphire Rapids; Intel ARK 8460Y+ / 9480 spec sheets; WikiChip Fuse AMX intro article; Tuning Guide for AI on 4th Generation Intel Xeon Scalable Processors; ServeTheHome "Intel Xeon Max CPU is the Sapphire Rapids HBM Line."

#### SkylakeX (Intel Xeon Scalable 1st gen + Core X-series + Xeon W, 14nm, 2017)

Codename: Skylake-SP / -X / -W. ARK lineup: Xeon Platinum/Gold/Silver/Bronze 8100/6100/5100/4100/3100, Core i9-7000X series, Xeon W-2100/W-3100.

- AVX-512: **F, CD, VL, DQ, BW.** This is the founding AVX-512 baseline; everything since adds on top.
- No VNNI (Cascade Lake added it), no IFMA, no VBMI, no BF16, no FP16, no AMX, no VPOPCNTDQ, no BITALG, no GFNI/VAES/VPCLMULQDQ.
- 4FMAPS / 4VNNIW / ERI / PFI subsets were spec'd on Knights Mill / Knights Landing only (Xeon Phi), never on mainline Skylake-X — out of scope for this matrix.
- Two FMA units on high-bin SKUs (Platinum/Gold); one FMA unit on Silver/Bronze. Either way, AVX-512F + CD + VL + DQ + BW is the full ISA-relevant feature set.
- Source: WikiChip Skylake (server) page; Intel ARK Xeon Platinum 8180 spec; Wikipedia AVX-512 table.

#### TigerLakeU (Intel Core 11th gen mobile, 10nm SuperFin, 2020-2021)

Codename: Tiger Lake-U / -H. ARK lineup: Core i7-1185G7, i7-11800H, etc. Willow Cove P-core.

- AVX-512: **F, CD, VL, DQ, BW, IFMA, VBMI, VBMI2, VNNI, VPOPCNTDQ, BITALG, GFNI, VAES, VPCLMULQDQ, VP2INTERSECT.**
- No BF16, no FP16, no AMX.
- Notable: only consumer-class Intel CPU to ship with the full Ice Lake-class AVX-512 ISA *plus* VP2INTERSECT. Rocket Lake (11th gen desktop) lacks VP2INTERSECT. Alder Lake (12th gen) onwards has AVX-512 disabled in firmware regardless of silicon presence — outside of this matrix.
- Source: AnandTech Tiger Lake review; WikiChip Tiger Lake page; Wikipedia AVX-512 table.

#### ArrowLake (Intel Core Ultra Series 2, 15th gen consumer, TSMC N3B P-cores, 2024)

Codename: Arrow Lake-S / -H / -HX. ARK lineup: Core Ultra 9 285K, Core Ultra 7 265K, etc. Lion Cove P-cores + Skymont E-cores.

- **No AVX-512** at all (hybrid CPU design; AVX-512 was dropped from consumer Intel post-Tiger Lake).
- AVX2 + FMA + F16C present (baseline).
- **AVX-VNNI** (256-bit, ymm-VEX-encoded `VPDPBUSD`), **AVX-VNNI-INT8** (signed-signed and unsigned-unsigned variants), **AVX-IFMA** (256-bit integer FMA), **AVX-NE-CONVERT** (BF16/FP16 conversion — *conversion only, not arithmetic*).
- GFNI / VAES / VPCLMULQDQ present in their 256-bit / VEX forms.
- The workspace's `simd_amx.rs::vnni2_dot_u8_i8` (256-bit `_mm256_dpbusd_epi32`) is the Arrow Lake path; correctly dispatched via `is_x86_feature_detected!("avxvnniint8")` at `matvec_dispatch` line 291.
- Source: Wikipedia Arrow Lake; Intel ARK Core Ultra 285K spec; Intel ISA Extensions Programming Reference for AVX-VNNI-INT8 / AVX-IFMA / AVX-NE-CONVERT.

#### HaswellAvx2 (catch-all: Haswell through Coffee Lake / Zen 1-3 desktop)

Spans Intel Haswell (2013) → Comet Lake (2020) and AMD Zen 1 → Zen 3 (2017-2020). All have AVX2 + FMA + F16C + BMI1/2; none have AVX-512.

- F16C: yes (Ivy Bridge+, 2012, ubiquitous on AVX2 silicon).
- BMI1, BMI2, ADX: yes.
- AES-NI, PCLMULQDQ, SHA-NI (Zen 1+ and Goldmont+): yes.
- No VNNI of any kind. No BF16. No AMX. No VBMI (that's AVX-512 only).
- Source: Standard Intel/AMD ISA references; Wikipedia AVX article.

#### Zen4 (AMD Ryzen 7000 desktop / EPYC 9004 server, TSMC N5, 2022)

ARK equivalents: Ryzen 9 7950X, EPYC 9654 (Genoa) etc.

- AVX-512: **F, CD, VL, DQ, BW, IFMA, VBMI, VBMI2, VNNI, BF16, FP16, VPOPCNTDQ, BITALG, GFNI, VAES, VPCLMULQDQ.** First AMD architecture with AVX-512.
- No AMX. No VP2INTERSECT.
- AVX-VNNI: yes (256-bit form), shares logic with AVX-512-VNNI internally.
- 256-bit FPU "double-pumped" for 512-bit ops — `VFMADD132PS zmm` takes 1 µop but two cycles to retire. Latency same as 256-bit; throughput halved. Practically: AVX-512 on Zen 4 ≈ 32 single-precision ops per cycle (same as Intel SPR), but with different power profile.
- Source: AMD EPYC 9004 series data sheet; Hot Chips 2023 Zen 4 / Genoa presentation; Wikipedia Zen 4; WikiChip Zen 4 microarchitecture page; ServeTheHome "AMD EPYC 9004 Genoa Zen 4 AVX 512 Bfloat16 And VNNI."

#### Zen5 (AMD Ryzen 9000 desktop / EPYC 9005 server / Threadripper 9000, TSMC N4P, 2024)

ARK equivalents: Ryzen 9 9950X, EPYC 9755 (Turin), Threadripper 9980X.

- AVX-512 ISA: **identical sub-feature set to Zen 4** — F, CD, VL, DQ, BW, IFMA, VBMI, VBMI2, VNNI, BF16, FP16, VPOPCNTDQ, BITALG, GFNI, VAES, VPCLMULQDQ.
- No AMX. No VP2INTERSECT. No new SIMD bits over Zen 4.
- The headline difference is microarchitectural: **full native 512-bit datapath** on Zen 5 desktop / EPYC (no double-pumping), 2× FMA throughput vs Zen 4 at the same clock. Zen 5c compact cores remain 256-bit double-pumped. There is a BIOS option ("double-pumped mode") to run Zen 5 as 256-bit for power reasons.
- Dispatch perspective: from `crate::simd::*` the ISA is the same as Zen 4 → use the same `Zen4Avx512` profile. The hardware difference is throughput, not capability. Splitting into `Zen5Native512` vs `Zen5DoublePumped` only matters if tile sizes need to differ — defer until benchmarks demand it.
- Source: AMD EPYC 9005 series page (amd.com); Phoronix "AVX-512 Performance With 256-bit vs. 512-bit Data Path For AMD EPYC 9005 CPUs"; Wikipedia Zen 5; Tom's Hardware EPYC 9005 launch.

---

## Master matrix — aarch64

Rows ordered by SoC tier (Pi family naming as canonical). **The existing detection helper `ArmProfile::arm_profile()` at `src/hpc/simd_caps.rs:317-336` already implements this dispatch and is the canonical reference.** It admits in its own comments that A72 silicon and A53-with-crypto silicon cannot be distinguished by HWCAP alone, and pragmatically maps both to `A72Fast` since the dispatch tables would be identical at the ISA level (both are ARMv8.0+crypto with no dotprod). The `A53Baseline` variant catches the rare case of NEON-without-crypto (QEMU, minimal aarch64 builds).

| CPU silicon | Runtime profile | NEON | dotprod | fp16 | bf16+ (BFMMLA/BFDOT) | i8mm (SMMLA/UMMLA) | crypto (aes+sha2) | crc32 | sve | sve2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cortex-A53 + crypto** (Pi 3, Pi Zero 2W) | `A72Fast` (heuristic) | DOC | ✗ | ✗ | ✗ | ✗ | DOC | DOC | ✗ | ✗ |
| **Cortex-A53 no crypto** (QEMU, minimal) | `A53Baseline` | DOC | ✗ | ✗ | ✗ | ✗ | ✗ | DOC | ✗ | ✗ |
| **Cortex-A72** (Pi 4, Orange Pi 4) | `A72Fast` | DOC | ✗ | ✗ | ✗ | ✗ | DOC | DOC | ✗ | ✗ |
| **Cortex-A76+** (Pi 5, Orange Pi 5, Apple M1+) | `A76DotProd` | DOC | DOC | DOC | DOC | DOC | DOC | DOC | ✗ | ✗ |

Apple M-series add SVE/SVE2 from M4 onwards; not yet in scope for this matrix.

### Microarchitecture notes (aarch64)

#### Cortex-A53 (ARMv8.0-A, 2013) — runtime profile `Armv8Neon`

Found in: Raspberry Pi 3, Pi Zero 2 W. Single NEON pipeline.

- NEON 128-bit: yes (mandatory on aarch64).
- AES, SHA-1, SHA-256, CRC32: yes.
- No dotprod, no fp16 arithmetic, no bf16+, no i8mm, no SVE.
- Source: ARM Cortex-A53 Technical Reference Manual; Pi 3 SoC docs.

#### Cortex-A72 (ARMv8.0-A, 2015) — runtime profile `Armv8Neon`

Found in: Raspberry Pi 4 (BCM2711), Orange Pi 4 LTS.

- NEON 128-bit: yes, with **2× pipeline width** vs A53 (dual NEON issue).
- AES, SHA-1, SHA-256, CRC32: yes.
- No dotprod (that's ARMv8.2+), no fp16 arithmetic, no bf16+, no i8mm, no SVE.
- Source: ARM Cortex-A72 TRM; Pi 4 SoC docs (BCM2711).

#### Cortex-A76+ (ARMv8.2-A, 2018) — runtime profile `A76DotProd`

Found in: Raspberry Pi 5 (BCM2712), Orange Pi 5 (RK3588 — A76+A55 big.LITTLE), Apple M1+ Firestorm cores (architecturally A77/A78-class, ARMv8.5+, but the relevant ISA bits are the same).

- NEON 128-bit: yes, dual pipeline.
- **dotprod (SDOT/UDOT, ARMv8.2-A `+dotprod`):** yes — 4× int8 dot product throughput vs scalar widen-multiply.
- **fp16 arithmetic (ARMv8.2-A `+fp16`):** yes — `vfmaq_f16`, `vaddq_f16`, etc.
- **bf16+ (ARMv8.6-A `+bf16`):** yes on Pi 5 A76 (via Cortex-A76 r4p1 firmware; check by `cat /proc/cpuinfo | grep bf16`). BFMMLA, BFDOT, BFCVT.
- **i8mm (ARMv8.6-A `+i8mm`):** yes — SMMLA / UMMLA / USMMLA 8×8 int8 matrix multiply.
- AES, SHA-2, CRC32: yes.
- No SVE on Pi 5 (Cortex-A76 predates SVE). Apple M4 onwards has SVE2.
- Source: ARM Cortex-A76 TRM; BCM2712 SoC docs; Raspberry Pi 5 release notes; Linux `/proc/cpuinfo` features observed on Pi 5 (`fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 asimddp sha512 asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 dgh bti ecv`).

---

## CPUID detection — exact leaves

For each x86 feature the matrix above lists, here's the CPUID leaf where its bit lives. This is what `simd_caps.rs::detect()` already queries via `is_x86_feature_detected!` (which itself does the CPUID lookup once and caches). Listed here for future hand-coded detection paths.

| Feature | CPUID leaf | Register | Bit |
|---|---|---|---|
| AVX-512F | 7,0 | EBX | 16 |
| AVX-512DQ | 7,0 | EBX | 17 |
| AVX-512IFMA | 7,0 | EBX | 21 |
| AVX-512CD | 7,0 | EBX | 28 |
| AVX-512BW | 7,0 | EBX | 30 |
| AVX-512VL | 7,0 | EBX | 31 |
| AVX-512VBMI | 7,0 | ECX | 1 |
| AVX-512VBMI2 | 7,0 | ECX | 6 |
| GFNI | 7,0 | ECX | 8 |
| VAES | 7,0 | ECX | 9 |
| VPCLMULQDQ | 7,0 | ECX | 10 |
| AVX-512VNNI | 7,0 | ECX | 11 |
| AVX-512BITALG | 7,0 | ECX | 12 |
| AVX-512VPOPCNTDQ | 7,0 | ECX | 14 |
| AVX-512VP2INTERSECT | 7,0 | EDX | 8 |
| AMX-BF16 | 7,0 | EDX | 22 |
| AVX-512FP16 | 7,0 | EDX | 23 |
| AMX-TILE | 7,0 | EDX | 24 |
| AMX-INT8 | 7,0 | EDX | 25 |
| AVX-VNNI | 7,1 | EAX | 4 |
| AVX-512BF16 | 7,1 | EAX | 5 |
| AVX-IFMA | 7,1 | EAX | 23 |
| AVX-NE-CONVERT | 7,1 | EDX | 5 |
| AMX-FP16 | 7,1 | EAX | 21 |
| AMX-COMPLEX | 7,1 | EDX | 8 |
| AVX-VNNI-INT8 | 7,1 | EDX | 4 |

The current `simd_amx.rs::amx_available()` correctly reads leaf 7,0 EDX bits 22/24/25 for AMX-TILE/INT8/BF16. The current `simd_caps.rs::detect()` does NOT read leaf 7,1, which means **AMX-FP16 and AMX-COMPLEX (Granite Rapids) are not currently detectable** — must add before GNR dispatch lands.

## OS state checks (additional gating beyond CPUID)

CPUID reports *capability*; the OS must additionally have enabled the relevant XSAVE state for the SIMD register file to be usable. Without that, the instruction SIGILLs even though CPUID is positive.

| Feature class | Required XCR0 bits | Required prctl |
|---|---|---|
| AVX | XCR0 bit 2 (YMM) | none |
| AVX-512 (all variants) | XCR0 bits 5+6+7 (OPMASK, ZMM_Hi256, Hi16_ZMM) | none |
| AMX-TILE/INT8/BF16/FP16 | XCR0 bits 17 (TILECFG) + 18 (TILEDATA) | `arch_prctl(ARCH_REQ_XCOMP_PERM, 18)` on Linux 5.16+ |

`simd_amx.rs::amx_available()` correctly checks all three for AMX. AVX-512 OS-state check is implicit in `is_x86_feature_detected!` (which calls into `std`'s detection path that checks XSAVE). No additional work needed for AVX-512 sub-features.

---

## `SimdProfile::detect()` mapping

The detect function in the integration plan resolves to one of these profiles. Decision order (most specific → least specific):

```rust
fn detect_x86() -> SimdProfile {
    let caps = simd_caps();
    // GraniteRapids: AMX-FP16 (CPUID 7,1 EAX bit 21)
    if caps.amx_fp16 { return SimdProfile::GraniteRapids; }
    // SapphireRapids / EmeraldRapids: AMX-TILE + AMX-BF16 + AMX-INT8 + AVX-512-FP16
    if caps.amx_tile && caps.amx_bf16 && caps.avx512fp16 { return SimdProfile::SapphireRapids; }
    // Zen 4 / Zen 5: AVX-512 + VBMI + BF16 + FP16, but NO AMX
    if caps.avx512f && caps.avx512vbmi && caps.avx512bf16 && caps.avx512fp16 && !caps.amx_tile {
        return SimdProfile::Zen4Avx512;
    }
    // CooperLake: AVX-512 + VNNI + BF16, but NO VBMI, NO FP16, NO AMX
    if caps.avx512f && caps.avx512vnni && caps.avx512bf16 && !caps.avx512vbmi {
        return SimdProfile::CooperLake;
    }
    // IceLakeSp + TigerLakeU: AVX-512 + VBMI + VNNI + IFMA, but NO BF16/FP16/AMX
    // Distinguish on VP2INTERSECT (TigerLake only)
    if caps.avx512f && caps.avx512vbmi && caps.avx512vnni && !caps.avx512bf16 {
        return if caps.avx512vp2intersect {
            SimdProfile::TigerLakeU
        } else {
            SimdProfile::IceLakeSp
        };
    }
    // CascadeLake: AVX-512 + VNNI, but NO VBMI/BF16
    if caps.avx512f && caps.avx512vnni { return SimdProfile::CascadeLake; }
    // SkylakeX: AVX-512F only
    if caps.avx512f { return SimdProfile::SkylakeX; }
    // ArrowLake: no AVX-512, has AVX-VNNI-INT8
    if caps.avxvnniint8 { return SimdProfile::ArrowLake; }
    // HaswellAvx2: AVX2 + FMA, nothing else
    if caps.avx2 && caps.fma { return SimdProfile::HaswellAvx2; }
    SimdProfile::Scalar
}
```

Critical detection invariants:

1. **GraniteRapids** must be checked *before* SapphireRapids — GNR has all the SPR bits plus AMX-FP16. If you check SPR first, GNR resolves as SPR and AMX-FP16 stays unused.
2. **Zen4 vs SapphireRapids** distinction: BOTH have F + VBMI + BF16 + FP16. The discriminator is **AMX-TILE present** (SPR) vs **AMX-TILE absent** (Zen 4). The pseudocode above orders SPR/GNR first to catch this.
3. **CooperLake vs IceLakeSp** distinction: BOTH are AVX-512 + VNNI. CPL has BF16, no VBMI. ICX has VBMI, no BF16. Mutually exclusive bit patterns — order doesn't matter as long as both branches check the discriminating bit.
4. **TigerLakeU vs IceLakeSp** distinction: same feature set EXCEPT VP2INTERSECT (TigerLake mobile only). Distinguish if you care; otherwise treat as one.
5. **ArrowLake** has no AVX-512 at all. Detection bypasses the AVX-512 cascade entirely.

## TEST verification checklist (promote DOC → TEST)

When this project gets first hardware access to a profile, run:

1. Boot the binary on the silicon and confirm `simd_profile()` returns the expected variant.
2. Exercise one instruction from each `DOC`-marked feature column (e.g. `_mm512_dpbf16_ps` for BF16, `_tile_dpbf16ps` for AMX-BF16) — confirm no SIGILL.
3. Compare a small reference computation against the scalar kernel — confirm bit-equivalent or within documented precision.
4. Promote the corresponding cells from `DOC` to `TEST` in this matrix, with a footnote citing the verification commit / CI run / serial of the hardware.

Until step 4 is done for a given profile, the dispatch is **safe but unverified by this project** — `DOC` means "we trust Intel/AMD, but we haven't run it."

---

## Out of scope for this matrix

- **Intel Knights Mill / Knights Landing** (Xeon Phi) — 4FMAPS, 4VNNIW, ERI, PFI subsets. Discontinued.
- **Intel Cannon Lake** (consumer 8th gen-ish, single SKU shipped) — limited AVX-512F/CD/VL/DQ/BW + IFMA + VBMI. Rare in the wild.
- **Intel Rocket Lake** (11th gen desktop) — similar to Ice Lake-SP ISA but missing VP2INTERSECT. Add if a consumer asks.
- **Intel Alder Lake / Raptor Lake** (12th/13th gen consumer) — AVX-512 silicon present but firmware-disabled. Cannot be reliably dispatched on.
- **Intel Meteor Lake** (14th gen Core Ultra) — no AVX-512; AVX-VNNI in P-cores only (hybrid). Similar to Arrow Lake but with different E-core (Crestmont vs Skymont).
- **Intel Sierra Forest** (Xeon 6 E-core, 2024+) — no AVX-512 at all. Pure E-core server SKU.
- **Lunar Lake** (consumer Ultra 200V, 2024) — similar ISA to Arrow Lake; same `ArrowLake` profile.
- **AMD Zen 3** and earlier — covered by `HaswellAvx2` (catch-all AVX2 + FMA).

Adding any of these is a one-row addition to the matrix above + one branch in `SimdProfile::detect()`.

