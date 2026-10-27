# Palette512 / SignedMagnitude9: isolated arithmetic research gate

> **2026-10-27 · EXPERIMENT / NOT ADMITTED**  
> Owner for this probe: **ndarray numerical kernels**. This note and `examples/palette512_probe.rs` do **not** approve a production codec, a BGZ codebook expansion, an ONNX/TFLite compatibility claim, a Quack/DT64 change, or a wire migration.

## Hypothesis

Use a separate sign plane with 64 unsigned magnitudes, giving 512 encodings per lane:

- `sign=0, magnitude=0`: valid +0.
- `sign=0, magnitude=1..255`: +1..+255.
- `sign=1, magnitude=1..255`: -1..-255.
- `sign=1, magnitude=0`: one NaN/invalid sentinel (negative zero).

There are **511 numeric values** and one invalid codeword, versus 255 numeric values and one NaN codeword for the existing 8-bit `Sym + NaNAtMin` domain (`-127..127` plus `-128`). At equal physical range, the added levels *could* allow finer quantization; calibrated quality and end-to-end cost are unmeasured.

A **64-lane** block stores **64 bytes of u8 magnitude + one u64 sign mask = 72 bytes**, or exactly 9 bits per lane. This is a packed block representation, **not** a 9-bit signed integer that native hardware can multiply as `i8`. Physical `repr(C)` footprint is *not* a persisted wire format.

### Arithmetic seam

For magnitudes `a,b` and signs `sa,sb`:

```
product = u8(a) * u8(b)         // widened temporary for correctness
acc = (sa XOR sb) ? (acc - product) : (acc + product)
```

That is the proposed **sign-aware multiply-accumulate**: sign handling belongs inside the operation, not in a prior full-buffer decode. The scalar probe uses checked `i32` accumulation and fails closed on NaN or overflow, distinctly. SIMD fusion is a *hypothesis*, not an implemented vector kernel or performance result.

**Important:** x86 VNNI `VPDPBUSD` accepts `u8 × i8`, not two arbitrary `u8` magnitudes. Unsigned × unsigned exists elsewhere: AVX-VNNI-INT8 `VPDPBUUD`, NEON `UDOT`, wasm `extmul_*_u8x16`, or zero-extend plus `pmaddwd`. That list is from documentation, not measured. A lowering that splits lanes by `sign_a ^ sign_b` into two unsigned accumulators (`dot = pos - neg`) needs no correction identity. No SIMD efficiency is presumed.

## Semantic separation from current Palette256

**This is not a generic 512-centroid palette.** Existing `bgz-tensor` `WeightPalette` uses `u8` *archetype indices*, with 256 distinct codes and associated pair LUTs. The sign-magnitude encoding instead indexes a *structured signed numeric domain*. It cannot reinterpret arbitrary Palette256 identifiers by declaring half of their coordinates negative, and it cannot steal an independent presence bit without proving the original presence contract.

An actual **512-archetype** codebook would need a 9-bit index and new pair-LUT addressing: a flat 512×512×u16 table is 512 KiB, compared with 128 KiB at 256²; a 512×512×u16 compose table adds another 512 KiB rather than the existing 64 KiB u8 table. Any sign-symmetric factorization must be individually proven for the relevant distance/compose algebra. The present PR adds **none** of these tables.

**NaN = `-0` = presence clear over a stored 0** (settled in #381). Where the column has a presence plane, NaN is a cleared presence bit with magnitude 0 stored under it, and "sign set, magnitude 0" is the same NaN written without the plane. All of −255..=+255 then stay values, and NaN never reaches a multiply: lanes are selected by `presence_a & presence_b & filter` first. A codec that needs *several* distinguishable invalid states (NULL, unknown, denied) still needs more than one codeword. Quack's predicate/null semantics remain its own contract.

## First executable evidence (this PR)

`cargo test --example palette512_probe`

Five tests:

1. Enumerate all 512 codewords and prove round-trip, **511 numeric + 1 NaN**.
2. Enumerate all **262,144 ordered pairs** of codewords; compare sign-aware `mul_add` against an independently decoded signed multiplication plus a nonzero accumulator.
3. Check a 64-lane sign-plane block against an independent wide `i64` dot-product reference and assert its 72-byte in-memory footprint.
4. Verify NaN propagation even when the other operand is zero.
5. Distinguish overflow from NaN, including positive and negative accumulator boundaries.

This is a reference-quality **scalar example**, not SIMD code, a production API, or a calibrated numerical comparison. All five tests pass (`cargo test --example palette512_probe`, rustc 1.99.0), including the 262,144-pair check.

## Admission and falsification

- **Correctness falsifier:** Any of the 512² pairs disagrees with independently decoded arithmetic; positive zero aliases negative-zero sentinel; NaN silently becomes zero; overflow is misreported.
- **Representation falsifier:** Existing source presence/null meaning is lost, or conversion cannot preserve the independently required missingness states.
- **Numerical falsifier:** A *held-out, fixed dynamic-range* calibration shows no meaningful improvement in endpoint saturation/tail error/task-level accuracy versus current `Codec8<Sym,NaNAtMin>`.
- **Performance falsifier:** After fair end-to-end measurement, packing+sign correction outweighs quality gains or unpacks/widening dominate; evaluate AVX2, AVX-512/VNNI, NEON, WASM, scalar separately, without assuming each ISA offers the same fused instruction.
- **Algebra falsifier:** Any attempted folded sign-aware LUT violates the independently defined distance/compose relation, especially zero/NaN, negation, and multi-hop cases.

**Next gate, separately approved:** implement SIMD lowering and benchmark sign-aware `mul_add` vs established `i8` and `u8 × i8` kernels; measure cycles/value, packing cost, throughput and errors separately. A later **independent** gate may investigate compressed/factored 512-value LUT laws if relevant.

## Related authorities (read-only links)

- [ndarray Codec8 plan](https://github.com/AdaWorldAPI/ndarray/blob/master/.claude/plans/quantized-codec8-v2.md), [#370](https://github.com/AdaWorldAPI/ndarray/pull/370): compile-time numeric codec; **unchanged**.
- [Existing VNNI GEMM](https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/vnni_gemm.rs): `u8 × i8 → i32`; **not** a ready-made unsigned-unsigned signed-mask MAC.
- [BGZ palette](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/palette.rs), [attention LUT](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/attention.rs): **distinct** archetype-index contract.
- [D-NAM-PZL / EXO-152](https://linear.app/exo-red/issue/EXO-152/d-nam-capstone-normalized-native-algebra-onnxtflitemlir-dto-method), [EXO-50 coordination](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id): context only; no production permission implied.
