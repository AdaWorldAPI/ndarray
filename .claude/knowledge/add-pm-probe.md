# D-ADD-PM: Codec9 `P × M` lanes added in place

READ BY: savant-architect, vector-synthesis, simd-savant, the DR512 / Codec9
owner. Follows `dr512-epistemic-pm-invariant.md` (the law) and
`u9-abacus-probe.md` (D: carry escalation into a wider tally).

Probe: `examples/add_pm/` (`pm.rs` holds the arithmetic, `main.rs` times it).
Correctness: `tests/add_pm.rs`. Facade words only; no backend, facade or
`codec9` change.

```sh
cargo test --release --test add_pm --features std
cargo run --release --example add_pm --features std
```

## Question

Can signed deltas in DR512 form (one `u8` magnitude + one presence bit per
lane; `P=0, M=0` never touched) be added without decoding to a signed
integer, keep the never-touched state, and report overflow instead of
saturating?

## The operation

`add_pm(a, b) -> (lanes, carry)` over 64 lanes, sign-magnitude:

- same polarity: `M = sat(M_a + M_b)`, `P = P_a`; carry where `M_a > 255 - M_b`;
- opposite polarity: `M = max - min`, `P` of the larger magnitude;
- a zero result is `+0` when either input was touched; two never-touched
  inputs stay never touched.

Words: `saturating_add`, `saturating_sub`, `simd_max`, `simd_min`,
`cmpgt_mask` (unsigned), `cmpeq_mask`, one `mask_blend`, and `u64` mask
logic. No shuffle, no `u8`/`i8` crossing.

**Finding:** no identity blend is needed. The never-touched code is a zero
magnitude, so `untouched ⊕ x = x` is the sign-magnitude addition itself. A
disable run that removed the first version's identity step stayed green; the
step was dead and is gone. Presence matters in exactly one place: the
zero-to-+0 normalization must not fire when both inputs were untouched.

The carry mask is the exact escape Codec9's encoder does not have (it
saturates silently, as D-XLOG-C9-0 measured in lance-graph): the caller adds
the carry lanes' exact sum through a wider tally, which is D's carry
escalation.

## Correctness (all green)

| test | what |
|---|---|
| `every_codeword_pair_matches_the_i32_oracle` | all 512 × 512 pairs; carry count = exactly the touched same-polarity pairs with \|a\|+\|b\| > 255 (65,280 = 2 × 32,640) |
| `never_touched_comes_out_only_of_two_never_touched_inputs` | touched(out) = touched(a) \| touched(b), all pairs |
| `a_touched_zero_is_plus_zero` | −5 + +5, +0 + +0, −255 + +255 give `P=1, M=0` |
| `add_pm_equals_the_decode_path_on_every_pair` | bit-identical to decode → i32 → encode |
| `addition_is_associative_where_nothing_carries` | 20,000 random triple blocks, > 100,000 lanes compared |
| `a_fold_with_the_carry_escape_is_exact` | 5,000-block fold, carry lanes escaped to a wide tally, equals the i64 sum |

Disable controls, each red:
D1 no zero-to-+0 normalization (5 tests);
D2 normalization not gated on presence (3);
D3 saturated sum treated as exact, carry forced 0 (4);
D4 opposite polarity takes the smaller magnitude's sign (4).

## Cost (ns per 64 lanes, best of 50, 4,096 blocks, L2-resident)

| tier | `add_pm` | decode → i32 → encode |
|---|---|---|
| AVX-512 (default `target-cpu=native`, built `avx512bw=true`) | 11.8 – 11.9 | 89.7 |
| AVX2 (`--config .cargo/config-v3.toml`, built `avx512bw=false`) | 284.7 – 285.3 | 133.5 – 134.7 |

On AVX-512 the in-place sum is about 7.5x the decode path. On AVX2 it is
about 2x SLOWER, and not because of the algebra: the AVX2 `U8x64` is
array-backed, and its `mask_blend`, `saturating_add` and `saturating_sub` are
per-byte scalar loops (`src/simd_avx2.rs`, the 81 array-backed AVX2 hits of
the 2026-10-27 facet-stride audit). The AVX2 number measures that facade
gap. Same host for both rows; the AVX2 row is the v3 build on an AVX-512
machine, not AVX2 silicon.

## Verdict

ADOPT-LAB as the signed-delta operand shape: exact, presence-preserving, and
the carry is the escape. Not adopted into `codec9` or the facade.

## Open

- Burn down the AVX2 array-backed `U8x64` words this uses; then re-time.
- End to end in lance-graph: fold the triangle-pairing delta stream
  (D-XLOG-C9-0) with `add_pm` and check the carry lanes equal its escape lanes.
- Only addition is covered. Subtraction is addition with the polarity bit
  flipped on touched lanes (an untouched lane must stay untouched); not built.
