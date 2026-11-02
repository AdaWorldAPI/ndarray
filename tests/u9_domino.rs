#![cfg(feature = "std")]
//! D-DOMINO-MOV-0 exactness (`examples/u9_domino/domino.rs`) against
//! independent per-lane `u128` arithmetic.

#[path = "../examples/u9_head_to_head/kernels.rs"]
mod kernels;
#[path = "../examples/u9_abacus/carry16.rs"]
mod carry16;
#[path = "../examples/u9_domino/domino.rs"]
mod domino;

use domino::*;
use ndarray::simd::U16x32;

const N: usize = 96;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn full_tracked(name: &str, step: usize, cols: &[u64; N], full: u128) {
    for (k, &c) in cols.iter().enumerate() {
        assert_eq!(
            c == u64::MAX,
            full >> k & 1 == 1,
            "{name} step {step}: column {k} is {c:#x} but its FULL bit is {}",
            full >> k & 1
        );
    }
}

/// Applies `adds` to all four counters, checking every lane against a `u128`
/// oracle after each add (observations between additions).
/// P and PL also run over stale words (`STALE` under every EMPTY column), so
/// a lost EMPTY transition or a write skipped into a non-empty column shows.
fn check_all(adds: &[(usize, u64)]) -> [[u64; N]; 6] {
    const STALE: u64 = 0xDEAD_BEEF_F00D_CAFE;
    let (mut f, mut p) = (Eager::<N>::new(), Subst::<N>::new());
    let (mut l, mut pl) = (LazyFull::<N>::new(), SubstLazy::<N>::new());
    let (mut ps, mut pls) = (Subst::<N>::stale(STALE), SubstLazy::<N>::stale(STALE));
    let mut oracle = [0u128; 64];
    for (step, &(k, m)) in adds.iter().enumerate() {
        f.add_at(k, m).unwrap();
        p.add_at(k, m).unwrap();
        l.add_at(k, m).unwrap();
        pl.add_at(k, m).unwrap();
        ps.add_at(k, m).unwrap();
        pls.add_at(k, m).unwrap();
        for (j, o) in oracle.iter_mut().enumerate() {
            *o = o.checked_add(u128::from(m >> j & 1) << k).unwrap();
        }
        // L and PL hold every logically all-ones column as FULL, whichever
        // path produced it (the FULL -> EMPTY state-only carry depends on it).
        full_tracked("L", step, &l.columns(), l.full_bits());
        full_tracked("PL", step, &pl.columns(), pl.full_bits());
        full_tracked("PL stale", step, &pls.columns(), pls.full_bits());
        for j in [0usize, 1, 31, 32, 62, 63] {
            let want = oracle[j];
            assert_eq!(f.lane(j), want, "F step {step} lane {j}");
            assert_eq!(p.lane(j), want, "P step {step} lane {j}");
            assert_eq!(l.lane(j), want, "L step {step} lane {j}");
            assert_eq!(pl.lane(j), want, "PL step {step} lane {j}");
            assert_eq!(ps.lane(j), want, "P stale step {step} lane {j}");
            assert_eq!(pls.lane(j), want, "PL stale step {step} lane {j}");
        }
    }
    for (j, &o) in oracle.iter().enumerate() {
        assert_eq!(pl.lane(j), o, "PL final lane {j}");
        assert_eq!(f.lane(j), o, "F final lane {j}");
    }
    [f.columns(), p.columns(), l.columns(), pl.columns(), ps.columns(), pls.columns()]
}

/// The bit flip on every input pair, in every lane and in single lanes.
#[test]
fn flip_is_a_half_adder_in_every_lane() {
    for (b, c, nb, out) in [(0u64, 0u64, 0u64, 0u64), (0, 1, 1, 0), (1, 0, 1, 0), (1, 1, 0, 1)] {
        let fill = |x: u64| if x == 1 { u64::MAX } else { 0 };
        assert_eq!(flip(fill(b), fill(c)), (fill(nb), fill(out)));
        for j in [0, 17, 63] {
            assert_eq!(flip(b << j, c << j), (nb << j, out << j));
        }
    }
    let mut s = 7u64;
    for _ in 0..1000 {
        let (b, c) = (xorshift(&mut s), xorshift(&mut s));
        let (nb, out) = flip(b, c);
        for j in 0..64 {
            assert_eq!((b >> j & 1) + (c >> j & 1), (nb >> j & 1) + 2 * (out >> j & 1));
        }
    }
}

/// Random masks (zero, full, one lane, random) at random significances,
/// including 16, 63, 64 and beyond, in two different orders.
#[test]
fn random_adds_match_the_oracle_in_any_order() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut adds = Vec::new();
    for t in 0..3000u64 {
        let r = xorshift(&mut s);
        let m = match t % 5 {
            0 => 0,
            1 => u64::MAX,
            2 => 1 << (r % 64),
            _ => r,
        };
        let k = [0usize, 1, 15, 16, 17, 31, 63, 64, 65][(r >> 20) as usize % 9];
        adds.push((k, m));
    }
    let fwd = check_all(&adds);
    adds.reverse();
    let rev = check_all(&adds);
    for v in 0..6 {
        assert_eq!(fwd[v], rev[v], "variant {v}: order changed the result");
        assert_eq!(fwd[v], fwd[0], "variant {v} disagrees with F");
    }
}

/// Carry through 1, 2, 8, 16, 32 and 64 consecutive FULL columns, for all
/// lanes and for one lane, and two successive carries into the same column.
#[test]
fn carries_run_through_long_full_chains() {
    for n in [1usize, 2, 8, 16, 32, 64] {
        for m in [u64::MAX, 1 << 37] {
            let mut adds: Vec<(usize, u64)> = (0..n).map(|k| (k, m)).collect();
            adds.push((0, m));
            adds.push((0, m));
            let cols = check_all(&adds);
            // Value 2^n + 1 in the lanes of m, 0 elsewhere.
            for c in cols {
                assert_eq!(c[0], m);
                assert_eq!(c[n], m, "chain {n}");
                for (k, &w) in c.iter().enumerate() {
                    if k != 0 && k != n {
                        assert_eq!(w, 0, "chain {n} column {k}");
                    }
                }
            }
        }
    }
}

/// The 64K boundary: low16 + p crosses 2^16 exactly when the sum is at least
/// 65,536; the carry flips bit 16 and moves on to bit 17 when bit 16 is set.
#[test]
fn the_64k_boundary_flips_bit_16_then_carries_into_bit_17() {
    for low0 in [0u16, 1, 510, 511, 512, 65_000, 65_534, 65_535] {
        for p in [0u16, 1, 511, 65_024, 65_025] {
            for bit16_set in [false, true] {
                let mut lows = [U16x32::splat(low0); 2];
                let mut hi = Eager::<N>::new();
                if bit16_set {
                    hi.add_at(0, u64::MAX).unwrap();
                }
                let pv = [U16x32::splat(p); 2];
                let s = [lows[0] + pv[0], lows[1] + pv[1]];
                let m =
                    carry_mask(carry16::carry_after(lows[0], pv[0], s[0]), carry16::carry_after(lows[1], pv[1], s[1]));
                lows = s;
                hi.add_at(0, m).unwrap();
                let sum = u32::from(low0) + u32::from(p) + if bit16_set { 65_536 } else { 0 };
                for j in [0usize, 31, 32, 63] {
                    let low = lows[j / 32].to_array()[j % 32];
                    assert_eq!(u32::from(low) + ((hi.lane(j) as u32) << 16), sum, "low {low0} p {p} bit16 {bit16_set}");
                }
                let crossed = u32::from(low0) + u32::from(p) >= 65_536;
                let c = hi.columns();
                assert_eq!(c[0] == u64::MAX, crossed != bit16_set, "bit 16");
                assert_eq!(c[1] == u64::MAX, crossed && bit16_set, "bit 17");
            }
        }
    }
}

/// u16 products summed through every high representation against `u128`.
#[test]
fn product_sums_match_native_and_d() {
    let mut s = 0x0123_4567_89AB_CDEFu64;
    for (blocks, dense) in [(1usize, false), (300, false), (300, true), (70_000, true)] {
        let raw: Vec<[u16; 64]> = (0..blocks)
            .map(|_| {
                core::array::from_fn(|_| {
                    if dense {
                        65_025
                    } else {
                        let a = xorshift(&mut s);
                        ((a & 0xFF) * (a >> 8 & 0xFF)) as u16
                    }
                })
            })
            .collect();
        let prods: Vec<[U16x32; 2]> = raw
            .iter()
            .map(|p| [U16x32::from_slice(&p[..32]), U16x32::from_slice(&p[32..])])
            .collect();
        let want: u128 = raw.iter().flatten().map(|&x| u128::from(x)).sum();
        assert_eq!(sum_native(&raw), want);
        assert_eq!(sum_d(&prods), want, "D {blocks}");
        for each in [false, true] {
            assert_eq!(sum_bitsliced::<Eager<N>, N>(&prods, each), want, "F {blocks} {each}");
            assert_eq!(sum_bitsliced::<Subst<N>, N>(&prods, each), want, "P {blocks} {each}");
            assert_eq!(sum_bitsliced::<LazyFull<N>, N>(&prods, each), want, "L {blocks} {each}");
            assert_eq!(sum_bitsliced::<SubstLazy<N>, N>(&prods, each), want, "PL {blocks} {each}");
        }
    }
}

/// A carry past the last column is refused, in every variant.
#[test]
fn overflow_past_the_last_column_is_refused() {
    fn run<C: Counter<4>>() {
        let mut c = C::new();
        for k in 0..4 {
            c.add_at(k, 0b1010).unwrap();
        }
        assert_eq!(c.add_at(0, 0b0011), Err(Overflow(0b0010)));
        assert_eq!(c.add_at(4, 1), Err(Overflow(1)));
        assert_eq!(c.add_at(4, 0), Ok(()));
    }
    run::<Eager<4>>();
    run::<Subst<4>>();
    run::<LazyFull<4>>();
    run::<SubstLazy<4>>();
}

/// R1 (two words) and R2 (one u128) hold the same two planes and agree on
/// every add; the roundtrip is the identity.
#[test]
fn pair_and_packed_agree_and_roundtrip() {
    let mut s = 42u64;
    let mut p = Pair128 { lo: 0, hi: 0 };
    let mut x = 0u128;
    for _ in 0..10_000 {
        let m = xorshift(&mut s);
        let (np, o1) = pair_add(p, m);
        let (nx, o2) = packed_add(x, m);
        assert_eq!((pack(np), o1), (nx, o2));
        assert_eq!(unpack(nx), np);
        // 64 independent 2-bit counters, not one 128-bit integer.
        for j in 0..64 {
            let before = (p.lo >> j & 1) + 2 * (p.hi >> j & 1);
            let after = (np.lo >> j & 1) + 2 * (np.hi >> j & 1) + 4 * (o1 >> j & 1);
            assert_eq!(after, before + (m >> j & 1), "lane {j}");
        }
        p = np;
        x = nx;
    }
}

/// A column made all ones by an ordinary XOR (complementary masks at one
/// significance) is FULL in PL as in L, and the next all-ones carry passes it
/// as FULL -> EMPTY without touching the word.
#[test]
fn xor_made_full_column_is_full_and_carries_by_state() {
    let m = 0x0F0F_0F0F_0F0F_0F0Fu64;
    let mut l = LazyFull::<N>::new();
    let mut pl = SubstLazy::<N>::new();
    for (k, mask) in [(0, m), (0, !m)] {
        l.add_at(k, mask).unwrap();
        pl.add_at(k, mask).unwrap();
    }
    assert_eq!(l.full_bits() & 1, 1, "L marks the XOR-made column FULL");
    assert_eq!(pl.full_bits() & 1, 1, "PL marks the XOR-made column FULL");
    pl.add_at(0, u64::MAX).unwrap();
    assert_eq!(pl.full_bits() & 1, 0);
    assert_eq!(pl.empty_bits() & 1, 1, "FULL + all ones leaves column 0 EMPTY");
    // Column 1 was EMPTY, so it takes the all-ones carry by assignment and
    // is FULL in turn.
    assert_eq!(pl.full_bits() >> 1 & 1, 1, "the all-ones carry lands in column 1 as FULL");
    for j in 0..64 {
        assert_eq!(pl.lane(j), 2, "lane {j}");
    }
}

// ── D-BINARY-ABACUS-POLYFILL-0: candidate C (`Abacus<BITS>`) ─────────────────

/// 64 independent u128 counters: the reference for `Abacus`.
fn reference_add(want: &mut [u128; 64], k: usize, m: u64) {
    for (j, w) in want.iter_mut().enumerate() {
        *w += u128::from(m >> j & 1) << k;
    }
}

fn assert_matches<const B: usize>(a: &Abacus<B>, want: &[u128; 64], ctx: &str) {
    for (j, &w) in want.iter().enumerate() {
        assert_eq!(a.lane_value(j), w, "{ctx}: lane {j}");
    }
    assert_eq!(a.total(), want.iter().sum::<u128>(), "{ctx}: total");
}

#[test]
fn abacus_single_lanes_stay_in_their_lane() {
    for j in [0usize, 31, 32, 63] {
        let mut a = Abacus::<16>::new();
        for _ in 0..1000 {
            a.try_add_at(0, 1 << j).unwrap();
        }
        for l in 0..64 {
            assert_eq!(a.lane_value(l), if l == j { 1000 } else { 0 }, "lane {j} leaked into {l}");
        }
    }
}

#[test]
fn abacus_carry_chains_of_every_length() {
    for len in [1usize, 2, 8, 16, 32, 64] {
        for j in [0usize, 31, 32, 63] {
            let mut a = Abacus::<96>::new();
            // Preset lane j to 2^len - 1 (columns 0..len all ones in lane j).
            for k in 0..len {
                a.try_add_at(k, 1 << j).unwrap();
            }
            assert_eq!(a.lane_value(j), (1u128 << len) - 1);
            a.try_add_at(0, 1 << j).unwrap();
            assert_eq!(a.lane_value(j), 1u128 << len, "chain {len} lane {j}");
            let cols = a.columns();
            assert_eq!(cols[len], 1 << j, "carry landed in column {len}");
            assert!(cols[..len].iter().all(|&c| c == 0), "chain {len}: lower columns cleared");
        }
    }
}

#[test]
fn abacus_repeated_carries_add_up() {
    // Two carries into the same column are two units, never an XOR cancel.
    let mut a = Abacus::<8>::new();
    a.try_add_at(3, u64::MAX).unwrap();
    a.try_add_at(3, u64::MAX).unwrap();
    for j in 0..64 {
        assert_eq!(a.lane_value(j), 16);
    }
    a.try_add_at(4, u64::MAX).unwrap();
    for j in 0..64 {
        assert_eq!(a.lane_value(j), 32);
    }
}

#[test]
fn abacus_overflow_refuses_and_leaves_the_counter_unchanged() {
    let mut s = 0xC0FF_EE00_DEAD_BEEFu64;
    let mut a = Abacus::<8>::new();
    // Lanes 0, 31, 32, 63 at 255 (top of 8 bits), the rest random below 255.
    let full = (1u64 << 0) | (1 << 31) | (1 << 32) | (1 << 63);
    for k in 0..8 {
        a.try_add_at(k, full | (xorshift(&mut s) & !full & !(1 << 5)))
            .unwrap();
    }
    let before = a.columns();
    // Adding 1 to a mask containing a full lane must carry out of column 7.
    for m in [full, 1 << 63, (1 << 0) | 0xF0, u64::MAX] {
        let err = a.try_add_at(0, m);
        assert!(matches!(err, Err(AbacusError::Overflow { lanes }) if lanes & full != 0), "{m:#x}: {err:?}");
        assert_eq!(a.columns(), before, "refused add {m:#x} changed the counter");
    }
    // A lane with headroom still accepts.
    a.try_add_at(0, 1 << 5).unwrap();
    assert_ne!(a.columns(), before);
    // Significance past the top is refused without touching anything.
    let before = a.columns();
    assert_eq!(
        a.try_add_at(8, 1),
        Err(AbacusError::SignificanceOutOfRange {
            significance: 8,
            bits: 8
        })
    );
    assert_eq!(a.columns(), before);
}

#[test]
fn abacus_random_sequences_match_64_u128_counters() {
    let mut s = 0x0123_4567_89AB_CDEFu64;
    const B: usize = 12;
    let (mut refused, mut accepted) = (0, 0);
    for seq in 0..40 {
        let mut a = Abacus::<B>::new();
        let mut want = [0u128; 64];
        for step in 0..600 {
            let r = xorshift(&mut s);
            let m = match r % 6 {
                0 => 0,
                1 => u64::MAX,
                2 => 0xAAAA_AAAA_AAAA_AAAA,
                3 => 1 << (r >> 8 & 63),
                _ => xorshift(&mut s),
            };
            let k = (xorshift(&mut s) % B as u64) as usize;
            let mut next = want;
            reference_add(&mut next, k, m);
            let fits = next.iter().all(|&w| w < 1 << B);
            let got = a.try_add_at(k, m);
            if fits {
                assert_eq!(got, Ok(()), "seq {seq} step {step}");
                want = next;
                accepted += 1;
            } else {
                assert!(matches!(got, Err(AbacusError::Overflow { .. })), "seq {seq} step {step}: {got:?}");
                refused += 1;
            }
            assert_matches(&a, &want, &format!("seq {seq} step {step}"));
        }
    }
    // Anti-vacuity: both outcomes occur often.
    assert!(accepted > 5000 && refused > 1000, "accepted={accepted} refused={refused}");
}

#[test]
fn abacus_reversed_order_gives_the_same_counters() {
    // Addition is commutative, so the fold may run in either order.
    let mut s = 77u64;
    let adds: Vec<(usize, u64)> = (0..3000)
        .map(|_| ((xorshift(&mut s) % 40) as usize, xorshift(&mut s)))
        .collect();
    let (mut fwd, mut rev) = (Abacus::<72>::new(), Abacus::<72>::new());
    for &(k, m) in &adds {
        fwd.try_add_at(k, m).unwrap();
    }
    for &(k, m) in adds.iter().rev() {
        rev.try_add_at(k, m).unwrap();
    }
    assert_eq!(fwd.columns(), rev.columns());
}
