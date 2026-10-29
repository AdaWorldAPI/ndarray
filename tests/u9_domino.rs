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
