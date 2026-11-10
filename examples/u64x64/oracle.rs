//! D-U64X64-LLVM-1 independent scalar oracle.
//!
//! Every operation is literal per-element code over `(k, r)`: one bit read at
//! a time with `(word >> r) & 1`, arithmetic in `u64`, no SIMD, no ndarray API,
//! no `count_ones`. This is the reference every route is checked against, so it
//! is written to be obviously correct, not fast. Imports only fixture types.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::fixture::*;

/// Bit `r` of `w`, as 0 or 1.
fn bit(w: u64, r: usize) -> u64 {
    (w >> r) & 1
}

/// Population count, one bit at a time.
fn popcount_bits(w: u64) -> u64 {
    let mut n = 0u64;
    for r in 0..64 {
        n += bit(w, r);
    }
    n
}

/// O1 masked count/intersection over bits.
///
/// `plane_cnt[k] = popcnt(A[k] & B[k] & live)`; `row_cnt[r]` counts categories
/// whose bit r is set in `A[k] & B[k]`, for r in `live`, else 0.
pub fn o1(a: &BitTile, b: &BitTile, live: u64, out: &mut O1Out) {
    let mut total = 0u64;
    for k in 0..64 {
        let mut n = 0u64;
        for r in 0..64 {
            n += bit(a[k], r) & bit(b[k], r) & bit(live, r);
        }
        out.plane_cnt[k] = n as u8;
        total += n;
    }
    for r in 0..64 {
        let mut n = 0u64;
        if bit(live, r) == 1 {
            for k in 0..64 {
                n += bit(a[k], r) & bit(b[k], r);
            }
        }
        out.row_cnt[r] = n as u8;
    }
    out.total = total as u16;
}

/// O2 exact unsigned threshold: `ge[k]` bit r = `S[k][r] >= t[k]` for demanded
/// k (zero otherwise); `all_ge` is the AND over demanded k, all ones if none.
/// `t = 65536` never matches.
pub fn o2(bound: &Bound, t: &[u32; 64], demanded: u64, out: &mut O2Out) {
    let mut all_ge = u64::MAX;
    for k in 0..64 {
        let mut ge = 0u64;
        if bit(demanded, k) == 1 {
            for r in 0..64 {
                let s = bound.src.stock[k][r] as u32;
                if s >= t[k] {
                    ge |= 1u64 << r;
                }
            }
            all_ge &= ge;
        }
        out.ge[k] = ge;
    }
    out.all_ge = all_ge;
}

/// O3 presence-aware min/argmin per row over demanded categories.
///
/// Only present slots count. `arg` is the smallest k attaining the min. No
/// present value gives `min = 0, arg = 0xFF` and the `none` bit. The `unknown`
/// bit is set when any demanded category is absent at that row. Stock of an
/// absent slot is never read.
pub fn o3(bound: &Bound, demanded: u64, out: &mut O3Out) {
    let mut none = 0u64;
    let mut unknown = 0u64;
    for r in 0..64 {
        let mut have = false;
        let mut best = 0u16;
        let mut arg = 0u8;
        for k in 0..64 {
            if bit(demanded, k) == 0 {
                continue;
            }
            if bit(bound.src.present[k], r) == 0 {
                unknown |= 1u64 << r;
                continue;
            }
            let v = bound.src.stock[k][r];
            if !have || v < best {
                have = true;
                best = v;
                arg = k as u8;
            }
        }
        if have {
            out.min[r] = best;
            out.arg[r] = arg;
        } else {
            none |= 1u64 << r;
            out.min[r] = 0;
            out.arg[r] = 0xFF;
        }
    }
    out.none = none;
    out.unknown = unknown;
}

/// O4 weighted row sum: `sum[r] = sum_k w_k * C[k][r]` (u32); `total` is the
/// sum of `sum[r]` over rows in `live` (u64).
pub fn o4(cost: &ValueTile<u8>, w: &[u16; 64], live: u64, out: &mut O4Out) {
    let mut total = 0u64;
    for r in 0..64 {
        let mut s = 0u64;
        for k in 0..64 {
            s += (w[k] as u64) * (cost[k][r] as u64);
        }
        out.sum[r] = s as u32;
        if bit(live, r) == 1 {
            total += s;
        }
    }
    out.total = total;
}

/// O4a checked add: `Z = (X + Y) mod 2^16`; `ovf[k]` bit r = `X + Y >= 2^16`.
pub fn o4a(x: &ValueTile<u16>, y: &ValueTile<u16>, out: &mut O4aOut) {
    for k in 0..64 {
        let mut ovf = 0u64;
        for r in 0..64 {
            let s = x[k][r] as u64 + y[k][r] as u64;
            out.z[k][r] = (s & 0xFFFF) as u16;
            if s >= 65536 {
                ovf |= 1u64 << r;
            }
        }
        out.ovf[k] = ovf;
    }
}

/// O5 one Abakus update. `b >= 16` refuses `SignificanceOutOfRange`; if any
/// lane r of `m` has `c + 2^b >= 2^16` the update refuses `Overflow{lanes}`
/// with those lanes and changes nothing; otherwise `2^b` is added on every
/// lane of `m`.
pub fn o5(counters: &mut Counters, u: O5Update) -> Result<(), Refusal> {
    if u.b >= 16 {
        return Err(Refusal::SignificanceOutOfRange);
    }
    let k = u.k as usize;
    let inc = 1u64 << u.b;
    let mut lanes = 0u64;
    for r in 0..64 {
        if bit(u.m, r) == 1 && counters[k][r] as u64 + inc >= 65536 {
            lanes |= 1u64 << r;
        }
    }
    if lanes != 0 {
        return Err(Refusal::Overflow { lanes });
    }
    for r in 0..64 {
        if bit(u.m, r) == 1 {
            counters[k][r] = (counters[k][r] as u64 + inc) as u16;
        }
    }
    Ok(())
}

/// O5 stream: apply each update in order, refused ones leave the state
/// unchanged. Returns `(applied, refused)`.
pub fn o5_stream(counters: &mut Counters, updates: &[O5Update]) -> (usize, usize) {
    let mut applied = 0;
    let mut refused = 0;
    for &u in updates {
        match o5(counters, u) {
            Ok(()) => applied += 1,
            Err(_) => refused += 1,
        }
    }
    (applied, refused)
}

/// O6 CanCook capacity (Foundry G1), per row.
///
/// Gate closed: Denied. Else any demanded category absent: Unknown with the
/// smallest such k. Else Known with
/// `cap = min(min_{k in D} floor(S / d_k), staff * u, 128)` and `limit` the
/// first of (k ascending, `STAFF`, `CEILING`) whose value equals `cap`.
/// Empty demand gives `min(staff * u, 128)`.
pub fn o6(bound: &Bound, params: &Params, out: &mut O6Out) {
    let src = bound.src;
    let d = &params.recipe.d;
    let mut known = 0u64;
    let mut unknown = 0u64;
    let mut denied = 0u64;
    for r in 0..64 {
        out.cap[r] = 0;
        out.limit[r] = 0;
        if bit(src.gate, r) == 0 {
            out.status[r] = RowStatus::Denied;
            denied |= 1u64 << r;
            continue;
        }
        let mut missing: Option<u8> = None;
        for k in 0..64 {
            if d[k] > 0 && bit(src.present[k], r) == 0 {
                missing = Some(k as u8);
                break;
            }
        }
        if let Some(k) = missing {
            out.status[r] = RowStatus::Unknown { k };
            unknown |= 1u64 << r;
            continue;
        }
        // Known row.
        let staff_cap = src.staff[r] as u64 * params.u as u64;
        let mut cap = staff_cap.min(QMAX as u64);
        for k in 0..64 {
            if d[k] > 0 {
                let q = src.stock[k][r] as u64 / d[k] as u64;
                if q < cap {
                    cap = q;
                }
            }
        }
        let mut limit: Option<u8> = None;
        for k in 0..64 {
            if d[k] > 0 && src.stock[k][r] as u64 / d[k] as u64 == cap {
                limit = Some(k as u8);
                break;
            }
        }
        let limit = match limit {
            Some(k) => k,
            None if staff_cap == cap => STAFF,
            None => CEILING,
        };
        out.status[r] = RowStatus::Known;
        out.cap[r] = cap as u8;
        out.limit[r] = limit;
        known |= 1u64 << r;
    }
    out.known = known;
    out.unknown = unknown;
    out.denied = denied;
}

/// Rows `r` that are Known with `cap_r >= q`.
pub fn can_cook(out: &O6Out, q: u32) -> u64 {
    let mut m = 0u64;
    for r in 0..64 {
        if out.status[r] == RowStatus::Known && out.cap[r] as u32 >= q {
            m |= 1u64 << r;
        }
    }
    m
}

/// `sum over Known rows of cap_r`.
pub fn network_capacity(out: &O6Out) -> u64 {
    let mut n = 0u64;
    for r in 0..64 {
        if out.status[r] == RowStatus::Known {
            n += out.cap[r] as u64;
        }
    }
    n
}

/// `sum_{q=1}^{128} Count(CanCook(q))`, counted bit by bit.
pub fn network_capacity_by_q(out: &O6Out) -> u64 {
    let mut n = 0u64;
    for q in 1..=QMAX {
        n += popcount_bits(can_cook(out, q));
    }
    n
}

/// `sum over r in CanCook(q) of cost_r(q)`, `cost_r(q) = sum_{k in D} d_k * q * C[k][r]`.
pub fn network_cost(bound: &Bound, params: &Params, out: &O6Out, q: u32) -> u64 {
    let can = can_cook(out, q);
    let mut total = 0u64;
    for r in 0..64 {
        if bit(can, r) == 0 {
            continue;
        }
        for k in 0..64 {
            let d = params.recipe.d[k] as u64;
            if d > 0 {
                total += d * q as u64 * bound.src.cost[k][r] as u64;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound_of(src: &Source) -> Bound<'_> {
        bind(src, &Expect::of(src)).unwrap()
    }

    #[test]
    fn o1_hand_computed() {
        let mut a = [0u64; 64];
        let mut b = [0u64; 64];
        a[0] = 0b1011;
        b[0] = 0b0110;
        a[5] = 0b0010;
        b[5] = 0b0010;
        let mut out = O1Out::new();
        o1(&a, &b, u64::MAX, &mut out);
        // A&B: word 0 = 0b0010, word 5 = 0b0010.
        assert_eq!(out.plane_cnt[0], 1);
        assert_eq!(out.plane_cnt[5], 1);
        assert_eq!(out.plane_cnt[1], 0);
        assert_eq!(out.row_cnt[1], 2);
        assert_eq!(out.row_cnt[0], 0);
        assert_eq!(out.total, 2);
        // Row 1 masked out of live.
        o1(&a, &b, !0b10u64, &mut out);
        assert_eq!(out.row_cnt[1], 0);
        assert_eq!(out.plane_cnt[0], 0);
        assert_eq!(out.total, 0);
    }

    #[test]
    fn o1_all_ones() {
        let (a, b) = carry_bits();
        let mut out = O1Out::new();
        o1(&a, &b, u64::MAX, &mut out);
        assert_eq!(out.plane_cnt, [64; 64]);
        assert_eq!(out.row_cnt, [64; 64]);
        assert_eq!(out.total, 4096);
    }

    #[test]
    fn o2_threshold_neighbours() {
        let mut s = Source::blank(50);
        s.stock[3][5] = 100;
        let b = bound_of(&s);
        let d = 1u64 << 3;
        let mut out = O2Out::new();
        let mut t = [0u32; 64];

        t[3] = 99;
        o2(&b, &t, d, &mut out);
        assert_eq!(out.ge[3], 1 << 5);
        assert_eq!(out.all_ge, 1 << 5);

        t[3] = 100;
        o2(&b, &t, d, &mut out);
        assert_eq!(out.ge[3], 1 << 5);

        t[3] = 101;
        o2(&b, &t, d, &mut out);
        assert_eq!(out.ge[3], 0);
        assert_eq!(out.all_ge, 0);

        t[3] = 65536;
        o2(&b, &t, d, &mut out);
        assert_eq!(out.ge[3], 0);

        t[3] = 0;
        o2(&b, &t, d, &mut out);
        assert_eq!(out.ge[3], u64::MAX);

        // Non-demanded categories produce no bits; empty demand is all ones.
        assert_eq!(out.ge[4], 0);
        o2(&b, &t, 0, &mut out);
        assert_eq!(out.all_ge, u64::MAX);
    }

    #[test]
    fn o3_ties_and_absent_stale() {
        let mut s = Source::blank(51);
        // Row 0: k=1 present 50, k=2 absent but holds a lower stale value,
        // k=4 present 60. The absent 1 must not win.
        s.set_present(1, 0, true);
        s.stock[1][0] = 50;
        s.set_present(2, 0, false);
        s.stock[2][0] = 1;
        s.set_present(4, 0, true);
        s.stock[4][0] = 60;
        // Row 2: tie 70/70/80 -> smallest k.
        s.set_present(1, 2, true);
        s.stock[1][2] = 70;
        s.set_present(2, 2, true);
        s.stock[2][2] = 70;
        s.set_present(4, 2, true);
        s.stock[4][2] = 80;
        // Row 1: nothing present.
        let b = bound_of(&s);
        let d = (1u64 << 1) | (1 << 2) | (1 << 4);
        let mut out = O3Out::new();
        o3(&b, d, &mut out);

        assert_eq!(out.min[0], 50);
        assert_eq!(out.arg[0], 1);
        assert_eq!(out.unknown & 1, 1);
        assert_eq!((out.none >> 0) & 1, 0);

        assert_eq!(out.min[2], 70);
        assert_eq!(out.arg[2], 1);
        assert_eq!((out.unknown >> 2) & 1, 0);

        assert_eq!(out.min[1], 0);
        assert_eq!(out.arg[1], 0xFF);
        assert_eq!((out.none >> 1) & 1, 1);
        assert_eq!((out.unknown >> 1) & 1, 1);
    }

    #[test]
    fn o3_equal_values_pick_arg_zero() {
        let s = ties_source();
        let b = bound_of(&s);
        let mut out = O3Out::new();
        o3(&b, (1 << 2) | (1 << 7), &mut out);
        assert_eq!(out.arg, [2; 64]);
        assert_eq!(out.min, [40; 64]);
        assert_eq!(out.none, 0);
        assert_eq!(out.unknown, 0);
    }

    #[test]
    fn o4_and_o4a_hand_computed() {
        let mut cost = [[0u8; 64]; 64];
        cost[0][3] = 10;
        cost[2][3] = 7;
        let mut w = [0u16; 64];
        w[0] = 600;
        w[2] = 3;
        let mut out = O4Out::new();
        o4(&cost, &w, u64::MAX, &mut out);
        assert_eq!(out.sum[3], 6000 + 21);
        assert_eq!(out.total, 6021);
        o4(&cost, &w, 0, &mut out);
        assert_eq!(out.total, 0);

        let (x, y) = overflow_pair();
        let mut o = O4aOut::new();
        o4a(&x, &y, &mut o);
        assert_eq!(o.z, [[0u16; 64]; 64]);
        assert_eq!(o.ovf, [u64::MAX; 64]);
    }

    #[test]
    fn o5_overflow_leaves_state_unchanged() {
        let mut c: Counters = [[0u16; 64]; 64];
        o5(&mut c, O5Update { k: 2, b: 3, m: 0b101 }).unwrap();
        assert_eq!(c[2][0], 8);
        assert_eq!(c[2][2], 8);
        assert_eq!(c[2][1], 0);

        let mut c: Counters = [[0u16; 64]; 64];
        c[1][1] = 0xFFFF;
        let before = c;
        let r = o5(&mut c, O5Update { k: 1, b: 0, m: 0b11 });
        assert_eq!(r, Err(Refusal::Overflow { lanes: 0b10 }));
        assert_eq!(c, before);

        assert_eq!(
            o5(&mut c, O5Update { k: 0, b: 16, m: 1 }),
            Err(Refusal::SignificanceOutOfRange)
        );

        // Exactly reaching 0xFFFF is fine; one more is not.
        let mut c = carry_counters(4);
        o5(&mut c, O5Update { k: 7, b: 4, m: u64::MAX }).unwrap();
        assert_eq!(c[7][63], 0xFFFF);
        assert!(o5(&mut c, O5Update { k: 7, b: 0, m: 1 }).is_err());
    }

    #[test]
    fn o5_stream_counts() {
        let mut c: Counters = [[0u16; 64]; 64];
        let ups = [
            O5Update { k: 0, b: 15, m: 1 },
            O5Update { k: 0, b: 15, m: 1 }, // 0x8000 + 0x8000 overflows
            O5Update { k: 0, b: 1, m: 1 },
        ];
        let (a, r) = o5_stream(&mut c, &ups);
        assert_eq!((a, r), (2, 1));
        assert_eq!(c[0][0], 0x8002);
    }

    #[test]
    fn o6_limit_cases() {
        for case in limit_cases() {
            let b = bound_of(&case.src);
            let p = Params::new(case.recipe, case.u, 1, [0; 64]).unwrap();
            let mut out = O6Out::new();
            o6(&b, &p, &mut out);
            assert_eq!(out.known, u64::MAX, "{}", case.name);
            for r in 0..64 {
                assert_eq!(out.cap[r], case.cap, "{} cap", case.name);
                assert_eq!(out.limit[r], case.limit, "{} limit", case.name);
            }
        }
    }

    #[test]
    fn o6_store_09_unknown_and_cost() {
        let st = store_09();
        let b = bound_of(&st.src);
        let p = Params::new(st.recipe, 5, 2, [0; 64]).unwrap();
        let mut out = O6Out::new();
        o6(&b, &p, &mut out);
        assert_eq!(out.status[st.row], RowStatus::Unknown { k: st.category as u8 });
        assert_eq!(out.unknown, 1u64 << st.row);
        // Other rows: stock 70 gives 70, 35, 70; staff 3 * 5 = 15 binds.
        assert_eq!(out.cap[0], 15);
        assert_eq!(out.limit[0], STAFF);
        assert_eq!(can_cook(&out, 2), !(1u64 << st.row));
        // cost_r(2) = (1 + 2 + 1) * 2 * 100 = 800 on 63 rows.
        assert_eq!(network_cost(&b, &p, &out, 2), 63 * 800);
    }

    #[test]
    fn o6_denied_and_edge_demands() {
        let (s, r) = stock_below_demand();
        let b = bound_of(&s);
        let p = Params::new(r, 1, 1, [0; 64]).unwrap();
        let mut out = O6Out::new();
        o6(&b, &p, &mut out);
        assert_eq!(out.cap, [0; 64]);

        let (s, r) = empty_demand();
        let b = bound_of(&s);
        let p = Params::new(r, 20, 1, [0; 64]).unwrap();
        o6(&b, &p, &mut out);
        // staff 3 * u 20 = 60, below the ceiling.
        assert_eq!(out.cap, [60; 64]);
        assert_eq!(out.limit, [STAFF; 64]);

        let z = zeros();
        let b = bound_of(&z);
        o6(&b, &p, &mut out);
        assert_eq!(out.denied, u64::MAX);
    }

    #[test]
    fn network_capacity_identity_on_random_source() {
        let s = random_source(42);
        let b = bound_of(&s);
        let p = Params::new(random_recipe(43), 30, 1, [0; 64]).unwrap();
        let mut out = O6Out::new();
        o6(&b, &p, &mut out);
        assert!(out.known != 0, "fixture must have Known rows");
        assert_eq!(network_capacity(&out), network_capacity_by_q(&out));
        assert!(network_capacity(&out) > 0);
    }
}
