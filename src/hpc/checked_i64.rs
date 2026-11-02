//! Checked i64 arithmetic for shape computation (ONNX `Add`, `Sub`, `Mul`,
//! `Div` and `Range` on int64 tensors).
//!
//! One policy covers every operation: a result that does not fit in `i64`
//! is refused, never wrapped and never saturated, and the refusal says which
//! operation, at which output position, on which operands. `Div` truncates
//! toward zero, as ONNX Runtime does for integers; a zero divisor and
//! `i64::MIN / -1` are refused.
//!
//! Operands are flat slices. Each operand has the output's length or length
//! 1 (a scalar that broadcasts); any other length is refused. N-dimensional
//! broadcasting belongs to the caller's graph, not here.
//!
//! Nothing is written on refusal: every element is checked before `out` is
//! touched.

/// The binary operation applied by [`checked_binary_i64`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum I64Op {
    /// `lhs + rhs`.
    Add,
    /// `lhs - rhs`.
    Sub,
    /// `lhs * rhs`.
    Mul,
    /// `lhs / rhs`, truncated toward zero.
    Div,
}

/// Why a checked i64 operation refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I64ArithError {
    /// The exact result of `lhs op rhs` is outside `i64`. Covers
    /// `i64::MIN / -1`.
    Overflow {
        /// The operation.
        op: I64Op,
        /// Output position of the first refused element.
        position: usize,
        /// Left operand at that position.
        lhs: i64,
        /// Right operand at that position.
        rhs: i64,
    },
    /// Division by zero.
    DivisionByZero {
        /// Output position of the first refused element.
        position: usize,
        /// Dividend at that position.
        lhs: i64,
    },
    /// An operand's length is neither the output's length nor 1.
    LengthMismatch {
        /// Length of `lhs`.
        lhs: usize,
        /// Length of `rhs`.
        rhs: usize,
        /// Length of `out`.
        out: usize,
    },
    /// `Range` with `delta == 0`.
    RangeZeroDelta,
    /// `Range` would produce more elements than `out` holds.
    RangeExceedsCapacity {
        /// Number of elements the range has.
        len: u64,
        /// Length of `out`.
        capacity: usize,
    },
}

impl core::fmt::Display for I64ArithError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            I64ArithError::Overflow { op, position, lhs, rhs } => {
                write!(f, "i64 {op:?} overflows at position {position}: {lhs} and {rhs}")
            }
            I64ArithError::DivisionByZero { position, lhs } => {
                write!(f, "i64 Div by zero at position {position}: {lhs} / 0")
            }
            I64ArithError::LengthMismatch { lhs, rhs, out } => {
                write!(f, "i64 operands of length {lhs} and {rhs} do not broadcast to {out}")
            }
            I64ArithError::RangeZeroDelta => write!(f, "i64 Range with delta 0"),
            I64ArithError::RangeExceedsCapacity { len, capacity } => {
                write!(f, "i64 Range has {len} elements, out holds {capacity}")
            }
        }
    }
}

impl std::error::Error for I64ArithError {}

/// Applies `op` to one pair: the single arithmetic policy every slice
/// operation goes through.
///
/// # Errors
///
/// [`I64ArithError::Overflow`] or [`I64ArithError::DivisionByZero`], with
/// `position` 0.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::checked_i64::{checked_i64, I64ArithError, I64Op};
///
/// assert_eq!(checked_i64(I64Op::Div, -7, 2), Ok(-3));
/// assert!(matches!(checked_i64(I64Op::Add, i64::MAX, 1), Err(I64ArithError::Overflow { .. })));
/// ```
pub fn checked_i64(op: I64Op, lhs: i64, rhs: i64) -> Result<i64, I64ArithError> {
    let r = match op {
        I64Op::Add => lhs.checked_add(rhs),
        I64Op::Sub => lhs.checked_sub(rhs),
        I64Op::Mul => lhs.checked_mul(rhs),
        I64Op::Div => {
            if rhs == 0 {
                return Err(I64ArithError::DivisionByZero { position: 0, lhs });
            }
            lhs.checked_div(rhs)
        }
    };
    r.ok_or(I64ArithError::Overflow {
        op,
        position: 0,
        lhs,
        rhs,
    })
}

fn at_position(e: I64ArithError, position: usize) -> I64ArithError {
    match e {
        I64ArithError::Overflow { op, lhs, rhs, .. } => I64ArithError::Overflow { op, position, lhs, rhs },
        I64ArithError::DivisionByZero { lhs, .. } => I64ArithError::DivisionByZero { position, lhs },
        other => other,
    }
}

/// `out[i] = lhs[i] op rhs[i]`, where a length-1 operand broadcasts.
///
/// # Errors
///
/// - [`I64ArithError::LengthMismatch`] if an operand's length is neither
///   `out.len()` nor 1.
/// - [`I64ArithError::Overflow`] / [`I64ArithError::DivisionByZero`] for the
///   first refused position.
///
/// `out` is unchanged when an error is returned.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::checked_i64::{checked_binary_i64, I64Op};
///
/// let mut out = [0i64; 3];
/// checked_binary_i64(I64Op::Mul, &[2, 3, 4], &[-1], &mut out).unwrap();
/// assert_eq!(out, [-2, -3, -4]);
/// ```
pub fn checked_binary_i64(op: I64Op, lhs: &[i64], rhs: &[i64], out: &mut [i64]) -> Result<(), I64ArithError> {
    let n = out.len();
    let fits = |len: usize| len == n || len == 1;
    if !fits(lhs.len()) || !fits(rhs.len()) {
        return Err(I64ArithError::LengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
            out: n,
        });
    }
    let pair = |i: usize| (lhs[if lhs.len() == 1 { 0 } else { i }], rhs[if rhs.len() == 1 { 0 } else { i }]);
    for i in 0..n {
        let (a, b) = pair(i);
        checked_i64(op, a, b).map_err(|e| at_position(e, i))?;
    }
    for (i, o) in out.iter_mut().enumerate() {
        let (a, b) = pair(i);
        // Checked above; cannot fail.
        *o = checked_i64(op, a, b).map_err(|e| at_position(e, i))?;
    }
    Ok(())
}

/// Number of elements of ONNX `Range(start, limit, delta)`:
/// `max(ceil((limit - start) / delta), 0)`, computed exactly.
///
/// # Errors
///
/// [`I64ArithError::RangeZeroDelta`] if `delta == 0`.
pub fn range_len_i64(start: i64, limit: i64, delta: i64) -> Result<u64, I64ArithError> {
    if delta == 0 {
        return Err(I64ArithError::RangeZeroDelta);
    }
    let (diff, step) = (i128::from(limit) - i128::from(start), i128::from(delta));
    // Make the step positive; a range that runs the wrong way is empty.
    let (diff, step) = if step < 0 { (-diff, -step) } else { (diff, step) };
    if diff <= 0 {
        return Ok(0);
    }
    // diff < 2^64 and step >= 1, so the quotient fits in u64.
    Ok(((diff + step - 1) / step) as u64)
}

/// Writes ONNX `Range(start, limit, delta)` into the front of `out` and
/// returns its length. Every element lies between `start` and `limit`, so
/// none overflows.
///
/// # Errors
///
/// - [`I64ArithError::RangeZeroDelta`] if `delta == 0`.
/// - [`I64ArithError::RangeExceedsCapacity`] if the range is longer than
///   `out`; `out` is unchanged.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::checked_i64::range_i64;
///
/// let mut out = [0i64; 8];
/// let n = range_i64(10, 1, -3, &mut out).unwrap();
/// assert_eq!(&out[..n], &[10, 7, 4]);
/// ```
pub fn range_i64(start: i64, limit: i64, delta: i64, out: &mut [i64]) -> Result<usize, I64ArithError> {
    let len = range_len_i64(start, limit, delta)?;
    let n = match usize::try_from(len) {
        Ok(n) if n <= out.len() => n,
        _ => {
            return Err(I64ArithError::RangeExceedsCapacity {
                len,
                capacity: out.len(),
            })
        }
    };
    for (i, o) in out[..n].iter_mut().enumerate() {
        // start + i*delta lies in [start, limit) (or (limit, start]), so it fits.
        *o = (i128::from(start) + i128::from(delta) * i as i128) as i64;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: [I64Op; 4] = [I64Op::Add, I64Op::Sub, I64Op::Mul, I64Op::Div];

    /// Exact result in i128, the reference the checked policy must agree with.
    fn exact(op: I64Op, a: i64, b: i64) -> Option<i128> {
        let (a, b) = (i128::from(a), i128::from(b));
        match op {
            I64Op::Add => Some(a + b),
            I64Op::Sub => Some(a - b),
            I64Op::Mul => Some(a * b),
            I64Op::Div => (b != 0).then(|| a / b),
        }
    }

    const EDGES: [i64; 13] = [
        i64::MIN,
        i64::MIN + 1,
        -(1 << 32),
        -3_037_000_500,
        -7,
        -1,
        0,
        1,
        2,
        7,
        3_037_000_500,
        1 << 32,
        i64::MAX,
    ];

    #[test]
    fn every_edge_pair_matches_exact_arithmetic_or_refuses() {
        let (mut ok, mut refused) = (0, 0);
        for op in OPS {
            for a in EDGES {
                for b in EDGES {
                    match (exact(op, a, b), checked_i64(op, a, b)) {
                        (Some(e), Ok(r)) => {
                            assert_eq!(i128::from(r), e, "{op:?} {a} {b}");
                            ok += 1;
                        }
                        (Some(e), Err(I64ArithError::Overflow { op: o, lhs, rhs, .. })) => {
                            assert!(i64::try_from(e).is_err(), "{op:?} {a} {b} refused but fits");
                            assert_eq!((o, lhs, rhs), (op, a, b));
                            refused += 1;
                        }
                        (None, Err(I64ArithError::DivisionByZero { lhs, .. })) => {
                            assert_eq!((op, lhs, b), (I64Op::Div, a, 0));
                            refused += 1;
                        }
                        other => panic!("{op:?} {a} {b}: {other:?}"),
                    }
                }
            }
        }
        // Anti-vacuity: the edge set exercises both outcomes.
        assert!(ok > 100 && refused > 40, "ok={ok} refused={refused}");
    }

    #[test]
    fn div_truncates_toward_zero() {
        let cases = [
            (7, 2, 3),
            (-7, 2, -3),
            (7, -2, -3),
            (-7, -2, 3),
            (i64::MIN, 1, i64::MIN),
            (i64::MIN, 2, i64::MIN / 2),
        ];
        for (a, b, want) in cases {
            assert_eq!(checked_i64(I64Op::Div, a, b), Ok(want), "{a} / {b}");
        }
        // Floor division would give -4 here; truncation gives -3.
        assert_ne!(checked_i64(I64Op::Div, -7, 2), Ok((-7_i64).div_euclid(2)));
    }

    /// Negative fixtures for the Tract integration, one per refusal kind.
    #[test]
    fn negative_fixtures_refuse_with_their_information() {
        let mut out = [0i64; 1];
        let cases: [(I64Op, i64, i64, I64ArithError); 6] = [
            (
                I64Op::Add,
                i64::MAX,
                1,
                I64ArithError::Overflow {
                    op: I64Op::Add,
                    position: 0,
                    lhs: i64::MAX,
                    rhs: 1,
                },
            ),
            (
                I64Op::Sub,
                i64::MIN,
                1,
                I64ArithError::Overflow {
                    op: I64Op::Sub,
                    position: 0,
                    lhs: i64::MIN,
                    rhs: 1,
                },
            ),
            (
                I64Op::Mul,
                3_037_000_500,
                3_037_000_500,
                I64ArithError::Overflow {
                    op: I64Op::Mul,
                    position: 0,
                    lhs: 3_037_000_500,
                    rhs: 3_037_000_500,
                },
            ),
            (
                I64Op::Mul,
                i64::MIN,
                -1,
                I64ArithError::Overflow {
                    op: I64Op::Mul,
                    position: 0,
                    lhs: i64::MIN,
                    rhs: -1,
                },
            ),
            (
                I64Op::Div,
                i64::MIN,
                -1,
                I64ArithError::Overflow {
                    op: I64Op::Div,
                    position: 0,
                    lhs: i64::MIN,
                    rhs: -1,
                },
            ),
            (I64Op::Div, 5, 0, I64ArithError::DivisionByZero { position: 0, lhs: 5 }),
        ];
        for (op, a, b, want) in cases {
            assert_eq!(checked_binary_i64(op, &[a], &[b], &mut out), Err(want));
        }
    }

    #[test]
    fn slices_broadcast_length_one_and_report_the_failing_position() {
        let mut out = [0i64; 4];
        checked_binary_i64(I64Op::Sub, &[10], &[1, 2, 3, 4], &mut out).unwrap();
        assert_eq!(out, [9, 8, 7, 6]);
        checked_binary_i64(I64Op::Div, &[9, -9, 8, 1], &[2], &mut out).unwrap();
        assert_eq!(out, [4, -4, 4, 0]);
        // Position 2 overflows; nothing is written.
        let before = [111, 222, 333, 444];
        out = before;
        let err = checked_binary_i64(I64Op::Add, &[1, 2, i64::MAX, 4], &[1], &mut out);
        assert_eq!(
            err,
            Err(I64ArithError::Overflow {
                op: I64Op::Add,
                position: 2,
                lhs: i64::MAX,
                rhs: 1
            })
        );
        assert_eq!(out, before, "out written on refusal");
        let err = checked_binary_i64(I64Op::Div, &[6, 6, 6, 6], &[3, 2, 1, 0], &mut out);
        assert_eq!(err, Err(I64ArithError::DivisionByZero { position: 3, lhs: 6 }));
        assert_eq!(out, before, "out written on refusal");
    }

    #[test]
    fn unsupported_broadcasting_refuses() {
        let mut out = [0i64; 4];
        let err = checked_binary_i64(I64Op::Add, &[1, 2], &[1, 2, 3, 4], &mut out);
        assert_eq!(err, Err(I64ArithError::LengthMismatch { lhs: 2, rhs: 4, out: 4 }));
        let err = checked_binary_i64(I64Op::Add, &[1, 2, 3, 4], &[1, 2, 3, 4], &mut out[..3]);
        assert_eq!(err, Err(I64ArithError::LengthMismatch { lhs: 4, rhs: 4, out: 3 }));
        // Empty output with scalar operands is a valid broadcast.
        assert_eq!(checked_binary_i64(I64Op::Mul, &[2], &[3], &mut []), Ok(()));
    }

    /// ONNX Range reference: `start + i*delta` while it is short of `limit`.
    fn range_reference(start: i64, limit: i64, delta: i64) -> Vec<i64> {
        let mut v = Vec::new();
        let mut x = i128::from(start);
        while (delta > 0 && x < i128::from(limit)) || (delta < 0 && x > i128::from(limit)) {
            v.push(x as i64);
            x += i128::from(delta);
        }
        v
    }

    #[test]
    fn range_matches_the_onnx_definition() {
        let mut out = [0i64; 64];
        for start in -9..=9 {
            for limit in -9..=9 {
                for delta in [-5, -3, -1, 1, 2, 4, 7] {
                    let want = range_reference(start, limit, delta);
                    let n = range_i64(start, limit, delta, &mut out).unwrap();
                    assert_eq!(&out[..n], &want[..], "Range({start}, {limit}, {delta})");
                }
            }
        }
        // Wrong direction is empty, not an error.
        assert_eq!(range_i64(5, 1, 1, &mut out), Ok(0));
        assert_eq!(range_i64(1, 5, -1, &mut out), Ok(0));
        // Near the ends of i64: no overflow on the last element.
        let n = range_i64(i64::MAX - 5, i64::MAX, 2, &mut out).unwrap();
        assert_eq!(&out[..n], &[i64::MAX - 5, i64::MAX - 3, i64::MAX - 1]);
        let n = range_i64(i64::MIN + 5, i64::MIN, -2, &mut out).unwrap();
        assert_eq!(&out[..n], &[i64::MIN + 5, i64::MIN + 3, i64::MIN + 1]);
    }

    #[test]
    fn range_refuses_zero_delta_and_overlong_output() {
        let mut out = [7i64; 4];
        assert_eq!(range_i64(0, 10, 0, &mut out), Err(I64ArithError::RangeZeroDelta));
        assert_eq!(range_i64(0, 5, 1, &mut out), Err(I64ArithError::RangeExceedsCapacity { len: 5, capacity: 4 }));
        assert_eq!(out, [7; 4], "out written on refusal");
        // Full i64 span: the length is computed exactly, not wrapped.
        assert_eq!(range_len_i64(i64::MIN, i64::MAX, 1), Ok(u64::MAX));
        assert_eq!(
            range_i64(i64::MIN, i64::MAX, 1, &mut out),
            Err(I64ArithError::RangeExceedsCapacity {
                len: u64::MAX,
                capacity: 4
            })
        );
        // Exactly at capacity is accepted.
        assert_eq!(range_i64(0, 4, 1, &mut out), Ok(4));
        assert_eq!(out, [0, 1, 2, 3]);
    }
}
