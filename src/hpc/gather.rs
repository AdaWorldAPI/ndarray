//! Row gather from a table into a caller buffer (ONNX `Gather` on axis 0,
//! e.g. an embedding lookup).
//!
//! The table is never copied: each selected row is copied once, straight
//! into its row of `out`. Indices follow the ONNX rule: an index `i` is valid
//! when `-rows <= i < rows`, and a negative index counts from the end. An
//! index outside that range is reported as an error, never a panic and never
//! a clamp.

use crate::{ArrayView2, ArrayViewMut2, Axis};

/// Why [`gather_rows_f32`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherError {
    /// `ids[position]` is outside `[-rows, rows)`.
    IndexOutOfRange {
        /// Position in `ids`.
        position: usize,
        /// The offending index.
        index: i64,
        /// Number of rows in the table.
        rows: usize,
    },
    /// `out` is not `[ids.len(), table columns]`.
    ShapeMismatch {
        /// Expected `out` shape.
        expected: (usize, usize),
        /// Actual `out` shape.
        actual: (usize, usize),
    },
}

impl core::fmt::Display for GatherError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            GatherError::IndexOutOfRange { position, index, rows } => {
                write!(f, "gather: ids[{position}] = {index} is outside [-{rows}, {rows})")
            }
            GatherError::ShapeMismatch { expected, actual } => {
                write!(f, "gather: out is {actual:?}, expected {expected:?}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GatherError {}

/// `out[r] = table[ids[r]]` for every `r`, with ONNX index rules.
///
/// Every index is checked before anything is written, so on an error `out`
/// is left unchanged. Contiguous rows are copied with `copy_from_slice`;
/// other layouts are copied element by element through the views.
///
/// # Errors
///
/// [`GatherError::IndexOutOfRange`] for an index outside `[-rows, rows)`,
/// [`GatherError::ShapeMismatch`] if `out` is not `[ids.len(), columns]`.
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use ndarray::hpc::gather::gather_rows_f32;
///
/// let table = array![[0.0f32, 0.5], [1.0, 1.5], [2.0, 2.5]];
/// let mut out = Array2::<f32>::zeros((3, 2));
/// gather_rows_f32(table.view(), &[2, 0, -1], out.view_mut()).unwrap();
/// assert_eq!(out, array![[2.0, 2.5], [0.0, 0.5], [2.0, 2.5]]);
/// assert!(gather_rows_f32(table.view(), &[3], out.view_mut()).is_err());
/// ```
pub fn gather_rows_f32(
    table: ArrayView2<'_, f32>, ids: &[i64], mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), GatherError> {
    let (rows, cols) = table.dim();
    let expected = (ids.len(), cols);
    if out.dim() != expected {
        return Err(GatherError::ShapeMismatch {
            expected,
            actual: out.dim(),
        });
    }
    let row_of = |position: usize, index: i64| -> Result<usize, GatherError> {
        let err = GatherError::IndexOutOfRange { position, index, rows };
        let r = i128::from(index);
        let n = rows as i128;
        if r < -n || r >= n {
            return Err(err);
        }
        Ok(if r < 0 { (r + n) as usize } else { r as usize })
    };
    // Validate everything first, so a bad index leaves `out` untouched.
    for (p, &i) in ids.iter().enumerate() {
        row_of(p, i)?;
    }
    for (p, &i) in ids.iter().enumerate() {
        let src = table.index_axis(Axis(0), row_of(p, i)?);
        let mut dst = out.index_axis_mut(Axis(0), p);
        match (src.as_slice(), dst.as_slice_mut()) {
            (Some(s), Some(d)) => d.copy_from_slice(s),
            _ => dst.assign(&src),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{s, Array2, ShapeBuilder};

    #[test]
    fn gathers_rows_with_onnx_negative_indices_in_any_layout() {
        let table = Array2::from_shape_fn((5, 3), |(i, j)| (i * 10 + j) as f32);
        let ids = [4i64, 0, -1, -5, 2];
        let want = Array2::from_shape_fn((5, 3), |(r, j)| [4, 0, 4, 0, 2][r] as f32 * 10.0 + j as f32);
        let mut out = Array2::<f32>::zeros((5, 3));
        gather_rows_f32(table.view(), &ids, out.view_mut()).unwrap();
        assert_eq!(out, want);
        // A transposed (non-contiguous rows) table takes the view path.
        let tf = Array2::from_shape_fn((5, 3).f(), |(i, j)| (i * 10 + j) as f32);
        let tv = tf.view();
        assert!(tv.index_axis(Axis(0), 0).as_slice().is_none(), "fixture must take the view path");
        let mut out2 = Array2::<f32>::zeros((5, 3));
        gather_rows_f32(tv, &ids, out2.view_mut()).unwrap();
        assert_eq!(out2, want);
        // A stepped `out` view.
        let mut big = Array2::<f32>::from_elem((10, 3), -9.0);
        gather_rows_f32(table.view(), &ids, big.slice_mut(s![..;2, ..])).unwrap();
        assert_eq!(big.slice(s![..;2, ..]), want);
        assert!(big.slice(s![1..;2, ..]).iter().all(|&v| v == -9.0));
    }

    #[test]
    fn refuses_out_of_range_indices_and_leaves_out_untouched() {
        let table = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f32);
        for (ids, bad_pos, bad) in [
            (vec![0i64, 5], 1usize, 5i64),
            (vec![-6i64], 0, -6),
            (vec![1, i64::MIN], 1, i64::MIN),
            (vec![i64::MAX], 0, i64::MAX),
        ] {
            let mut out = Array2::<f32>::from_elem((ids.len(), 2), 7.0);
            assert_eq!(
                gather_rows_f32(table.view(), &ids, out.view_mut()),
                Err(GatherError::IndexOutOfRange {
                    position: bad_pos,
                    index: bad,
                    rows: 5
                })
            );
            assert!(out.iter().all(|&v| v == 7.0), "out written before the error: {out:?}");
        }
        let mut wrong = Array2::<f32>::zeros((2, 3));
        assert!(matches!(
            gather_rows_f32(table.view(), &[0, 1], wrong.view_mut()),
            Err(GatherError::ShapeMismatch { .. })
        ));
        // An empty table refuses every index, including 0 and -0.
        let empty = Array2::<f32>::zeros((0, 2));
        let mut o = Array2::<f32>::zeros((1, 2));
        assert!(gather_rows_f32(empty.view(), &[0], o.view_mut()).is_err());
    }
}
