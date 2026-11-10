//! D-U64X64-LLVM-1 route E ("prepared").
//!
//! Work done once per binding and reused across queries, with a full witness
//! so a stale reuse is refused rather than silently answered.
//!
//! What is prepared: the bit-sliced stock planes for ALL 64 categories
//! (`planes[k]` = 16 bit-planes of 64 rows, built by `route_b::planes_u16`).
//! They depend only on the stock field of one frame, not on the recipe or on
//! q, so one `Prepared` serves every query against that frame.
//!
//! What invalidates it, and the refusal each gives (checked in this order):
//!
//! - different source          -> `ForeignSource`
//! - version differs           -> `Stale`        (E1)
//! - row order differs         -> `RowOrder`     (E2, hash and array)
//! - stock field not in frame  -> `Field`        (E3)
//!
//! E1 preparation (narrowing): an i32 source is narrowed to u16 once, which
//! refuses `Negative` and `TooLarge`; `o2_e1` times narrowing + bind + route C.
//!
//! The memo (`MemoO6`) is the object of disable run E4: its key must cover
//! the frame and the full parameters, otherwise a hit answers a different
//! query. A hit never differs from a fresh `o6_e2` for the same inputs.
//!
//! Disable-run anchors are marked `// E<n>:` (plan §6). No allocation inside
//! any query function; no shuffles, no i8, no unsafe.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::fixture::*;
use crate::route_b;

/// Bit-sliced stock planes for one frame, with the witness they came from.
pub struct Prepared {
    witness: Witness,
    field: u64,
    planes: [[u64; 16]; 64],
}

/// Build the stock planes for every category. The only place planes are built.
pub fn prepare_e2(bound: &Bound<'_>) -> Prepared {
    let mut planes = [[0u64; 16]; 64];
    for k in 0..64 {
        planes[k] = route_b::planes_u16(&bound.src.stock[k]);
    }
    Prepared {
        witness: bound.witness,
        field: F_STOCK,
        planes,
    }
}

impl Prepared {
    /// Refuse a reuse against a frame other than the one prepared from.
    pub fn check(&self, bound: &Bound<'_>) -> Result<(), Refusal> {
        let w = &bound.witness;
        if w.source_id != self.witness.source_id {
            return Err(Refusal::ForeignSource);
        }
        // E1: the version must match the prepared one.
        if w.version != self.witness.version {
            return Err(Refusal::Stale);
        }
        // E2: the row order must match, by hash and by array.
        if w.row_order_hash != self.witness.row_order_hash || w.row_order != self.witness.row_order {
            return Err(Refusal::RowOrder);
        }
        // E3: the prepared field must be present in the frame.
        if w.fields & self.field != self.field {
            return Err(Refusal::Field);
        }
        Ok(())
    }

    /// Witness the planes were built from.
    pub fn witness(&self) -> &Witness {
        &self.witness
    }
}

/// O2 on prepared planes.
#[inline(never)]
pub fn o2_e2(p: &Prepared, bound: &Bound<'_>, t: &[u32; 64], demanded: u64, out: &mut O2Out) -> Result<(), Refusal> {
    p.check(bound)?;
    route_b::o2_on_planes(&p.planes, t, demanded, out);
    Ok(())
}

/// O6 on prepared planes.
#[inline(never)]
pub fn o6_e2(p: &Prepared, bound: &Bound<'_>, params: &Params, out: &mut O6Out) -> Result<(), Refusal> {
    p.check(bound)?;
    route_b::o6_on_planes(&p.planes, bound, params, out);
    Ok(())
}

/// E1 preparation: narrow the i32 source (refuses `Negative`, then `TooLarge`).
pub fn prepare_e1(src: &SourceI32) -> Result<Source, Refusal> {
    Source::from_i32(src)
}

/// "E1 + C on i32": narrow, bind, then route C's O2, all inside the timed call.
#[inline(never)]
pub fn o2_e1(src: &SourceI32, expect: &Expect, t: &[u32; 64], demanded: u64, out: &mut O2Out) -> Result<(), Refusal> {
    let narrowed = prepare_e1(src)?;
    let bound = bind(&narrowed, expect)?;
    crate::route_c::u64x64_o2_c(&bound, t, demanded, out);
    Ok(())
}

/// Everything an O6 answer depends on besides the (checked) prepared planes:
/// the frame and the full parameters (recipe, u, q, t).
#[derive(Clone, Copy, PartialEq, Eq)]
struct MemoKey {
    source_id: u64,
    version: u64,
    row_order: [u16; 64],
    params: Params,
}

impl MemoKey {
    /// A hit requires the same frame AND the same parameters.
    fn hits(&self, other: &MemoKey) -> bool {
        let same_frame =
            self.source_id == other.source_id && self.version == other.version && self.row_order == other.row_order;
        // E4: the parameters are part of the key.
        same_frame && self.params == other.params
    }
}

/// One-entry memo of the last O6 answer (disable run E4).
pub struct MemoO6 {
    last: Option<(MemoKey, O6Out)>,
}

impl MemoO6 {
    pub fn new() -> Self {
        MemoO6 { last: None }
    }

    /// Answer from the memo on an identical (frame, params) key, else compute
    /// with `o6_e2` and remember. The prepared check runs on every call, so a
    /// stale frame is refused even when the key would match.
    pub fn get_or_run(
        &mut self, p: &Prepared, bound: &Bound<'_>, params: &Params, out: &mut O6Out,
    ) -> Result<(), Refusal> {
        p.check(bound)?;
        let key = MemoKey {
            source_id: bound.witness.source_id,
            version: bound.witness.version,
            row_order: bound.witness.row_order,
            params: *params,
        };
        if let Some((k, cached)) = &self.last {
            if k.hits(&key) {
                *out = *cached;
                return Ok(());
            }
        }
        o6_e2(p, bound, params, out)?;
        self.last = Some((key, *out));
        Ok(())
    }
}
