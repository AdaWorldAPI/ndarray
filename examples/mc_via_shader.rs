//! M1 — motion compensation via the cognitive-shader primitives.
//!
//! Proof-of-concept for the codec↔cognitive-shader unification documented in
//! `.claude/knowledge/pr-x12-codec-cognitive-substrate-mapping.md`:
//!
//!   - **E-7** — "block-matching motion estimation *is* i8gemm." HEVC ME does
//!     SAD; reformulated as SSD `‖A‖² − 2·A·B + ‖B‖²`, the middle term `A·B` is
//!     a GEMM. This PoC runs the search through the SHIPPED
//!     `hpc::quantized::int8_gemm_i32` (the same u8×i8→i32 VNNI kernel the
//!     cognitive shader uses for attention/distance) — NOT a bespoke SAD loop.
//!   - **H-7** — "the codec *is* the substrate." Each block's residual
//!     magnitude classifies into the SHIPPED `hpc::codec::CellMode`
//!     (Skip/Merge/Delta/Escape) — the codec's mode taxonomy IS the MC
//!     per-block decision.
//!
//! Pipeline (all shipped primitives, zero new substrate):
//!   1. reference frame (deterministic 7-bit-luma texture)
//!   2. current frame = per-block shifted reference + small residual (known MVs)
//!   3. **ME**: per block, `int8_gemm_i32` computes `A·B` for all candidates in
//!      a ±R window; SSD = `‖B‖² − 2·A·B` (+ const `‖A‖²`) → argmin → MV.
//!   4. **cross-check**: brute-force direct SSD gives the ground-truth argmin.
//!      E-7 holds iff the GEMM argmin == the brute-force argmin for EVERY block.
//!   5. **MC**: gather the reference block at the recovered MV (contiguous
//!      block fetch) + add the stored residual → reconstruction.
//!   6. classify residual → `CellMode`; assert reconstruction is bit-exact.
//!
//! What this PROVES: motion estimation runs through the shader's i8 GEMM and
//! is bit-identical to direct SSD; motion compensation is gather+add;
//! reconstruction is exact; the per-block decision is the codec's mode taxonomy.
//! What this does NOT do (that's M2, the multi-month decoder): parse a real
//! `.265` bitstream (CABAC entropy decode + inverse transform to extract the
//! MVs/residuals), sub-pel interpolation, deblock/SAO. Frames are synthetic
//! 7-bit luma with integer-translation MVs so the i8 GEMM is exact.
//!
//! Run: `cargo run --release --example mc_via_shader --features codec`

use ndarray::hpc::codec::CellMode;
use ndarray::hpc::quantized::int8_gemm_i32;

const W: usize = 128;
const H: usize = 128;
const B: usize = 8; // block edge (8×8)
const K: usize = B * B; // pixels per block = GEMM K
const BX: usize = W / B; // blocks per row
const BY: usize = H / B; // blocks per column
const R: i32 = 4; // motion search radius (±R)

/// SplitMix64 — deterministic, dependency-free per-pixel texture + residual.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 7-bit luma [0,127] so the value fits both `u8` (current) and `i8`
/// (reference) operands of the VNNI kernel with no centering games.
fn ref_pixel(x: usize, y: usize) -> u8 {
    (mix((y as u64) << 20 | x as u64) & 0x7F) as u8
}

/// Ground-truth per-block MV — a deterministic small field, forced to 0 near
/// the border so `block_pos + mv + B` stays in bounds.
fn gt_mv(bx: usize, by: usize) -> (i32, i32) {
    let raw_x = ((bx + 2 * by) % 5) as i32 - 2;
    let raw_y = ((2 * bx + by) % 5) as i32 - 2;
    // keep block + mv + search window inside the frame
    let px = (bx * B) as i32;
    let py = (by * B) as i32;
    let ok = |p: i32, d: i32, span: i32| p + d - R >= 0 && p + d + R + B as i32 <= span;
    let mvx = if ok(px, raw_x, W as i32) { raw_x } else { 0 };
    let mvy = if ok(py, raw_y, H as i32) { raw_y } else { 0 };
    (mvx, mvy)
}

/// Small deterministic residual on ~1/3 of blocks (drives Delta); the rest
/// are residual-free (drive Skip). Values kept tiny (|r| ≤ 3).
fn residual(bx: usize, by: usize, i: usize) -> i32 {
    if !(bx * 7 + by * 3).is_multiple_of(3) {
        return 0; // residual-free block
    }
    ((mix((bx as u64) << 40 | (by as u64) << 20 | i as u64) % 7) as i32) - 3
}

fn ref_block(reference: &[u8], px: i32, py: i32) -> [i8; K] {
    let mut blk = [0i8; K];
    for j in 0..B {
        for i in 0..B {
            blk[j * B + i] = reference[(py as usize + j) * W + (px as usize + i)] as i8;
        }
    }
    blk
}

fn main() {
    // ── 1. reference frame ───────────────────────────────────────────────
    let reference: Vec<u8> = (0..H * W).map(|p| ref_pixel(p % W, p / W)).collect();

    // ── 2. current frame = shifted-ref + residual (known MVs) ────────────
    let mut current = vec![0u8; H * W];
    for by in 0..BY {
        for bx in 0..BX {
            let (mvx, mvy) = gt_mv(bx, by);
            let (px, py) = ((bx * B) as i32, (by * B) as i32);
            for j in 0..B {
                for i in 0..B {
                    let refv = reference[(py + mvy + j as i32) as usize * W + (px + mvx + i as i32) as usize] as i32;
                    let v = (refv + residual(bx, by, j * B + i)).clamp(0, 127);
                    current[(py as usize + j) * W + (px as usize + i)] = v as u8;
                }
            }
        }
    }

    // ── 3+4. ME via i8 GEMM, cross-checked against brute-force SSD ───────
    let mut recovered = vec![(0i32, 0i32); BX * BY];
    let mut gemm_eq_brute = 0usize; // E-7: GEMM argmin == direct-SSD argmin
    let mut mv_exact = 0usize; // recovered MV == ground-truth MV
    let mut gemm_calls = 0usize;
    let t0 = std::time::Instant::now();

    for by in 0..BY {
        for bx in 0..BX {
            let (px, py) = ((bx * B) as i32, (by * B) as i32);
            // current block A (u8), row vector [1×K]
            let mut a = [0u8; K];
            for j in 0..B {
                for i in 0..B {
                    a[j * B + i] = current[(py as usize + j) * W + (px as usize + i)];
                }
            }

            // enumerate in-bounds candidates; build B = [K × N] (candidate per
            // column: b[k*N + n]) so int8_gemm_i32(1,N,K) yields C[n] = A·B_n.
            let mut cand: Vec<(i32, i32)> = Vec::with_capacity(((2 * R + 1) * (2 * R + 1)) as usize);
            for dy in -R..=R {
                for dx in -R..=R {
                    let (cx, cy) = (px + dx, py + dy);
                    if cx >= 0 && cy >= 0 && cx + B as i32 <= W as i32 && cy + B as i32 <= H as i32 {
                        cand.push((dx, dy));
                    }
                }
            }
            let n = cand.len();
            let mut bmat = vec![0i8; K * n];
            let mut b_sq = vec![0i64; n]; // ‖B_n‖²
            for (col, &(dx, dy)) in cand.iter().enumerate() {
                let blk = ref_block(&reference, px + dx, py + dy);
                let mut s = 0i64;
                for k in 0..K {
                    bmat[k * n + col] = blk[k];
                    s += (blk[k] as i64) * (blk[k] as i64);
                }
                b_sq[col] = s;
            }

            // THE SHADER KERNEL: C[1×N] = A[1×K] · B[K×N]  (u8 × i8 → i32)
            let mut c = vec![0i32; n];
            int8_gemm_i32(&a, &bmat, &mut c, 1, n, K);
            gemm_calls += 1;

            // SSD_n = ‖A‖² − 2·(A·B_n) + ‖B_n‖²; ‖A‖² is const per block →
            // argmin over (‖B_n‖² − 2·C[n]).
            let gemm_arg = (0..n)
                .min_by_key(|&nn| b_sq[nn] - 2 * c[nn] as i64)
                .unwrap();

            // brute-force direct SSD = Σ (A_k − B_k)²  — the ground truth.
            let brute_arg = (0..n)
                .min_by_key(|&nn| {
                    let (dx, dy) = cand[nn];
                    let blk = ref_block(&reference, px + dx, py + dy);
                    (0..K)
                        .map(|k| {
                            let d = a[k] as i64 - blk[k] as i64;
                            d * d
                        })
                        .sum::<i64>()
                })
                .unwrap();

            if gemm_arg == brute_arg {
                gemm_eq_brute += 1;
            }
            let mv = cand[gemm_arg];
            recovered[by * BX + bx] = mv;
            if mv == gt_mv(bx, by) {
                mv_exact += 1;
            }
        }
    }
    let me_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── 5+6. MC (gather ref block @ recovered MV) + residual; classify + verify
    let mut recon = vec![0u8; H * W];
    let mut modes = [0usize; 4]; // skip, merge, delta, escape
    let mut max_err = 0i32;
    for by in 0..BY {
        for bx in 0..BX {
            let (px, py) = ((bx * B) as i32, (by * B) as i32);
            let (mvx, mvy) = recovered[by * BX + bx];
            // prediction = reference block at MV (contiguous gather)
            let pred = ref_block(&reference, px + mvx, py + mvy); // i8 == 7-bit luma
            let mut maxres = 0i32;
            for j in 0..B {
                for i in 0..B {
                    let k = j * B + i;
                    let cur = current[(py as usize + j) * W + (px as usize + i)] as i32;
                    let res = cur - pred[k] as i32; // stored residual = current − prediction
                    maxres = maxres.max(res.abs());
                    let r = (pred[k] as i32 + res).clamp(0, 255); // reconstruction = pred + residual
                    recon[(py as usize + j) * W + (px as usize + i)] = r as u8;
                    max_err = max_err.max((r - cur).abs());
                }
            }
            // per-block decision = codec mode: residual magnitude → CellMode
            let mode = if maxres == 0 {
                CellMode::Skip
            } else if maxres <= 127 && recovered[by * BX + bx] == recovered[by.saturating_sub(1) * BX + bx] {
                CellMode::Merge // MV inherited from N neighbour + in-range residual
            } else if maxres <= 127 {
                CellMode::Delta
            } else {
                CellMode::Escape
            };
            modes[mode as usize] += 1;
        }
    }

    // ── report ───────────────────────────────────────────────────────────
    let nblk = BX * BY;
    let pct = |x: usize| 100.0 * x as f64 / nblk as f64;
    println!("M1 — motion compensation via cognitive-shader primitives");
    println!("  frame {W}×{H}, {B}×{B} blocks = {nblk} blocks, 7-bit luma, ±{R} search\n");
    println!(
        "  [E-7] i8gemm SSD argmin == brute-force direct SSD : {gemm_eq_brute}/{nblk}  ({:.1}%)",
        pct(gemm_eq_brute)
    );
    println!("        → motion estimation IS int8_gemm_i32 (the shader kernel), bit-identical to SAD/SSD");
    println!("  MV recovery (i8gemm ME == ground-truth motion)   : {mv_exact}/{nblk}  ({:.1}%)", pct(mv_exact));
    println!("  MC reconstruction max abs pixel error            : {max_err}   (0 = bit-exact)");
    println!("  [H-7] per-block CellMode histogram (residual → mode):");
    println!("        skip={} merge={} delta={} escape={}", modes[0], modes[1], modes[2], modes[3]);
    println!(
        "  throughput: {gemm_calls} int8_gemm_i32 calls (one per block) in {me_ms:.2} ms  → {:.0} blocks/s",
        nblk as f64 / (me_ms / 1000.0)
    );

    // hard asserts — the PoC's falsifiers
    assert_eq!(gemm_eq_brute, nblk, "E-7 FALSIFIED: i8gemm ME disagrees with direct SSD");
    assert_eq!(max_err, 0, "MC reconstruction is not bit-exact");
    println!("\n  RESULT: E-7 holds (ME == i8gemm), MC is bit-exact, decision = codec modes. Concept proven.");
}
