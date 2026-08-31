//! Pre-registration sweep: how does the hardware-vs-reference error scale,
//! and under WHICH normalization is it a stable gate?
use ndarray::hpc::pillar::signature::signature_d2_deg3;
use sigker::signature_truncated;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

// level of each of the 15 coefficients
const LEVEL: [usize; 15] = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];

fn main() {
    println!("{:>6} {:>12} {:>14} {:>16}", "N", "worst |abs|", "worst /coeff", "worst /levelmax");
    for &n in &[16usize, 32, 64, 128, 256] {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut wa, mut wc, mut wl) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..1000 {
            let (mut x, mut y) = (0.0f32, 0.0f32);
            let mut flat = Vec::with_capacity(n * 2);
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                flat.push(x);
                flat.push(y);
                pts.push(vec![x as f64, y as f64]);
                x += rng.f();
                y += rng.f();
            }
            let hw = signature_d2_deg3(&flat, n);
            let refr: Vec<f64> = signature_truncated(&pts, 3)
                .levels
                .iter()
                .flat_map(|l| l.iter().copied())
                .collect();
            // characteristic magnitude per level, from the REFERENCE
            let mut lvmax = [0.0f64; 4];
            for i in 0..15 {
                lvmax[LEVEL[i]] = lvmax[LEVEL[i]].max(refr[i].abs());
            }
            for i in 0..15 {
                let d = (hw[i] as f64 - refr[i]).abs();
                wa = wa.max(d);
                if refr[i].abs() > 1e-12 {
                    wc = wc.max(d / refr[i].abs());
                }
                let s = lvmax[LEVEL[i]].max(1e-12);
                wl = wl.max(d / s);
            }
        }
        println!("{n:>6} {wa:>12.3e} {wc:>14.3e} {wl:>16.3e}");
    }
}
