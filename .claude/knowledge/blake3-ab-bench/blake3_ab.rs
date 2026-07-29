//! A/B: ndarray's in-tree BLAKE3 vs the external `blake3` crate, on the input
//! sizes ndarray actually hashes. Throwaway; deleted after the measurement.
use std::time::Instant;

fn bench<F: FnMut() -> [u8; 32]>(name: &str, iters: u32, mut f: F) -> f64 {
    let mut sink = 0u8;
    for _ in 0..iters / 10 { sink ^= f()[0]; }          // warm
    let t = Instant::now();
    for _ in 0..iters { sink ^= f()[0]; }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(sink);
    println!("    {name:<12} {ns:>10.1} ns/op");
    ns
}

fn main() {
    // 16 B ~ a word (crystal_encoder); 256 B ~ text; 2 KB ~ VSA_BYTES; 64 KB ~ bulk.
    for &n in &[16usize, 256, 2048, 65536] {
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let iters = if n >= 65536 { 2_000 } else { 50_000 };
        println!("  input = {n} B");
        let ours = bench("in-tree", iters, || *ndarray::hpc::blake3::hash(&data).as_bytes());
        let theirs = bench("blake3 crate", iters, || *blake3::hash(&data).as_bytes());
        println!("    ratio        {:>10.2}x  (in-tree / crate)\n", ours / theirs);
    }
}
