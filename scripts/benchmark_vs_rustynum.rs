use std::time::Instant;
use std::hint::black_box;

fn median_ns(times: &mut Vec<u64>) -> u64 {
    times.sort_unstable();
    times[times.len() / 2]
}

fn bench_fn<F: FnMut()>(iters: usize, mut f: F) -> u64 {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters.min(100) { f(); }
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_nanos() as u64);
    }
    median_ns(&mut times)
}

fn gen_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 13) % 1000) as f32 * 0.001).collect()
}

fn gen_u8(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 7 + 13) % 256) as u8).collect()
}

fn test1_blas1() {
    println!("\n═══ TEST 1: BLAS Level 1 — dot_f32 ═══");
    println!("{:<12} {:>12} {:>12} {:>12} {:>12}",
        "SIZE", "RUSTYNUM", "NDARRAY", "ND/RN", "STATUS");

    for &n in &[256, 1024, 4096, 16384, 65536, 262144, 1048576] {
        let a = gen_f32(n);
        let b = gen_f32(n);
        let iters = if n <= 4096 { 10000 } else if n <= 65536 { 5000 } else { 1000 };
        let rn = bench_fn(iters, || { black_box(rustynum_core::simd::dot_f32(&a, &b)); });
        let nd = bench_fn(iters, || { black_box(ndarray::backend::dot_f32(&a, &b)); });
        let ratio = nd as f64 / rn as f64;
        let status = if ratio >= 0.85 && ratio <= 1.15 { "✅" } else { "⚠️" };
        println!("{:<12} {:>10}ns {:>10}ns {:>10.3}x {}", n, rn, nd, ratio, status);
    }

    println!("\n═══ TEST 1b: axpy_f32 ═══");
    println!("{:<12} {:>12} {:>12} {:>12} {:>12}",
        "SIZE", "RUSTYNUM", "NDARRAY", "ND/RN", "STATUS");

    for &n in &[256, 1024, 4096, 16384, 65536, 262144, 1048576] {
        let x = gen_f32(n);
        let iters = if n <= 4096 { 10000 } else if n <= 65536 { 5000 } else { 1000 };
        let rn = {
            let mut y = gen_f32(n);
            bench_fn(iters, || { rustynum_core::simd::axpy_f32(2.0, &x, &mut y); })
        };
        let nd = {
            let mut y = gen_f32(n);
            bench_fn(iters, || { ndarray::backend::axpy_f32(2.0, &x, &mut y); })
        };
        let ratio = nd as f64 / rn as f64;
        let status = if ratio >= 0.85 && ratio <= 1.15 { "✅" } else { "⚠️" };
        println!("{:<12} {:>10}ns {:>10}ns {:>10.3}x {}", n, rn, nd, ratio, status);
    }
}

fn test2_hamming() {
    println!("\n═══ TEST 2: Hamming Distance (both VPOPCNTDQ) ═══");
    println!("{:<12} {:>12} {:>12} {:>12} {:>14}",
        "BITS", "RUSTYNUM", "NDARRAY", "ND/RN", "GBPS_RN");

    for &size_bits in &[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] {
        let size_bytes = size_bits / 8;
        let a_data = gen_u8(size_bytes);
        let b_data = gen_u8(size_bytes);
        let iters = if size_bytes <= 1024 { 50000 } else if size_bytes <= 8192 { 20000 } else { 5000 };

        let rn = bench_fn(iters, || {
            black_box(rustynum_core::simd::hamming_distance(&a_data, &b_data));
        });
        // Use the new raw-slice API
        let nd = bench_fn(iters, || {
            black_box(ndarray::hpc::bitwise::hamming_distance_raw(&a_data, &b_data));
        });
        let gbps = (size_bytes * 2) as f64 / rn as f64;
        let ratio = nd as f64 / rn as f64;
        let marker = if size_bits == 65536 { " ← FINGERPRINT" } else { "" };
        println!("{:<12} {:>10}ns {:>10}ns {:>10.3}x {:>12.1}{}",
            size_bits, rn, nd, ratio, gbps, marker);
    }
}

fn test3_gemm() {
    println!("\n═══ TEST 3: GEMM (sgemm) ═══");
    println!("{:<12} {:>12} {:>12} {:>12} {:>14} {:>14}",
        "SIZE", "RUSTYBLAS", "NDARRAY", "RB/ND", "RB_GFLOPS", "ND_GFLOPS");

    use rustynum_core::layout::{Layout, Transpose};

    for &n in &[64, 128, 256, 512, 1024] {
        let a = gen_f32(n * n);
        let b = gen_f32(n * n);
        let iters = if n <= 128 { 200 } else if n <= 256 { 50 } else if n <= 512 { 10 } else { 3 };

        let rb = {
            let mut c = vec![0.0f32; n * n];
            bench_fn(iters, || {
                c.fill(0.0);
                rustyblas::level3::sgemm(
                    Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                    n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n,
                );
                black_box(&c);
            })
        };

        let nd = {
            let mut c = vec![0.0f32; n * n];
            bench_fn(iters, || {
                c.fill(0.0);
                ndarray::backend::gemm_f32(n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n);
                black_box(&c);
            })
        };

        let flops = 2.0 * (n as f64).powi(3);
        let rb_gflops = flops / rb as f64;
        let nd_gflops = flops / nd as f64;
        let ratio = rb as f64 / nd as f64;
        println!("{:<12} {:>10.2}ms {:>10.2}ms {:>10.3}x {:>12.1} {:>12.1}",
            format!("{}x{}", n, n),
            rb as f64 / 1e6, nd as f64 / 1e6,
            ratio, rb_gflops, nd_gflops);
    }
}

fn test4_batch_hamming() {
    println!("\n═══ TEST 4: Batch Hamming (Cascade Stroke 1) ═══");
    println!("{:<12} {:>14} {:>12} {:>12} {:>12} {:>18}",
        "CANDIDATES", "STROKE_B", "RUSTYNUM", "ND_RAW", "ND_TRAIT", "M_CAND/S_RN");

    use ndarray::hpc::bitwise::BitwiseOps;

    for &(num_cand, stroke_bytes) in &[
        (1000usize, 128usize), (10000, 128), (100000, 128),
        (1000000, 128), (1000000, 512), (1000000, 2048),
    ] {
        let query_data = gen_u8(stroke_bytes);
        let db_data: Vec<u8> = (0..num_cand * stroke_bytes)
            .map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let iters = if num_cand <= 10000 { 100 } else { 10 };

        // rustynum: direct batch kernel
        let rn = bench_fn(iters, || {
            black_box(rustynum_core::simd::hamming_batch(
                &query_data, &db_data, num_cand, stroke_bytes));
        });

        // ndarray raw-slice batch (zero alloc, same codepath)
        let nd_raw = bench_fn(iters, || {
            black_box(ndarray::hpc::bitwise::hamming_batch_raw(
                &query_data, &db_data, num_cand, stroke_bytes));
        });

        // ndarray trait batch (Array1 + hamming_query_batch)
        let query_arr = ndarray::Array1::from_vec(query_data.clone());
        let nd_trait = bench_fn(iters, || {
            black_box(query_arr.hamming_query_batch(&db_data, stroke_bytes));
        });

        let m_cand = num_cand as f64 / (rn as f64 * 1e-3);
        let ratio_raw = nd_raw as f64 / rn as f64;
        let ratio_trait = nd_trait as f64 / rn as f64;
        println!("{:<12} {:>14} {:>10}μs {:>10}μs {:>10}μs {:>12.1}M/s  raw={:.1}x trait={:.1}x",
            num_cand, stroke_bytes,
            rn / 1000, nd_raw / 1000, nd_trait / 1000,
            m_cand / 1e6, ratio_raw, ratio_trait);
    }
}

fn test5_correctness() {
    println!("\n═══ TEST 5: Correctness ═══");
    use ndarray::hpc::bitwise::BitwiseOps;

    let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.002 + 0.5).collect();
    let rd = rustynum_core::simd::dot_f32(&a, &b);
    let nd = ndarray::backend::dot_f32(&a, &b);
    println!("dot_f32:   bit-exact={}", rd.to_bits() == nd.to_bits());

    let a64: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.001).collect();
    let b64: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.002 + 0.5).collect();
    let rd64 = rustynum_core::simd::dot_f64(&a64, &b64);
    let nd64 = ndarray::backend::dot_f64(&a64, &b64);
    println!("dot_f64:   bit-exact={}", rd64.to_bits() == nd64.to_bits());

    let x: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    let y_orig: Vec<f32> = (0..1024).map(|i| i as f32 * 0.2 + 1.0).collect();
    let mut yr = y_orig.clone();
    let mut yn = y_orig.clone();
    rustynum_core::simd::axpy_f32(2.5, &x, &mut yr);
    ndarray::backend::axpy_f32(2.5, &x, &mut yn);
    let max_ulp: u64 = yr.iter().zip(yn.iter())
        .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
        .max().unwrap_or(0);
    println!("axpy_f32:  max_ulp={} (1 ULP = FMA rounding — expected)", max_ulp);

    let ha = gen_u8(8192); // 64Kbit
    let hb = gen_u8(8192);
    let rh = rustynum_core::simd::hamming_distance(&ha, &hb);
    let nh = ndarray::hpc::bitwise::hamming_distance_raw(&ha, &hb);
    println!("hamming:   exact={} (rn={} nd={})", rh == nh, rh, nh);

    // Batch correctness
    let query = gen_u8(128);
    let db = gen_u8(1000 * 128);
    let rn_batch = rustynum_core::simd::hamming_batch(&query, &db, 1000, 128);
    let nd_batch = ndarray::hpc::bitwise::hamming_batch_raw(&query, &db, 1000, 128);
    let batch_ok = rn_batch.iter().zip(nd_batch.iter()).all(|(a, b)| a == b);
    println!("batch:     exact={} (1000 candidates × 128 bytes)", batch_ok);

    // GEMM correctness
    use rustynum_core::layout::{Layout, Transpose};
    let n = 64;
    let ga: Vec<f32> = (0..n*n).map(|i| ((i*7+3) % 100) as f32 * 0.01).collect();
    let gb: Vec<f32> = (0..n*n).map(|i| ((i*11+5) % 100) as f32 * 0.01).collect();
    let mut cr = vec![0.0f32; n*n];
    let mut cn = vec![0.0f32; n*n];
    rustyblas::level3::sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        n, n, n, 1.0, &ga, n, &gb, n, 0.0, &mut cr, n);
    ndarray::backend::gemm_f32(n, n, n, 1.0, &ga, n, &gb, n, 0.0, &mut cn, n);
    let max_err: f32 = cr.iter().zip(cn.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let max_val: f32 = cr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("sgemm 64:  max_err={:.2e} rel={:.2e} (max_val={:.2})",
        max_err, max_err / max_val, max_val);

    // GEMM 256 correctness
    let n = 256;
    let ga: Vec<f32> = (0..n*n).map(|i| ((i*7+3) % 100) as f32 * 0.01).collect();
    let gb: Vec<f32> = (0..n*n).map(|i| ((i*11+5) % 100) as f32 * 0.01).collect();
    let mut cr = vec![0.0f32; n*n];
    let mut cn = vec![0.0f32; n*n];
    rustyblas::level3::sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        n, n, n, 1.0, &ga, n, &gb, n, 0.0, &mut cr, n);
    ndarray::backend::gemm_f32(n, n, n, 1.0, &ga, n, &gb, n, 0.0, &mut cn, n);
    let max_err: f32 = cr.iter().zip(cn.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let max_val: f32 = cr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("sgemm 256: max_err={:.2e} rel={:.2e}", max_err, max_err / max_val);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK: ndarray (port) vs rustynum (reference) vs numpy    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n--- Hardware ---");
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    println!("CPU: {}", cpuinfo.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .unwrap_or("unknown").trim());

    #[cfg(target_arch = "x86_64")]
    {
        print!("SIMD: ");
        if is_x86_feature_detected!("avx512f") { print!("AVX-512F "); }
        if is_x86_feature_detected!("avx512vpopcntdq") { print!("VPOPCNTDQ "); }
        if is_x86_feature_detected!("avx512vnni") { print!("VNNI "); }
        if is_x86_feature_detected!("avx2") { print!("AVX2 "); }
        if is_x86_feature_detected!("fma") { print!("FMA "); }
        println!();
    }

    test1_blas1();
    test2_hamming();
    test3_gemm();
    test4_batch_hamming();
    test5_correctness();

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  DONE                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
