//! Benchmark: ndarray SIMD OCR preprocessing vs tesseract.
//!
//! Loads raw grayscale page images, runs:
//! 1. ndarray SIMD: otsu + binarize + density + skew
//! 2. tesseract: full pipeline (preprocess + LSTM)
//!
//! Compares wall-clock time and output quality.

use ndarray::hpc::ocr_simd::*;
use std::time::Instant;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  OCR Benchmark: ndarray SIMD vs tesseract");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let pages = vec![
        "/tmp/ocr_bench/page-01.raw",
        "/tmp/ocr_bench/page-02.raw",
        "/tmp/ocr_bench/page-03.raw",
    ];

    let png_pages = vec![
        "/tmp/ocr_bench/page-01.png",
        "/tmp/ocr_bench/page-02.png",
        "/tmp/ocr_bench/page-03.png",
    ];

    // ── ndarray SIMD preprocessing ────────────────────────────────────
    eprintln!("=== ndarray SIMD preprocessing ===\n");
    let mut simd_total = std::time::Duration::ZERO;

    for (i, path) in pages.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("  skip {}: {}", path, e); continue; }
        };
        if data.len() < 8 { continue; }

        let width = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let height = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let pixels = &data[8..];

        eprintln!("  Page {}: {}×{} ({:.1}M pixels)", i + 1, width, height,
            (width * height) as f64 / 1_000_000.0);

        let img = GrayImage { data: pixels, width, height };

        // Warm up
        let _ = otsu_threshold(&img);

        // Benchmark: full preprocessing pipeline
        let t0 = Instant::now();
        let result = preprocess_page(&img);
        let elapsed = t0.elapsed();
        simd_total += elapsed;

        let fg_count = foreground_count(&result.binary);
        eprintln!("    Otsu threshold:  {}", result.threshold);
        eprintln!("    Foreground:      {} pixels ({:.1}%)", fg_count, result.density * 100.0);
        eprintln!("    Skew angle:      {:.2}°", result.skew_angle);
        eprintln!("    Is content:      {}", result.is_content);
        eprintln!("    Time:            {:.3}ms", elapsed.as_secs_f64() * 1000.0);

        // Also benchmark individual steps
        let t1 = Instant::now();
        let threshold = otsu_threshold(&img);
        let otsu_time = t1.elapsed();

        let t2 = Instant::now();
        let binary = binarize(&img, threshold);
        let binarize_time = t2.elapsed();

        let t3 = Instant::now();
        let _ = foreground_density(&binary);
        let density_time = t3.elapsed();

        let t4 = Instant::now();
        let _ = estimate_skew(&binary);
        let skew_time = t4.elapsed();

        eprintln!("    Breakdown:");
        eprintln!("      Otsu:     {:.3}ms", otsu_time.as_secs_f64() * 1000.0);
        eprintln!("      Binarize: {:.3}ms", binarize_time.as_secs_f64() * 1000.0);
        eprintln!("      Density:  {:.3}ms", density_time.as_secs_f64() * 1000.0);
        eprintln!("      Skew:     {:.3}ms", skew_time.as_secs_f64() * 1000.0);

        // Adaptive binarization benchmark
        let t5 = Instant::now();
        let _ = adaptive_binarize(&img, 31, 10.0);
        let adaptive_time = t5.elapsed();
        eprintln!("      Adaptive: {:.3}ms (window=31)", adaptive_time.as_secs_f64() * 1000.0);

        // Throughput
        let mpix = (width * height) as f64 / 1_000_000.0;
        let mpix_per_sec = mpix / elapsed.as_secs_f64();
        eprintln!("    Throughput:      {:.0} Mpix/s\n", mpix_per_sec);
    }

    // ── tesseract full pipeline ───────────────────────────────────────
    eprintln!("=== tesseract (full pipeline: preprocess + LSTM) ===\n");
    let mut tess_total = std::time::Duration::ZERO;

    for (i, path) in png_pages.iter().enumerate() {
        let t0 = Instant::now();
        let output = std::process::Command::new("tesseract")
            .args([path.as_ref(), "stdout", "-l", "eng", "--psm", "1"])
            .output();
        let elapsed = t0.elapsed();
        tess_total += elapsed;

        match output {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                let words = text.split_whitespace().count();
                eprintln!("  Page {}: {} words, {:.3}ms",
                    i + 1, words, elapsed.as_secs_f64() * 1000.0);
                // Show first 100 chars
                let preview: String = text.chars().take(100).collect();
                eprintln!("    Preview: {}", preview.replace('\n', " "));
            }
            _ => {
                eprintln!("  Page {}: FAILED, {:.3}ms", i + 1, elapsed.as_secs_f64() * 1000.0);
            }
        }
    }

    // ── Comparison ────────────────────────────────────────────────────
    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  COMPARISON (3 pages, 2481×3508 @ 300 DPI)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  ndarray SIMD preprocess:  {:.1}ms total", simd_total.as_secs_f64() * 1000.0);
    eprintln!("  tesseract full pipeline:  {:.1}ms total", tess_total.as_secs_f64() * 1000.0);

    if tess_total.as_secs_f64() > 0.001 {
        let speedup = tess_total.as_secs_f64() / simd_total.as_secs_f64().max(0.001);
        eprintln!("  Speedup (preprocess):     {:.0}x", speedup);
    }

    eprintln!("\n  Note: SIMD does preprocessing only (binarize, skew, density).");
    eprintln!("  tesseract does preprocessing + LSTM character recognition.");
    eprintln!("  Optimal: SIMD preprocess → pipe to tesseract LSTM only.");
    eprintln!("═══════════════════════════════════════════════════════════\n");
}
