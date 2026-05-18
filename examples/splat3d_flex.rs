//! `splat3d_flex` — CPU-SIMD 3D Gaussian Splatting end-to-end demo.
//!
//! Loads a pre-trained scene from `.ply`, renders frames along a
//! pre-baked circular camera path, writes PPM output, reports timing.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --features splat3d --example splat3d_flex -- \
//!   --scene path/to/scene.ply --frames 100 --out /tmp/render/
//! ```
//!
//! `--scene` accepts the Inria 3DGS canonical PLY layout (see
//! `ndarray::hpc::splat3d::ply` for the exact spec). The example also
//! works on a synthetic scene if `--scene` is omitted — see
//! `tests/splat3d_correctness.rs` for the synthetic-cube builder used
//! as the smoke-test fallback.
//!
//! ## Output format
//!
//! Frames are written as PPM (P6 binary) at 1080p. PPM is chosen over
//! PNG because the splat3d feature stack carries no compression
//! dependencies; PPM is trivially encoded (header + raw RGB bytes) and
//! widely supported by image viewers and post-processing tools. A
//! follow-up sprint can add PNG via `image` or `png` when the demo
//! gains real distribution channels.
//!
//! ## Why PPM not PNG
//!
//! PNG = IHDR + IDAT (DEFLATE-compressed) + IEND. DEFLATE requires
//! either an in-tree implementation (~800 LoC) or a `flate2`-like
//! dep — both out of scope for the sprint's "no new top-level deps"
//! invariant. PPM has identical pixel data and 14-byte overhead per
//! file, no compression, no library dep.

#![cfg(feature = "splat3d")]

use ndarray::hpc::splat3d::{read_ply, Camera, Gaussian3D, SplatFrame, SH_COEFFS_PER_GAUSSIAN};
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    scene: Option<PathBuf>,
    frames: usize,
    out: PathBuf,
    width: u32,
    height: u32,
}

impl Args {
    fn parse() -> Self {
        let mut scene: Option<PathBuf> = None;
        let mut frames: usize = 100;
        let mut out = PathBuf::from("/tmp/splat3d_render/");
        let mut width: u32 = 1920;
        let mut height: u32 = 1080;
        let mut argv = env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--scene" => scene = argv.next().map(PathBuf::from),
                "--frames" => frames = argv.next().and_then(|s| s.parse().ok()).unwrap_or(100),
                "--out" => out = argv.next().map(PathBuf::from).unwrap_or(out),
                "--width" => width = argv.next().and_then(|s| s.parse().ok()).unwrap_or(width),
                "--height" => height = argv.next().and_then(|s| s.parse().ok()).unwrap_or(height),
                "-h" | "--help" => {
                    eprintln!(
                        "Usage: splat3d_flex [--scene PATH.ply] [--frames N] [--out DIR] [--width W] [--height H]"
                    );
                    std::process::exit(0);
                }
                other => eprintln!("warning: unrecognized arg `{other}` (ignored)"),
            }
        }
        Args {
            scene,
            frames,
            out,
            width,
            height,
        }
    }
}

fn build_synthetic_fallback_scene(frame: &mut SplatFrame) {
    // Same shape as the integration test: 10×10×10 grid of small
    // gaussians spanning [-2, 2]³, colored by position.
    let n = 10;
    let sh_c0: f32 = 0.28209479177387814;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let x = -2.0 + (ix as f32) * (4.0 / (n - 1) as f32);
                let y = -2.0 + (iy as f32) * (4.0 / (n - 1) as f32);
                let z = -2.0 + (iz as f32) * (4.0 / (n - 1) as f32);
                let mut sh = [0.0f32; SH_COEFFS_PER_GAUSSIAN];
                sh[0] = (ix as f32) / (n - 1) as f32 / sh_c0;
                sh[16] = (iy as f32) / (n - 1) as f32 / sh_c0;
                sh[32] = (iz as f32) / (n - 1) as f32 / sh_c0;
                frame.gaussians.push(Gaussian3D {
                    mean: [x, y, z],
                    scale: [0.08, 0.08, 0.08],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    opacity: 0.9,
                    sh,
                });
            }
        }
    }
}

fn bake_circular_camera_path(width: u32, height: u32, n_frames: usize) -> Vec<Camera> {
    // Camera orbits the origin at radius 5 in the XZ plane, always
    // looking at (0, 0, 0). For each frame, build the world→camera
    // view matrix from the position + look-at.
    let radius = 5.0f32;
    let mut out = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let theta = (i as f32) / (n_frames as f32) * std::f32::consts::TAU;
        let cam_pos = [radius * theta.cos(), 0.0, radius * theta.sin()];
        // Look-at the origin: forward = normalize(origin - cam_pos);
        // up = (0, 1, 0); right = normalize(cross(forward, up)).
        let forward = {
            let f = [-cam_pos[0], -cam_pos[1], -cam_pos[2]];
            let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            [f[0] / n, f[1] / n, f[2] / n]
        };
        let up = [0.0f32, 1.0, 0.0];
        // right = forward × up, then up' = right × forward (for full ortho basis).
        let right = {
            let r = [
                forward[1] * up[2] - forward[2] * up[1],
                forward[2] * up[0] - forward[0] * up[2],
                forward[0] * up[1] - forward[1] * up[0],
            ];
            let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            [r[0] / n, r[1] / n, r[2] / n]
        };
        let up_ortho = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ];
        // View matrix: rows are right, up, forward (with translation
        // baked in as -dot(axis, cam_pos)).
        let tx = -(right[0] * cam_pos[0] + right[1] * cam_pos[1] + right[2] * cam_pos[2]);
        let ty = -(up_ortho[0] * cam_pos[0] + up_ortho[1] * cam_pos[1] + up_ortho[2] * cam_pos[2]);
        let tz = -(forward[0] * cam_pos[0] + forward[1] * cam_pos[1] + forward[2] * cam_pos[2]);
        let view = [
            [right[0], right[1], right[2], tx],
            [up_ortho[0], up_ortho[1], up_ortho[2], ty],
            [forward[0], forward[1], forward[2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let fx = width.max(height) as f32;
        out.push(Camera {
            view,
            fx,
            fy: fx,
            cx: width as f32 * 0.5,
            cy: height as f32 * 0.5,
            near: 0.01,
            far: 1000.0,
            width,
            height,
            position: cam_pos,
        });
    }
    out
}

fn write_ppm(path: &std::path::Path, fb: &[f32], width: u32, height: u32) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    write!(w, "P6\n{width} {height}\n255\n")?;
    let mut row = vec![0u8; (width * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = (fb[idx] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (fb[idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (fb[idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
            let dst = (x * 3) as usize;
            row[dst] = r;
            row[dst + 1] = g;
            row[dst + 2] = b;
        }
        w.write_all(&row)?;
    }
    Ok(())
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (values.len() as f64 - 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn main() {
    let args = Args::parse();
    create_dir_all(&args.out).expect("failed to create output dir");

    // Load the scene (PLY) or fall back to the synthetic cube.
    let mut frame = if let Some(scene_path) = &args.scene {
        eprintln!("Loading scene from {} …", scene_path.display());
        let file = File::open(scene_path).expect("scene file open failed");
        let batch = read_ply(BufReader::new(file)).expect("ply parse failed");
        eprintln!("Loaded {} gaussians", batch.len);
        let mut f = SplatFrame::with_capacity(batch.len, args.width, args.height);
        f.gaussians = batch;
        f
    } else {
        eprintln!("No --scene flag; using synthetic 1000-gaussian cube.");
        let mut f = SplatFrame::with_capacity(1000, args.width, args.height);
        build_synthetic_fallback_scene(&mut f);
        f
    };

    eprintln!("Rendering {} frames at {}×{} into {} …", args.frames, args.width, args.height, args.out.display());
    let path = bake_circular_camera_path(args.width, args.height, args.frames);
    let mut times_ms: Vec<f64> = Vec::with_capacity(args.frames);

    for (i, camera) in path.iter().enumerate() {
        let t0 = Instant::now();
        frame.tick(camera, [0.0, 0.0, 0.0]);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(dt);
        // Save every 10th frame to keep disk usage bounded.
        if i % 10 == 0 {
            let outpath = args.out.join(format!("frame_{i:04}.ppm"));
            if let Err(e) = write_ppm(&outpath, &frame.framebuffer, args.width, args.height) {
                eprintln!("failed to write {}: {e}", outpath.display());
            }
        }
    }

    let p50 = percentile(&mut times_ms.clone(), 50.0);
    let p95 = percentile(&mut times_ms.clone(), 95.0);
    let p99 = percentile(&mut times_ms.clone(), 99.0);
    let fps = if p50 > 0.0 { 1000.0 / p50 } else { f64::INFINITY };
    println!("Per-frame timing (ms): p50={p50:.2} p95={p95:.2} p99={p99:.2}");
    println!("Throughput (p50-derived): {fps:.1} fps");
}
