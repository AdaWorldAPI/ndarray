use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use phyllotactic_manifold::*;

fn make_seeds(n: usize) -> Vec<[i8; 34]> {
    let mut seeds = Vec::with_capacity(n);
    for i in 0..n {
        let mut seed = [0i8; 34];
        seed[0] = (i % 128) as i8; // HEEL
        seed[33] = ((i * 7) % 128) as i8; // GAMMA
        for j in 1..33 {
            seed[j] = ((i * 13 + j * 37) % 256) as i8;
        }
        seeds.push(seed);
    }
    seeds
}

fn bench_encode(c: &mut Criterion) {
    let seeds = make_seeds(1024);
    let mut group = c.benchmark_group("encode");

    group.bench_function("flat8", |b| {
        b.iter(|| {
            for seed in &seeds {
                black_box(flat8::encode(black_box(seed)));
            }
        })
    });

    group.bench_function("spiral8", |b| {
        b.iter(|| {
            for seed in &seeds {
                black_box(spiral8::encode(black_box(seed)));
            }
        })
    });

    group.bench_function("spiral8_gamma", |b| {
        b.iter(|| {
            for seed in &seeds {
                black_box(spiral8_gamma::encode(black_box(seed)));
            }
        })
    });

    group.bench_function("seven_plus_one", |b| {
        b.iter(|| {
            for seed in &seeds {
                black_box(seven_plus_one::encode(black_box(seed)));
            }
        })
    });

    group.finish();
}

fn bench_resonance(c: &mut Criterion) {
    let seeds = make_seeds(1024);
    let threshold = 1000.0;

    // Pre-encode
    let flat8_enc: Vec<_> = seeds.iter().map(|s| flat8::encode(s)).collect();
    let spiral8_enc: Vec<_> = seeds.iter().map(|s| spiral8::encode(s)).collect();
    let spiral8g_enc: Vec<_> = seeds.iter().map(|s| spiral8_gamma::encode(s)).collect();
    let s7p1_enc: Vec<_> = seeds.iter().map(|s| seven_plus_one::encode(s)).collect();

    let mut group = c.benchmark_group("resonance");

    group.bench_function("flat8", |b| {
        b.iter(|| {
            for enc in &flat8_enc {
                black_box(flat8::resonance(black_box(enc), threshold));
            }
        })
    });

    group.bench_function("spiral8", |b| {
        b.iter(|| {
            for (x, y) in &spiral8_enc {
                black_box(spiral8::resonance(black_box(x), black_box(y), threshold));
            }
        })
    });

    group.bench_function("spiral8_gamma", |b| {
        b.iter(|| {
            for (x, y) in &spiral8g_enc {
                black_box(spiral8_gamma::resonance(
                    black_box(x),
                    black_box(y),
                    threshold,
                ));
            }
        })
    });

    group.bench_function("seven_plus_one", |b| {
        b.iter(|| {
            for m in &s7p1_enc {
                black_box(seven_plus_one::resonance(black_box(m), threshold));
            }
        })
    });

    group.finish();
}

fn bench_clam48(c: &mut Criterion) {
    let seeds = make_seeds(1024);
    let manifolds: Vec<_> = seeds.iter().map(|s| seven_plus_one::encode(s)).collect();

    c.bench_function("clam48_extraction", |b| {
        b.iter(|| {
            for m in &manifolds {
                black_box(seven_plus_one::to_clam48(black_box(m), 100.0, 1e8));
            }
        })
    });
}

fn bench_dead_zone(c: &mut Criterion) {
    let seed = {
        let mut s = [0i8; 34];
        s[0] = 42;
        s[33] = 7;
        for i in 1..33 {
            s[i] = (i as i8).wrapping_mul(13).wrapping_add(37);
        }
        s
    };

    c.bench_function("dead_zone_full_272bit", |b| {
        b.iter(|| {
            black_box(dead_zone::run_benchmark(black_box(&seed), 100.0));
        })
    });
}

fn bench_encode_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_scaling");
    for n in [64, 256, 1024, 4096] {
        let seeds = make_seeds(n);
        group.bench_with_input(BenchmarkId::new("seven_plus_one", n), &seeds, |b, seeds| {
            b.iter(|| {
                for seed in seeds {
                    black_box(seven_plus_one::encode(black_box(seed)));
                }
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_resonance,
    bench_clam48,
    bench_dead_zone,
    bench_encode_scaling,
);
criterion_main!(benches);
