use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p64::*;

fn make_heels() -> HeelPlanes {
    HeelPlanes::new([
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0xFFFF_0000_FFFF_0000,
        0x0000_FFFF_0000_FFFF,
        0xFF00_FF00_FF00_FF00,
        0x00FF_00FF_00FF_00FF,
        0xF0F0_F0F0_F0F0_F0F0,
        0x0F0F_0F0F_0F0F_0F0F,
    ])
}

fn bench_expand(c: &mut Criterion) {
    let heels = make_heels();
    c.bench_function("expand_8_to_64", |b| {
        b.iter(|| black_box(black_box(&heels).expand()))
    });
}

fn bench_attend(c: &mut Criterion) {
    let palette = make_heels().expand();
    let query = 0xDEAD_BEEF_CAFE_BABEu64;

    c.bench_function("attend_single", |b| {
        b.iter(|| black_box(black_box(&palette).attend(black_box(query), 16)))
    });
}

fn bench_attend_batch(c: &mut Criterion) {
    let palette = make_heels().expand();
    let queries: Vec<u64> = (0..1024).map(|i| i * 0x0123_4567_89AB + 0xDEAD).collect();

    c.bench_function("attend_1024_queries", |b| {
        b.iter(|| {
            for &q in &queries {
                black_box(palette.attend(black_box(q), 16));
            }
        })
    });
}

fn bench_moe_gate(c: &mut Criterion) {
    let heels = make_heels();
    let query = 0xDEAD_BEEF_CAFE_BABEu64;

    c.bench_function("moe_gate", |b| {
        b.iter(|| black_box(black_box(&heels).moe_gate(black_box(query), 20)))
    });
}

fn bench_soft_moe(c: &mut Criterion) {
    let heels = make_heels();
    let query = 0xDEAD_BEEF_CAFE_BABEu64;

    c.bench_function("soft_moe", |b| {
        b.iter(|| black_box(black_box(&heels).soft_moe(black_box(query), 20)))
    });
}

fn bench_denoise(c: &mut Criterion) {
    let palette = make_heels().expand();
    let noisy = 0x1234_5678_9ABC_DEF0u64;

    c.bench_function("denoise_10_steps", |b| {
        b.iter(|| black_box(black_box(&palette).denoise(black_box(noisy), 10, 32)))
    });
}

fn bench_nearest_k(c: &mut Criterion) {
    let palette = make_heels().expand();
    let query = 0xDEAD_BEEF_CAFE_BABEu64;

    c.bench_function("nearest_k_8", |b| {
        b.iter(|| black_box(black_box(&palette).nearest_k(black_box(query), 8)))
    });
}

criterion_group!(
    benches,
    bench_expand,
    bench_attend,
    bench_attend_batch,
    bench_moe_gate,
    bench_soft_moe,
    bench_denoise,
    bench_nearest_k,
);
criterion_main!(benches);
