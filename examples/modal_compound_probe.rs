//! Modal-compound operator probe — is `Modal = How = Qualia ⊗ Means` a BIND, and what
//! is the right "absent factor" value (the manner-OR-means majority case)?
//!
//! The operator decision, settled by the math across BOTH bind carriers:
//!   • "How" is usually manner OR means, not both — one factor is frequently ABSENT.
//!   • MULTIPLY carrier (bipolar ±1; where lucency = coherence×valence lives): identity
//!     is +1, and 0 is the ANNIHILATOR — `bind(M, 0) = 0` destroys the present factor.
//!     So absence MUST be the multiplicative identity (all-ones), NOT 0. (The operator's
//!     "0 × something doesn't work except NaN" catch — confirmed here.)
//!   • XOR carrier (ndarray's real `vsa_bind` on Binary16K): identity is the ZERO vector,
//!     and XOR is self-inverse — `vsa_bind(M, 0) = M`. Here 0 IS the identity, so absence
//!     as zero works for free; the annihilation problem does not arise.
//!   • BUNDLE (add / majority): WRONG for a compound — the result stays similar to each
//!     factor, so "carefully" and "carefully-with-a-knife" conflate. (bundle is the right
//!     operator for the Markov/temporal SUM, per I-SUBSTRATE-MARKOV — a different job.)
//!
//! Grounded in `I-VSA-IDENTITIES` (bind on high-D IDENTITY fingerprints; the 16D qualia
//! POINTS TO content, the carrier is where ⊗ lives) and the real `ndarray::hpc::vsa`.
//!
//! ```text
//! M1  OR-regime: present factor survives when the other is ABSENT (per carrier × absent value)
//! M2  both-regime: the compound is DISTINCT from each simple factor (not conflated)
//! M3  both-regime: the (manner, means) pair is RECOVERABLE (unbind)
//! ```
//!
//!   cargo run --release --example modal_compound_probe --features std

use ndarray::hpc::vsa::{vsa_bind, vsa_bundle, vsa_similarity, vsa_unbind, VsaVector};

const D: usize = 4096; // bipolar-multiply carrier dimension

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

// ── Multiply carrier (bipolar ±1): bind = elementwise *, identity = +1, annihilator = 0 ──
fn bipolar(s: &mut u64) -> Vec<f64> {
    (0..D)
        .map(|_| if splitmix(s) < 0.5 { -1.0 } else { 1.0 })
        .collect()
}
fn mul_bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}
fn add_bundle(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0 // an annihilated (zero) vector has no direction — undefined → 0
    } else {
        dot / (na * nb)
    }
}

fn main() {
    println!("== Modal-compound operator probe: How = Qualia ⊗ Means — bind (× and XOR) vs bundle ==\n");

    let trials = 2000usize;
    let mut s = 0x4D0DA1_u64;

    // Multiply-carrier accumulators.
    let (mut mul_or_id, mut mul_or_zero, mut mul_distinct, mut mul_recover) = (0.0, 0.0, 0.0, 0.0);
    let mut mul_bundle_sim = 0.0;
    // XOR-carrier accumulators (vsa_similarity: 1.0 same, ~0.5 orthogonal/distinct).
    let (mut xor_or_zero, mut xor_distinct, mut xor_recover) = (0.0f32, 0.0f32, 0.0f32);
    let mut xor_bundle_sim = 0.0f32;

    let ones = vec![1.0f64; D];
    let zeros = vec![0.0f64; D];

    for t in 0..trials {
        // Multiply carrier.
        let m = bipolar(&mut s);
        let n = bipolar(&mut s);
        mul_or_id += cosine(&mul_bind(&m, &ones), &m); // absent=identity(+1) → survives
        mul_or_zero += cosine(&mul_bind(&m, &zeros), &m); // absent=0 → ANNIHILATES
        let both = mul_bind(&m, &n);
        mul_distinct += cosine(&both, &m).abs(); // → ~0: compound ⟂ factor (distinct)
        mul_recover += cosine(&mul_bind(&both, &n), &m); // re-bind = unbind (× self-inverse) → 1
        mul_bundle_sim += cosine(&add_bundle(&m, &n), &m); // → ~0.707: resembles factor (conflated)

        // XOR carrier (real ndarray vsa). Distinct seeds per trial.
        let mx = VsaVector::random(0x1000 + t as u64 * 2);
        let nx = VsaVector::random(0x1001 + t as u64 * 2);
        let zero = VsaVector::zero();
        xor_or_zero += vsa_similarity(&vsa_bind(&mx, &zero), &mx); // absent=0=IDENTITY → survives
        let bothx = vsa_bind(&mx, &nx);
        xor_distinct += vsa_similarity(&bothx, &mx); // → ~0.5: orthogonal (distinct)
        xor_recover += vsa_similarity(&vsa_unbind(&bothx, &nx), &mx); // → 1.0: recoverable
        xor_bundle_sim += vsa_similarity(&vsa_bundle(&[mx.clone(), mx.clone(), mx.clone(), nx.clone()]), &nx);
        // ^ a bundle stays similar to its members (conflation), unlike the orthogonal bind compound.
    }
    let t = trials as f64;
    let tf = trials as f32;

    println!("MULTIPLY carrier (bipolar ±1; where lucency = coherence×valence lives):");
    println!("  M1 OR-regime present-factor survival  cosine(bind(M, absent), M):");
    println!("       absent = identity (+1)  {:.3}   ← survives", mul_or_id / t);
    println!(
        "       absent = 0               {:.3}   ← ANNIHILATED (0×x=0 — the operator's catch)",
        mul_or_zero / t
    );
    println!(
        "  M2 compound distinctness  |cosine(bind(M,N), M)| = {:.3}   (→ 0: 'carefully-with-a-knife' ⟂ 'carefully')",
        mul_distinct / t
    );
    println!(
        "  M3 pair recoverable       cosine(unbind(both, N), M) = {:.3}   (× is self-inverse)",
        mul_recover / t
    );
    println!(
        "  bundle conflation         cosine(M+N, M) = {:.3}   ← resembles the factor (can't tell simple from compound)",
        mul_bundle_sim / t
    );

    println!("\nXOR carrier (ndarray real `vsa_bind` on Binary16K; 0.5 = orthogonal, 1.0 = identical):");
    println!(
        "  M1 OR-regime, absent = 0   vsa_similarity(vsa_bind(M, 0), M) = {:.3}   ← survives (0 IS the XOR identity!)",
        xor_or_zero / tf
    );
    println!(
        "  M2 compound distinctness   vsa_similarity(bind(M,N), M) = {:.3}   (~0.5 = orthogonal: distinct)",
        xor_distinct / tf
    );
    println!(
        "  M3 pair recoverable        vsa_similarity(vsa_unbind(both, N), M) = {:.3}   (XOR self-inverse)",
        xor_recover / tf
    );
    println!(
        "  bundle conflation          vsa_similarity(bundle[M,M,M,N], N) = {:.3}   ← bundle stays similar to members",
        xor_bundle_sim / tf
    );

    println!("\nVERDICT — Modal = bind(Qualia manner, Instrument means), NOT bundle:");
    println!(
        "  • Both bind algebras keep the compound DISTINCT (× |cos| {:.2}; XOR sim {:.2}≈orthogonal) and",
        mul_distinct / t,
        xor_distinct / tf
    );
    println!(
        "    RECOVERABLE (× {:.2}; XOR {:.2}). bundle CONFLATES (× cos {:.2}; XOR sim {:.2}) — wrong for a compound.",
        mul_recover / t,
        xor_recover / tf,
        mul_bundle_sim / t,
        xor_bundle_sim / tf
    );
    println!("  • The 'absent factor' value is CARRIER-SPECIFIC, exactly as the operator reasoned:");
    println!("      MULTIPLY (lucency, bipolar Vsa16kF32): absence = identity (+1); 0 annihilates ({:.2}). NaN/identity, never 0.", mul_or_zero / t);
    println!(
        "      XOR (Binary16K): absence = the ZERO vector = the identity ({:.2}) — '0' is correct here, for free.",
        xor_or_zero / tf
    );
    println!("  • So the resolver fills Modal by binding manner⊗means on whichever carrier; absent → that");
    println!("    carrier's bind-identity (0 for XOR, +1 for multiply); the four regimes never annihilate.");
}
