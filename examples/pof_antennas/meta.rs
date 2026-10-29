#![allow(dead_code)]
//! Scalar oracle for "one fold inspects another fold's receipt".
//!
//! First-level fold: `evaluate` reads a sealed [`World`] (optionally through a
//! speculative [`Thought`]), applies ordered [`Step`]s, then a Kleene
//! [`Method`], and emits a [`Receipt`].
//!
//! Second-level (meta) fold: `compare`, `may_reuse`, `aggregate`,
//! `which_changed` and `meta_evaluate` take receipts only (plus, for reuse, a
//! version lookup). Their signatures never mention `World`, `Thought` or a fact
//! array, so the meta fold cannot read the underlying facts.
//!
//! Mock typed data only; plain safe std Rust, no global state.

/// Three-valued (Kleene) truth. `Unknown` is a value, never coerced silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    /// Known true.
    True,
    /// Known false.
    False,
    /// Not known either way.
    Unknown,
}

impl Tri {
    /// Kleene conjunction: False dominates, then Unknown, then True.
    pub fn and(self, o: Tri) -> Tri {
        match (self, o) {
            (Tri::False, _) | (_, Tri::False) => Tri::False,
            (Tri::True, Tri::True) => Tri::True,
            _ => Tri::Unknown,
        }
    }

    /// Kleene disjunction: True dominates, then Unknown, then False.
    pub fn or(self, o: Tri) -> Tri {
        match (self, o) {
            (Tri::True, _) | (_, Tri::True) => Tri::True,
            (Tri::False, Tri::False) => Tri::False,
            _ => Tri::Unknown,
        }
    }

    /// Kleene negation; `Unknown` stays `Unknown`.
    pub fn not(self) -> Tri {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }

    fn byte(self) -> u8 {
        match self {
            Tri::False => 0,
            Tri::True => 1,
            Tri::Unknown => 2,
        }
    }
}

/// How `Unknown` is treated when a method runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Unknown stays Unknown.
    OpenWorld,
    /// Unknown is explicitly mapped to False before the method runs.
    ClosedWorld,
}

impl Policy {
    fn dep_id(self) -> u64 {
        match self {
            Policy::OpenWorld => 0,
            Policy::ClosedWorld => 1,
        }
    }
}

/// A sealed 64-lane world at one version. There is no `&mut` method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct World {
    /// World identity (stable across versions).
    pub id: u32,
    /// Monotone version.
    pub version: u64,
    facts: [Tri; 64],
}

impl World {
    /// Build a world from explicit facts.
    pub fn new(id: u32, version: u64, facts: [Tri; 64]) -> World {
        World { id, version, facts }
    }

    /// Fact at lane `i` (`Unknown` if out of range).
    pub fn fact(&self, i: usize) -> Tri {
        if i < 64 {
            self.facts[i]
        } else {
            Tri::Unknown
        }
    }

    /// New world, same id, version + 1, with `changes` applied.
    pub fn advance(&self, changes: &[(usize, Tri)]) -> World {
        let mut facts = self.facts;
        for &(i, t) in changes {
            if i < 64 {
                facts[i] = t;
            }
        }
        World {
            id: self.id,
            version: self.version + 1,
            facts,
        }
    }
}

/// A speculative projection over one world version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thought {
    /// Branch identity.
    pub branch: u32,
    /// World version this thought was made against.
    pub base_version: u64,
    /// Lane overrides.
    pub overlay: Vec<(usize, Tri)>,
}

/// Why a view was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stale {
    /// The thought's base version is not the world's version.
    World {
        /// World's current version.
        expected: u64,
        /// Version the thought was based on.
        found: u64,
    },
}

/// World facts with the thought overlay applied; refuses a stale thought.
pub fn view(w: &World, t: Option<&Thought>) -> Result<[Tri; 64], Stale> {
    let mut v = w.facts;
    if let Some(t) = t {
        if t.base_version != w.version {
            return Err(Stale::World {
                expected: w.version,
                found: t.base_version,
            });
        }
        for &(i, x) in &t.overlay {
            if i < 64 {
                v[i] = x;
            }
        }
    }
    Ok(v)
}

/// A Kleene method over a lane set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Conjunction over the lanes.
    AllOf {
        /// Method id.
        id: u32,
        /// Lane bitmask.
        lanes: u64,
    },
    /// Disjunction over the lanes.
    AnyOf {
        /// Method id.
        id: u32,
        /// Lane bitmask.
        lanes: u64,
    },
    /// At least `k` lanes true.
    AtLeast {
        /// Method id.
        id: u32,
        /// Lane bitmask.
        lanes: u64,
        /// Threshold.
        k: u32,
    },
}

impl Method {
    /// Method id.
    pub fn id(&self) -> u32 {
        match *self {
            Method::AllOf { id, .. } | Method::AnyOf { id, .. } | Method::AtLeast { id, .. } => id,
        }
    }

    /// Lane bitmask.
    pub fn lanes(&self) -> u64 {
        match *self {
            Method::AllOf { lanes, .. } | Method::AnyOf { lanes, .. } | Method::AtLeast { lanes, .. } => lanes,
        }
    }
}

/// Run a method over `v` under policy `p` (ClosedWorld coerces Unknown first).
pub fn apply_method(m: &Method, v: &[Tri; 64], p: Policy) -> Tri {
    let lane = |i: usize| -> Tri {
        let t = v[i];
        if p == Policy::ClosedWorld && t == Tri::Unknown {
            Tri::False
        } else {
            t
        }
    };
    let selected = (0..64usize).filter(|i| (m.lanes() >> i) & 1 == 1);
    match *m {
        Method::AllOf { .. } => selected.fold(Tri::True, |a, i| a.and(lane(i))),
        Method::AnyOf { .. } => selected.fold(Tri::False, |a, i| a.or(lane(i))),
        Method::AtLeast { k, .. } => {
            let (mut t, mut u) = (0u32, 0u32);
            for i in selected {
                match lane(i) {
                    Tri::True => t += 1,
                    Tri::Unknown => u += 1,
                    Tri::False => {}
                }
            }
            if t >= k {
                Tri::True
            } else if t + u < k {
                Tri::False
            } else {
                Tri::Unknown
            }
        }
    }
}

/// An ordered effect on the view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Overwrite a lane.
    SetFact(usize, Tri),
    /// Negate a lane (Kleene).
    NegateFact(usize),
}

/// Apply steps in order.
pub fn apply_steps(mut v: [Tri; 64], steps: &[Step]) -> [Tri; 64] {
    for s in steps {
        match *s {
            Step::SetFact(i, t) => {
                if i < 64 {
                    v[i] = t;
                }
            }
            Step::NegateFact(i) => {
                if i < 64 {
                    v[i] = v[i].not();
                }
            }
        }
    }
    v
}

/// Order-sensitive FNV-1a over a canonical byte encoding of the step list.
pub fn order_witness(steps: &[Step]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for s in steps {
        let (tag, i, t) = match *s {
            Step::SetFact(i, t) => (1u8, i as u8, t.byte()),
            Step::NegateFact(i) => (2u8, i as u8, 0u8),
        };
        eat(tag);
        eat(i);
        eat(t);
    }
    h
}

/// What a receipt says was consulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evidence {
    /// World id.
    pub world_id: u32,
    /// World version.
    pub world_version: u64,
    /// Lane mask the method read.
    pub lanes_consulted: u64,
}

/// Kind of dependency a receipt recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepKind {
    /// A world (id, version).
    World,
    /// A thought (branch, base_version).
    Thought,
    /// A method (id, 0).
    Method,
    /// A policy (0 open / 1 closed, 0).
    Policy,
    /// A step sequence (order witness, 0).
    Steps,
}

/// One recorded dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dep {
    /// Kind.
    pub kind: DepKind,
    /// Identity within the kind.
    pub id: u64,
    /// Version recorded at evaluation time.
    pub version: u64,
}

/// A first-level fold's receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    /// Receipt identity.
    pub receipt_id: u64,
    /// (world id, world version).
    pub frame: (u32, u64),
    /// Basis identity.
    pub basis_id: u64,
    /// Thought branch, if any.
    pub thought: Option<u32>,
    /// Method id.
    pub method_id: u32,
    /// Order witness of the step list.
    pub order_witness: u64,
    /// Law that produced the result.
    pub law: &'static str,
    /// Policy that ran.
    pub policy: Policy,
    /// Result.
    pub result: Tri,
    /// Evidence consulted.
    pub evidence: Evidence,
    /// Recorded dependencies.
    pub deps: Vec<Dep>,
}

/// First-level fold: view, then steps, then method.
pub fn evaluate(
    receipt_id: u64, w: &World, t: Option<&Thought>, steps: &[Step], m: &Method, p: Policy, basis_id: u64,
) -> Result<Receipt, Stale> {
    let v = view(w, t)?;
    let v = apply_steps(v, steps);
    let result = apply_method(m, &v, p);
    let witness = order_witness(steps);
    let mut deps = vec![Dep {
        kind: DepKind::World,
        id: w.id as u64,
        version: w.version,
    }];
    if let Some(t) = t {
        deps.push(Dep {
            kind: DepKind::Thought,
            id: t.branch as u64,
            version: t.base_version,
        });
    }
    deps.push(Dep {
        kind: DepKind::Method,
        id: m.id() as u64,
        version: 0,
    });
    deps.push(Dep {
        kind: DepKind::Policy,
        id: p.dep_id(),
        version: 0,
    });
    deps.push(Dep {
        kind: DepKind::Steps,
        id: witness,
        version: 0,
    });
    Ok(Receipt {
        receipt_id,
        frame: (w.id, w.version),
        basis_id,
        thought: t.map(|t| t.branch),
        method_id: m.id(),
        order_witness: witness,
        law: "kleene-fold",
        policy: p,
        result,
        evidence: Evidence {
            world_id: w.id,
            world_version: w.version,
            lanes_consulted: m.lanes(),
        },
        deps,
    })
}

// ---------------------------------------------------------------------------
// Second-level (meta) fold: receipts only.
// ---------------------------------------------------------------------------

/// Field-wise difference between two receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diff {
    /// Results differ.
    pub result_changed: bool,
    /// Thought branches differ.
    pub thought_changed: bool,
    /// World id/version or lanes consulted differ.
    pub evidence_changed: bool,
    /// Policy differs.
    pub rule_changed: bool,
    /// Method id differs.
    pub method_changed: bool,
    /// Order witness differs.
    pub order_changed: bool,
}

impl Diff {
    /// True if any field changed.
    pub fn any(&self) -> bool {
        self.result_changed
            || self.thought_changed
            || self.evidence_changed
            || self.rule_changed
            || self.method_changed
            || self.order_changed
    }
}

/// Compare two receipts without touching any world.
pub fn compare(a: &Receipt, b: &Receipt) -> Diff {
    Diff {
        result_changed: a.result != b.result,
        thought_changed: a.thought != b.thought,
        evidence_changed: a.evidence != b.evidence,
        rule_changed: a.policy != b.policy,
        method_changed: a.method_id != b.method_id,
        order_changed: a.order_witness != b.order_witness,
    }
}

/// Why a receipt may not be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalidated {
    /// A dependency's current version differs from the recorded one.
    StaleDep {
        /// Dependency kind.
        kind: DepKind,
        /// Dependency id.
        id: u64,
        /// Version recorded in the receipt.
        recorded: u64,
        /// Version now (`u64::MAX` if the dependency is gone).
        current: u64,
    },
    /// The requested step order is not the one the receipt ran.
    OrderChanged {
        /// Witness recorded.
        recorded: u64,
        /// Witness requested.
        requested: u64,
    },
}

/// Reuse check: every dep version unchanged and the order witness matches.
pub fn may_reuse(
    r: &Receipt, current_version: &dyn Fn(&DepKind, u64) -> Option<u64>, requested_order: u64,
) -> Result<(), Invalidated> {
    for d in &r.deps {
        let current = current_version(&d.kind, d.id).unwrap_or(u64::MAX);
        if current != d.version {
            return Err(Invalidated::StaleDep {
                kind: d.kind,
                id: d.id,
                recorded: d.version,
                current,
            });
        }
    }
    if r.order_witness != requested_order {
        return Err(Invalidated::OrderChanged {
            recorded: r.order_witness,
            requested: requested_order,
        });
    }
    Ok(())
}

/// Receipt over receipts: children, union of deps, Kleene AND of results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateReceipt {
    /// Child receipt ids.
    pub children: Vec<u64>,
    /// De-duplicated union of child deps.
    pub deps: Vec<Dep>,
    /// Kleene AND over child results.
    pub result: Tri,
}

/// Aggregate receipts, preserving provenance.
pub fn aggregate(rs: &[Receipt]) -> AggregateReceipt {
    let mut deps: Vec<Dep> = Vec::new();
    for r in rs {
        for d in &r.deps {
            if !deps.contains(d) {
                deps.push(*d);
            }
        }
    }
    AggregateReceipt {
        children: rs.iter().map(|r| r.receipt_id).collect(),
        deps,
        result: rs.iter().fold(Tri::True, |a, r| a.and(r.result)),
    }
}

/// Pair receipts by `method_id`; report `(next receipt id, diff)` where any field changed.
pub fn which_changed(prev: &[Receipt], next: &[Receipt]) -> Vec<(u64, Diff)> {
    let mut out = Vec::new();
    for n in next {
        if let Some(p) = prev.iter().find(|p| p.method_id == n.method_id) {
            let d = compare(p, n);
            if d.any() {
                out.push((n.receipt_id, d));
            }
        }
    }
    out
}

/// Answers to the meta questions, derived from two receipts alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetaVerdict {
    /// The alt thought branch, if switching thought is what changed the result.
    pub thought_changed_result: Option<u32>,
    /// Evidence consulted differs.
    pub evidence_changed: bool,
    /// Only the policy differs (same evidence, thought, method and order).
    pub only_rule_changed: bool,
}

/// Meta-evaluate `alt` against `base` from receipts only.
pub fn meta_evaluate(base: &Receipt, alt: &Receipt) -> MetaVerdict {
    let d = compare(base, alt);
    MetaVerdict {
        thought_changed_result: if d.thought_changed && d.result_changed {
            alt.thought
        } else {
            None
        },
        evidence_changed: d.evidence_changed,
        only_rule_changed: d.rule_changed
            && !d.evidence_changed
            && !d.thought_changed
            && !d.method_changed
            && !d.order_changed,
    }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Mock data for the probe and its tests.
pub struct Fixture {
    /// World id 7, version 1.
    pub world: World,
    /// Branch 1: flips lane 5 (read by `m1`) to True.
    pub t1: Thought,
    /// Branch 2: changes lane 60, which no method reads.
    pub t2: Thought,
    /// AllOf over lanes 0..8.
    pub m1: Method,
    /// AnyOf over lanes 10..16 (lane 12 Unknown).
    pub m2: Method,
    /// AtLeast 3 over lanes 20..25 (lane 22 Unknown).
    pub m3: Method,
    /// `[SetFact(3, True), NegateFact(3)]`.
    pub steps_fwd: Vec<Step>,
    /// The reverse order.
    pub steps_rev: Vec<Step>,
    /// Basis identity used by the fixture.
    pub basis_id: u64,
}

fn range_mask(lo: u32, hi: u32) -> u64 {
    (lo..hi).fold(0u64, |m, i| m | (1u64 << i))
}

/// Build the fixture.
pub fn fixture() -> Fixture {
    let mut facts = [Tri::False; 64];
    for (i, f) in facts.iter_mut().enumerate() {
        *f = if i % 2 == 0 { Tri::False } else { Tri::True };
    }
    for f in facts.iter_mut().take(8) {
        *f = Tri::True;
    }
    facts[5] = Tri::False;
    for f in &mut facts[10..16] {
        *f = Tri::False;
    }
    facts[12] = Tri::Unknown;
    facts[20] = Tri::True;
    facts[21] = Tri::True;
    facts[22] = Tri::Unknown;
    facts[23] = Tri::False;
    facts[24] = Tri::False;
    for i in [30usize, 31, 40, 50] {
        facts[i] = Tri::Unknown;
    }
    facts[60] = Tri::False;
    Fixture {
        world: World::new(7, 1, facts),
        t1: Thought {
            branch: 1,
            base_version: 1,
            overlay: vec![(5, Tri::True)],
        },
        t2: Thought {
            branch: 2,
            base_version: 1,
            overlay: vec![(60, Tri::True)],
        },
        m1: Method::AllOf {
            id: 101,
            lanes: range_mask(0, 8),
        },
        m2: Method::AnyOf {
            id: 102,
            lanes: range_mask(10, 16),
        },
        m3: Method::AtLeast {
            id: 103,
            lanes: range_mask(20, 25),
            k: 3,
        },
        steps_fwd: vec![Step::SetFact(3, Tri::True), Step::NegateFact(3)],
        steps_rev: vec![Step::NegateFact(3), Step::SetFact(3, Tri::True)],
        basis_id: 0xB,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Version lookup for a world and optional thought; method/policy/steps are version 0.
    fn lookup<'a>(w: &'a World, t: Option<&'a Thought>) -> impl Fn(&DepKind, u64) -> Option<u64> + 'a {
        move |k, id| match k {
            DepKind::World => (id == w.id as u64).then_some(w.version),
            DepKind::Thought => t.and_then(|t| (id == t.branch as u64).then_some(t.base_version)),
            _ => Some(0),
        }
    }

    fn open(fx: &Fixture, id: u64, t: Option<&Thought>, m: &Method) -> Receipt {
        evaluate(id, &fx.world, t, &[], m, Policy::OpenWorld, fx.basis_id).unwrap()
    }

    #[test]
    fn which_thought_changed_the_result() {
        let fx = fixture();
        let base = open(&fx, 1, None, &fx.m1);
        let r1 = open(&fx, 2, Some(&fx.t1), &fx.m1);
        let r2 = open(&fx, 3, Some(&fx.t2), &fx.m1);
        assert_eq!(base.result, Tri::False);
        assert_eq!(r1.result, Tri::True);
        assert_eq!(r2.result, base.result);
        assert_eq!(meta_evaluate(&base, &r1).thought_changed_result, Some(1));
        assert_eq!(meta_evaluate(&base, &r2).thought_changed_result, None);
        let d = compare(&base, &r2);
        assert!(d.thought_changed);
        assert!(!d.result_changed);
        assert_ne!(base, r2);
    }

    #[test]
    fn stale_receipt_after_source_version_change() {
        let fx = fixture();
        let r = open(&fx, 1, None, &fx.m1);
        let w2 = fx.world.advance(&[(5, Tri::True)]);
        assert_eq!(w2.version, fx.world.version + 1);
        assert_eq!(w2.id, fx.world.id);
        let ord = r.order_witness;
        assert_eq!(may_reuse(&r, &lookup(&fx.world, None), ord), Ok(()));
        let e = may_reuse(&r, &lookup(&w2, None), ord);
        assert!(matches!(
            e,
            Err(Invalidated::StaleDep {
                kind: DepKind::World,
                recorded: 1,
                current: 2,
                ..
            })
        ));
        // A vanished dependency is stale with current = u64::MAX.
        let gone = |_: &DepKind, _: u64| None;
        assert!(matches!(may_reuse(&r, &gone, ord), Err(Invalidated::StaleDep { current: u64::MAX, .. })));
    }

    #[test]
    fn policy_change_with_unchanged_evidence() {
        let fx = fixture();
        let o = evaluate(1, &fx.world, None, &[], &fx.m2, Policy::OpenWorld, fx.basis_id).unwrap();
        let c = evaluate(2, &fx.world, None, &[], &fx.m2, Policy::ClosedWorld, fx.basis_id).unwrap();
        let d = compare(&o, &c);
        assert!(d.rule_changed);
        assert!(!d.evidence_changed);
        assert!(d.result_changed);
        assert_eq!(o.result, Tri::Unknown);
        assert_eq!(c.result, Tri::False);
        let v = meta_evaluate(&o, &c);
        assert!(v.only_rule_changed);
        assert!(!v.evidence_changed);
        // Anti-vacuity: a different method is not "only rule changed".
        let other = evaluate(3, &fx.world, None, &[], &fx.m1, Policy::ClosedWorld, fx.basis_id).unwrap();
        assert!(!meta_evaluate(&o, &other).only_rule_changed);
    }

    #[test]
    fn unknown_is_not_false() {
        let fx = fixture();
        let o = evaluate(1, &fx.world, None, &[], &fx.m2, Policy::OpenWorld, fx.basis_id).unwrap();
        let c = evaluate(2, &fx.world, None, &[], &fx.m2, Policy::ClosedWorld, fx.basis_id).unwrap();
        assert_ne!(o.result, Tri::False);
        assert_eq!(o.result, Tri::Unknown);
        assert_eq!(o.policy, Policy::OpenWorld);
        assert_eq!(c.result, Tri::False);
        assert_eq!(c.policy, Policy::ClosedWorld);
        // The lane set really contains an Unknown.
        assert_eq!(fx.world.fact(12), Tri::Unknown);
        assert_ne!(o.deps, c.deps);
        // AtLeast: True+Unknown reaches k, so Unknown; closed world drops it to False.
        let a = open(&fx, 3, None, &fx.m3);
        assert_eq!(a.result, Tri::Unknown);
        let ac = evaluate(4, &fx.world, None, &[], &fx.m3, Policy::ClosedWorld, fx.basis_id).unwrap();
        assert_eq!(ac.result, Tri::False);
    }

    #[test]
    fn method_order_changes_effects() {
        let fx = fixture();
        let fwd = apply_steps(*view_all(&fx).as_ref(), &fx.steps_fwd);
        let rev = apply_steps(*view_all(&fx).as_ref(), &fx.steps_rev);
        assert_eq!(fwd[3], Tri::False);
        assert_eq!(rev[3], Tri::True);
        let (wf, wr) = (order_witness(&fx.steps_fwd), order_witness(&fx.steps_rev));
        assert_ne!(wf, wr);
        let r = evaluate(1, &fx.world, None, &fx.steps_fwd, &fx.m1, Policy::OpenWorld, fx.basis_id).unwrap();
        assert_eq!(r.order_witness, wf);
        let lk = lookup(&fx.world, None);
        assert_eq!(may_reuse(&r, &lk, wf), Ok(()));
        assert_eq!(
            may_reuse(&r, &lk, wr),
            Err(Invalidated::OrderChanged {
                recorded: wf,
                requested: wr
            })
        );
        let rr = evaluate(2, &fx.world, None, &fx.steps_rev, &fx.m1, Policy::OpenWorld, fx.basis_id).unwrap();
        assert!(compare(&r, &rr).order_changed);
    }

    fn view_all(fx: &Fixture) -> Box<[Tri; 64]> {
        Box::new(view(&fx.world, None).unwrap())
    }

    fn covers(agg: &AggregateReceipt, rs: &[Receipt]) -> bool {
        rs.iter()
            .all(|r| agg.children.contains(&r.receipt_id) && r.deps.iter().all(|d| agg.deps.contains(d)))
    }

    #[test]
    fn provenance_survives_aggregation() {
        let fx = fixture();
        let rs = vec![open(&fx, 1, Some(&fx.t1), &fx.m1), open(&fx, 2, None, &fx.m2), open(&fx, 3, None, &fx.m3)];
        let agg = aggregate(&rs);
        assert_eq!(agg.children, vec![1, 2, 3]);
        assert_eq!(agg.result, Tri::Unknown);
        assert!(covers(&agg, &rs));
        for r in &rs {
            assert!(agg.deps.contains(&Dep {
                kind: DepKind::World,
                id: 7,
                version: 1
            }));
            assert!(agg
                .deps
                .iter()
                .any(|d| d.kind == DepKind::Method && d.id == r.method_id as u64));
        }
        // Deduplicated: one World dep despite three children.
        assert_eq!(agg.deps.iter().filter(|d| d.kind == DepKind::World).count(), 1);
        // Negative: dropping m1's Method dep is detected.
        let mut cut = agg.clone();
        let before = cut.deps.len();
        cut.deps
            .retain(|d| !(d.kind == DepKind::Method && d.id == rs[0].method_id as u64));
        assert!(cut.deps.len() < before);
        assert!(!covers(&cut, &rs));
    }

    #[test]
    fn unchanged_dependencies_permit_reuse() {
        let fx = fixture();
        let r = open(&fx, 1, Some(&fx.t1), &fx.m1);
        assert_eq!(may_reuse(&r, &lookup(&fx.world, Some(&fx.t1)), r.order_witness), Ok(()));
        // Anti-vacuity: the same lookup without the thought makes the Thought dep vanish.
        assert!(may_reuse(&r, &lookup(&fx.world, None), r.order_witness).is_err());
        let a = open(&fx, 2, Some(&fx.t1), &fx.m1);
        assert!(which_changed(std::slice::from_ref(&r), &[a]).is_empty());
    }

    #[test]
    fn stale_thought_is_refused() {
        let fx = fixture();
        let old = Thought {
            branch: 9,
            base_version: 0,
            overlay: vec![],
        };
        let e = evaluate(1, &fx.world, Some(&old), &[], &fx.m1, Policy::OpenWorld, fx.basis_id);
        assert_eq!(e, Err(Stale::World { expected: 1, found: 0 }));
        assert!(evaluate(2, &fx.world, Some(&fx.t1), &[], &fx.m1, Policy::OpenWorld, fx.basis_id).is_ok());
    }

    #[test]
    fn meta_fold_reads_receipts_only() {
        let (base, alt, inner_diff, inner_verdict) = {
            let fx = fixture();
            let base = open(&fx, 1, None, &fx.m1);
            let alt = open(&fx, 2, Some(&fx.t1), &fx.m1);
            let d = compare(&base, &alt);
            let v = meta_evaluate(&base, &alt);
            drop(fx);
            (base.clone(), alt.clone(), d, v)
        };
        let outer_diff = compare(&base, &alt);
        let outer_verdict = meta_evaluate(&base, &alt);
        assert_eq!(outer_diff, inner_diff);
        assert_eq!(outer_verdict, inner_verdict);
        assert!(outer_diff.result_changed);
        assert_eq!(outer_verdict.thought_changed_result, Some(1));
        let changed = which_changed(&[base], &[alt]);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].0, 2);
    }
}
