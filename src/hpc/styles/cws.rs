//! #18 Context Window Simulation — snapshot and restore Base17 regions.
//! Science: Kanerva (1988) sparse distributed memory.

use super::super::bgz17_bridge::Base17;

pub struct Snapshot {
    pub entries: Vec<(u16, Base17)>,
}

pub fn snapshot_region(corpus: &[(u16, Base17)]) -> Snapshot {
    Snapshot { entries: corpus.to_vec() }
}

pub fn restore_region(snapshot: &Snapshot) -> Vec<(u16, Base17)> {
    snapshot.entries.clone()
}

pub fn merge_snapshots(a: &Snapshot, b: &Snapshot) -> Snapshot {
    let mut merged = a.entries.clone();
    for (addr, fp) in &b.entries {
        if !merged.iter().any(|(a, _)| a == addr) { merged.push((*addr, fp.clone())); }
    }
    Snapshot { entries: merged }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_snapshot_roundtrip() {
        let data = vec![(0u16, Base17 { dims: [42; 17] }), (1, Base17 { dims: [99; 17] })];
        let snap = snapshot_region(&data);
        let restored = restore_region(&snap);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].1.dims[0], 42);
    }
}
