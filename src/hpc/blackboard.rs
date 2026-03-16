//! Blackboard: HashMap-based arena allocator for heterogeneous typed data.
//!
//! A blackboard is a shared workspace where multiple subsystems can publish
//! and read typed data by string key. It uses type-erased storage internally
//! and provides safe typed access through `std::any::Any`.
//!
//! Ported verbatim from rustynum-core/blackboard.rs.

use std::any::Any;
use std::collections::HashMap;

/// Type-erased slot in the blackboard.
struct Slot {
    data: Box<dyn Any>,
}

/// HashMap-based arena allocator for heterogeneous typed data.
///
/// Stores values of any `'static` type keyed by string. Values are type-checked
/// at runtime via `std::any::Any` downcasting.
///
/// # Example
///
/// ```
/// use ndarray::hpc::blackboard::Blackboard;
///
/// let mut bb = Blackboard::new();
/// bb.alloc_f32("temperature", 98.6);
/// bb.alloc_string("name", "Alice".to_string());
///
/// assert_eq!(bb.get_f32("temperature"), Some(&98.6));
/// assert_eq!(bb.get_string("name"), Some(&"Alice".to_string()));
/// assert_eq!(bb.get_f32("missing"), None);
/// ```
pub struct Blackboard {
    slots: HashMap<String, Slot>,
}

impl Blackboard {
    /// Create an empty blackboard.
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    /// Create a blackboard with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: HashMap::with_capacity(cap),
        }
    }

    /// Number of entries in the blackboard.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the blackboard is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Check whether a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.slots.contains_key(key)
    }

    /// Remove an entry by key. Returns true if an entry was removed.
    pub fn remove(&mut self, key: &str) -> bool {
        self.slots.remove(key).is_some()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// List all keys (unordered).
    pub fn keys(&self) -> Vec<&str> {
        self.slots.keys().map(|s| s.as_str()).collect()
    }

    // -----------------------------------------------------------------------
    // Generic alloc / get
    // -----------------------------------------------------------------------

    /// Allocate (insert or replace) a value of any `'static` type.
    pub fn alloc<T: 'static>(&mut self, key: &str, value: T) {
        self.slots.insert(
            key.to_string(),
            Slot { data: Box::new(value) },
        );
    }

    /// Get an immutable reference to a value, returning `None` if the key is
    /// missing or the type does not match.
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.slots.get(key).and_then(|slot| slot.data.downcast_ref::<T>())
    }

    /// Get a mutable reference to a value, returning `None` if the key is
    /// missing or the type does not match.
    pub fn get_mut<T: 'static>(&mut self, key: &str) -> Option<&mut T> {
        self.slots
            .get_mut(key)
            .and_then(|slot| slot.data.downcast_mut::<T>())
    }

    // -----------------------------------------------------------------------
    // Typed alloc helpers
    // -----------------------------------------------------------------------

    /// Allocate an `f32` value.
    pub fn alloc_f32(&mut self, key: &str, value: f32) {
        self.alloc(key, value);
    }

    /// Allocate an `f64` value.
    pub fn alloc_f64(&mut self, key: &str, value: f64) {
        self.alloc(key, value);
    }

    /// Allocate an `i32` value.
    pub fn alloc_i32(&mut self, key: &str, value: i32) {
        self.alloc(key, value);
    }

    /// Allocate an `i64` value.
    pub fn alloc_i64(&mut self, key: &str, value: i64) {
        self.alloc(key, value);
    }

    /// Allocate a `u32` value.
    pub fn alloc_u32(&mut self, key: &str, value: u32) {
        self.alloc(key, value);
    }

    /// Allocate a `u64` value.
    pub fn alloc_u64(&mut self, key: &str, value: u64) {
        self.alloc(key, value);
    }

    /// Allocate a `usize` value.
    pub fn alloc_usize(&mut self, key: &str, value: usize) {
        self.alloc(key, value);
    }

    /// Allocate a `bool` value.
    pub fn alloc_bool(&mut self, key: &str, value: bool) {
        self.alloc(key, value);
    }

    /// Allocate a `String` value.
    pub fn alloc_string(&mut self, key: &str, value: String) {
        self.alloc(key, value);
    }

    /// Allocate a `Vec<f32>` value.
    pub fn alloc_vec_f32(&mut self, key: &str, value: Vec<f32>) {
        self.alloc(key, value);
    }

    /// Allocate a `Vec<f64>` value.
    pub fn alloc_vec_f64(&mut self, key: &str, value: Vec<f64>) {
        self.alloc(key, value);
    }

    /// Allocate a `Vec<u8>` value.
    pub fn alloc_vec_u8(&mut self, key: &str, value: Vec<u8>) {
        self.alloc(key, value);
    }

    /// Allocate a `Vec<i32>` value.
    pub fn alloc_vec_i32(&mut self, key: &str, value: Vec<i32>) {
        self.alloc(key, value);
    }

    // -----------------------------------------------------------------------
    // Typed get helpers
    // -----------------------------------------------------------------------

    /// Get an `f32` value.
    pub fn get_f32(&self, key: &str) -> Option<&f32> {
        self.get(key)
    }

    /// Get an `f64` value.
    pub fn get_f64(&self, key: &str) -> Option<&f64> {
        self.get(key)
    }

    /// Get an `i32` value.
    pub fn get_i32(&self, key: &str) -> Option<&i32> {
        self.get(key)
    }

    /// Get an `i64` value.
    pub fn get_i64(&self, key: &str) -> Option<&i64> {
        self.get(key)
    }

    /// Get a `u32` value.
    pub fn get_u32(&self, key: &str) -> Option<&u32> {
        self.get(key)
    }

    /// Get a `u64` value.
    pub fn get_u64(&self, key: &str) -> Option<&u64> {
        self.get(key)
    }

    /// Get a `usize` value.
    pub fn get_usize(&self, key: &str) -> Option<&usize> {
        self.get(key)
    }

    /// Get a `bool` value.
    pub fn get_bool(&self, key: &str) -> Option<&bool> {
        self.get(key)
    }

    /// Get a `String` value.
    pub fn get_string(&self, key: &str) -> Option<&String> {
        self.get(key)
    }

    /// Get a `Vec<f32>` value.
    pub fn get_vec_f32(&self, key: &str) -> Option<&Vec<f32>> {
        self.get(key)
    }

    /// Get a `Vec<f64>` value.
    pub fn get_vec_f64(&self, key: &str) -> Option<&Vec<f64>> {
        self.get(key)
    }

    /// Get a `Vec<u8>` value.
    pub fn get_vec_u8(&self, key: &str) -> Option<&Vec<u8>> {
        self.get(key)
    }

    /// Get a `Vec<i32>` value.
    pub fn get_vec_i32(&self, key: &str) -> Option<&Vec<i32>> {
        self.get(key)
    }

    // -----------------------------------------------------------------------
    // Typed get_mut helpers
    // -----------------------------------------------------------------------

    /// Get a mutable `f32` reference.
    pub fn get_mut_f32(&mut self, key: &str) -> Option<&mut f32> {
        self.get_mut(key)
    }

    /// Get a mutable `f64` reference.
    pub fn get_mut_f64(&mut self, key: &str) -> Option<&mut f64> {
        self.get_mut(key)
    }

    /// Get a mutable `Vec<f32>` reference.
    pub fn get_mut_vec_f32(&mut self, key: &str) -> Option<&mut Vec<f32>> {
        self.get_mut(key)
    }

    /// Get a mutable `Vec<f64>` reference.
    pub fn get_mut_vec_f64(&mut self, key: &str) -> Option<&mut Vec<f64>> {
        self.get_mut(key)
    }

    /// Get a mutable `Vec<u8>` reference.
    pub fn get_mut_vec_u8(&mut self, key: &str) -> Option<&mut Vec<u8>> {
        self.get_mut(key)
    }

    // -----------------------------------------------------------------------
    // borrow_2_mut / borrow_3_mut — simultaneous mutable access to distinct keys
    // -----------------------------------------------------------------------

    /// Borrow two `Vec<f32>` values mutably at the same time.
    ///
    /// # Panics
    ///
    /// Panics if `key_a == key_b` (aliasing), or if either key is missing or
    /// has the wrong type.
    ///
    /// # Safety note
    ///
    /// Uses raw pointers internally to circumvent the borrow checker for
    /// distinct HashMap entries. This is safe because the two keys are verified
    /// to be different, guaranteeing non-overlapping memory.
    pub fn borrow_2_mut_vec_f32(
        &mut self,
        key_a: &str,
        key_b: &str,
    ) -> (&mut Vec<f32>, &mut Vec<f32>) {
        assert_ne!(key_a, key_b, "borrow_2_mut: keys must be distinct");
        // SAFETY: key_a != key_b, so the two mutable references point to
        // different HashMap entries and cannot alias.
        unsafe {
            let ptr = &mut self.slots as *mut HashMap<String, Slot>;
            let a = (*ptr)
                .get_mut(key_a)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_a))
                .data
                .downcast_mut::<Vec<f32>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_a));
            let b = (*ptr)
                .get_mut(key_b)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_b))
                .data
                .downcast_mut::<Vec<f32>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_b));
            (a, b)
        }
    }

    /// Borrow two `Vec<f64>` values mutably at the same time.
    ///
    /// # Panics
    ///
    /// Panics if `key_a == key_b`, or if either key is missing or wrong type.
    pub fn borrow_2_mut_vec_f64(
        &mut self,
        key_a: &str,
        key_b: &str,
    ) -> (&mut Vec<f64>, &mut Vec<f64>) {
        assert_ne!(key_a, key_b, "borrow_2_mut: keys must be distinct");
        // SAFETY: key_a != key_b => distinct entries, no aliasing.
        unsafe {
            let ptr = &mut self.slots as *mut HashMap<String, Slot>;
            let a = (*ptr)
                .get_mut(key_a)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_a))
                .data
                .downcast_mut::<Vec<f64>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_a));
            let b = (*ptr)
                .get_mut(key_b)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_b))
                .data
                .downcast_mut::<Vec<f64>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_b));
            (a, b)
        }
    }

    /// Borrow two `Vec<u8>` values mutably at the same time.
    ///
    /// # Panics
    ///
    /// Panics if `key_a == key_b`, or if either key is missing or wrong type.
    pub fn borrow_2_mut_vec_u8(
        &mut self,
        key_a: &str,
        key_b: &str,
    ) -> (&mut Vec<u8>, &mut Vec<u8>) {
        assert_ne!(key_a, key_b, "borrow_2_mut: keys must be distinct");
        // SAFETY: key_a != key_b => distinct entries, no aliasing.
        unsafe {
            let ptr = &mut self.slots as *mut HashMap<String, Slot>;
            let a = (*ptr)
                .get_mut(key_a)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_a))
                .data
                .downcast_mut::<Vec<u8>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_a));
            let b = (*ptr)
                .get_mut(key_b)
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' not found", key_b))
                .data
                .downcast_mut::<Vec<u8>>()
                .unwrap_or_else(|| panic!("borrow_2_mut: key '{}' type mismatch", key_b));
            (a, b)
        }
    }

    /// Borrow three `Vec<f32>` values mutably at the same time.
    ///
    /// # Panics
    ///
    /// Panics if any two keys are equal, or if any key is missing or wrong type.
    pub fn borrow_3_mut_vec_f32(
        &mut self,
        key_a: &str,
        key_b: &str,
        key_c: &str,
    ) -> (&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>) {
        assert_ne!(key_a, key_b, "borrow_3_mut: keys must be distinct (a == b)");
        assert_ne!(key_b, key_c, "borrow_3_mut: keys must be distinct (b == c)");
        assert_ne!(key_a, key_c, "borrow_3_mut: keys must be distinct (a == c)");
        // SAFETY: all three keys are distinct => distinct entries, no aliasing.
        unsafe {
            let ptr = &mut self.slots as *mut HashMap<String, Slot>;
            let a = (*ptr)
                .get_mut(key_a)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_a))
                .data
                .downcast_mut::<Vec<f32>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_a));
            let b = (*ptr)
                .get_mut(key_b)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_b))
                .data
                .downcast_mut::<Vec<f32>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_b));
            let c = (*ptr)
                .get_mut(key_c)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_c))
                .data
                .downcast_mut::<Vec<f32>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_c));
            (a, b, c)
        }
    }

    /// Borrow three `Vec<f64>` values mutably at the same time.
    ///
    /// # Panics
    ///
    /// Panics if any two keys are equal, or if any key is missing or wrong type.
    pub fn borrow_3_mut_vec_f64(
        &mut self,
        key_a: &str,
        key_b: &str,
        key_c: &str,
    ) -> (&mut Vec<f64>, &mut Vec<f64>, &mut Vec<f64>) {
        assert_ne!(key_a, key_b, "borrow_3_mut: keys must be distinct (a == b)");
        assert_ne!(key_b, key_c, "borrow_3_mut: keys must be distinct (b == c)");
        assert_ne!(key_a, key_c, "borrow_3_mut: keys must be distinct (a == c)");
        // SAFETY: all three keys are distinct => distinct entries, no aliasing.
        unsafe {
            let ptr = &mut self.slots as *mut HashMap<String, Slot>;
            let a = (*ptr)
                .get_mut(key_a)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_a))
                .data
                .downcast_mut::<Vec<f64>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_a));
            let b = (*ptr)
                .get_mut(key_b)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_b))
                .data
                .downcast_mut::<Vec<f64>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_b));
            let c = (*ptr)
                .get_mut(key_c)
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' not found", key_c))
                .data
                .downcast_mut::<Vec<f64>>()
                .unwrap_or_else(|| panic!("borrow_3_mut: key '{}' type mismatch", key_c));
            (a, b, c)
        }
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let bb = Blackboard::new();
        assert!(bb.is_empty());
        assert_eq!(bb.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let bb = Blackboard::with_capacity(64);
        assert!(bb.is_empty());
    }

    #[test]
    fn test_alloc_get_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("x", 3.14);
        assert_eq!(bb.get_f32("x"), Some(&3.14));
        assert_eq!(bb.get_f32("y"), None);
    }

    #[test]
    fn test_alloc_get_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("pi", std::f64::consts::PI);
        assert_eq!(bb.get_f64("pi"), Some(&std::f64::consts::PI));
    }

    #[test]
    fn test_alloc_get_i32() {
        let mut bb = Blackboard::new();
        bb.alloc_i32("count", -42);
        assert_eq!(bb.get_i32("count"), Some(&-42));
    }

    #[test]
    fn test_alloc_get_i64() {
        let mut bb = Blackboard::new();
        bb.alloc_i64("big", i64::MAX);
        assert_eq!(bb.get_i64("big"), Some(&i64::MAX));
    }

    #[test]
    fn test_alloc_get_u32() {
        let mut bb = Blackboard::new();
        bb.alloc_u32("flags", 0xDEAD);
        assert_eq!(bb.get_u32("flags"), Some(&0xDEAD));
    }

    #[test]
    fn test_alloc_get_u64() {
        let mut bb = Blackboard::new();
        bb.alloc_u64("hash", 0xCAFEBABE);
        assert_eq!(bb.get_u64("hash"), Some(&0xCAFEBABE));
    }

    #[test]
    fn test_alloc_get_usize() {
        let mut bb = Blackboard::new();
        bb.alloc_usize("idx", 999);
        assert_eq!(bb.get_usize("idx"), Some(&999));
    }

    #[test]
    fn test_alloc_get_bool() {
        let mut bb = Blackboard::new();
        bb.alloc_bool("ready", true);
        assert_eq!(bb.get_bool("ready"), Some(&true));
    }

    #[test]
    fn test_alloc_get_string() {
        let mut bb = Blackboard::new();
        bb.alloc_string("name", "Alice".to_string());
        assert_eq!(bb.get_string("name"), Some(&"Alice".to_string()));
    }

    #[test]
    fn test_alloc_get_vec_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("weights", vec![1.0, 2.0, 3.0]);
        assert_eq!(bb.get_vec_f32("weights"), Some(&vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_alloc_get_vec_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f64("data", vec![1.0, 2.0]);
        assert_eq!(bb.get_vec_f64("data"), Some(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_alloc_get_vec_u8() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_u8("bytes", vec![0xFF, 0x00]);
        assert_eq!(bb.get_vec_u8("bytes"), Some(&vec![0xFF, 0x00]));
    }

    #[test]
    fn test_alloc_get_vec_i32() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_i32("ints", vec![-1, 0, 1]);
        assert_eq!(bb.get_vec_i32("ints"), Some(&vec![-1, 0, 1]));
    }

    #[test]
    fn test_type_mismatch_returns_none() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("x", 1.0);
        assert_eq!(bb.get_f64("x"), None); // stored as f32, not f64
        assert_eq!(bb.get_i32("x"), None);
    }

    #[test]
    fn test_overwrite() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("x", 1.0);
        bb.alloc_f32("x", 2.0);
        assert_eq!(bb.get_f32("x"), Some(&2.0));
        assert_eq!(bb.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut bb = Blackboard::new();
        assert!(!bb.contains("x"));
        bb.alloc_f32("x", 1.0);
        assert!(bb.contains("x"));
    }

    #[test]
    fn test_remove() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("x", 1.0);
        assert!(bb.remove("x"));
        assert!(!bb.contains("x"));
        assert!(!bb.remove("x"));
    }

    #[test]
    fn test_clear() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("a", 1.0);
        bb.alloc_i32("b", 2);
        bb.alloc_string("c", "three".to_string());
        assert_eq!(bb.len(), 3);
        bb.clear();
        assert!(bb.is_empty());
    }

    #[test]
    fn test_keys() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("alpha", 1.0);
        bb.alloc_f32("beta", 2.0);
        let mut keys = bb.keys();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_get_mut_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("x", 1.0);
        if let Some(val) = bb.get_mut_f32("x") {
            *val = 99.0;
        }
        assert_eq!(bb.get_f32("x"), Some(&99.0));
    }

    #[test]
    fn test_get_mut_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("x", 1.0);
        if let Some(val) = bb.get_mut_f64("x") {
            *val = 99.0;
        }
        assert_eq!(bb.get_f64("x"), Some(&99.0));
    }

    #[test]
    fn test_get_mut_vec_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("v", vec![1.0, 2.0]);
        if let Some(v) = bb.get_mut_vec_f32("v") {
            v.push(3.0);
        }
        assert_eq!(bb.get_vec_f32("v"), Some(&vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_get_mut_vec_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f64("v", vec![1.0]);
        if let Some(v) = bb.get_mut_vec_f64("v") {
            v.push(2.0);
        }
        assert_eq!(bb.get_vec_f64("v"), Some(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_get_mut_vec_u8() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_u8("buf", vec![0x01]);
        if let Some(v) = bb.get_mut_vec_u8("buf") {
            v.push(0x02);
        }
        assert_eq!(bb.get_vec_u8("buf"), Some(&vec![0x01, 0x02]));
    }

    #[test]
    fn test_borrow_2_mut_vec_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("a", vec![1.0, 2.0]);
        bb.alloc_vec_f32("b", vec![3.0, 4.0]);
        let (a, b) = bb.borrow_2_mut_vec_f32("a", "b");
        a[0] += b[0]; // a[0] = 1.0 + 3.0 = 4.0
        b[1] += a[1]; // b[1] = 4.0 + 2.0 = 6.0
        assert_eq!(bb.get_vec_f32("a"), Some(&vec![4.0, 2.0]));
        assert_eq!(bb.get_vec_f32("b"), Some(&vec![3.0, 6.0]));
    }

    #[test]
    fn test_borrow_2_mut_vec_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f64("x", vec![10.0]);
        bb.alloc_vec_f64("y", vec![20.0]);
        let (x, y) = bb.borrow_2_mut_vec_f64("x", "y");
        x[0] += y[0];
        assert_eq!(bb.get_vec_f64("x"), Some(&vec![30.0]));
    }

    #[test]
    fn test_borrow_2_mut_vec_u8() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_u8("p", vec![0x10]);
        bb.alloc_vec_u8("q", vec![0x20]);
        let (p, q) = bb.borrow_2_mut_vec_u8("p", "q");
        p[0] |= q[0];
        assert_eq!(bb.get_vec_u8("p"), Some(&vec![0x30]));
    }

    #[test]
    #[should_panic(expected = "keys must be distinct")]
    fn test_borrow_2_mut_same_key_panics() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("a", vec![1.0]);
        bb.borrow_2_mut_vec_f32("a", "a");
    }

    #[test]
    fn test_borrow_3_mut_vec_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("a", vec![1.0]);
        bb.alloc_vec_f32("b", vec![2.0]);
        bb.alloc_vec_f32("c", vec![3.0]);
        let (a, b, c) = bb.borrow_3_mut_vec_f32("a", "b", "c");
        a[0] = b[0] + c[0]; // a = 2 + 3 = 5
        assert_eq!(bb.get_vec_f32("a"), Some(&vec![5.0]));
    }

    #[test]
    fn test_borrow_3_mut_vec_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f64("x", vec![1.0]);
        bb.alloc_vec_f64("y", vec![2.0]);
        bb.alloc_vec_f64("z", vec![3.0]);
        let (x, y, z) = bb.borrow_3_mut_vec_f64("x", "y", "z");
        z[0] = x[0] + y[0];
        assert_eq!(bb.get_vec_f64("z"), Some(&vec![3.0]));
    }

    #[test]
    #[should_panic(expected = "keys must be distinct")]
    fn test_borrow_3_mut_duplicate_panics() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("a", vec![1.0]);
        bb.alloc_vec_f32("b", vec![2.0]);
        bb.borrow_3_mut_vec_f32("a", "b", "a");
    }

    #[test]
    fn test_generic_alloc_custom_type() {
        #[derive(Debug, PartialEq)]
        struct Point { x: f32, y: f32 }

        let mut bb = Blackboard::new();
        bb.alloc("origin", Point { x: 0.0, y: 0.0 });
        assert_eq!(bb.get::<Point>("origin"), Some(&Point { x: 0.0, y: 0.0 }));
    }

    #[test]
    fn test_mixed_types() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("temperature", 98.6);
        bb.alloc_i32("count", 42);
        bb.alloc_string("label", "test".to_string());
        bb.alloc_bool("active", true);
        bb.alloc_vec_f32("weights", vec![0.1, 0.9]);

        assert_eq!(bb.len(), 5);
        assert_eq!(bb.get_f32("temperature"), Some(&98.6));
        assert_eq!(bb.get_i32("count"), Some(&42));
        assert_eq!(bb.get_string("label"), Some(&"test".to_string()));
        assert_eq!(bb.get_bool("active"), Some(&true));
        assert_eq!(bb.get_vec_f32("weights"), Some(&vec![0.1, 0.9]));
    }

    #[test]
    fn test_default() {
        let bb = Blackboard::default();
        assert!(bb.is_empty());
    }
}
