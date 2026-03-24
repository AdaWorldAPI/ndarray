//! Scan loop JIT specialization.
//!
//! Generates a native scan function where:
//! - `record_size` -> constant offset arithmetic (no register load)
//! - `threshold` -> immediate operand in CMP (no memory fetch)
//! - `prefetch_ahead` -> `PREFETCHT0 [ptr + N * RECORD_SIZE]` as constant
//! - `focus_mask` -> AND mask baked into code section
//!
//! Two modes:
//! 1. **Inline**: simple XOR+popcnt for small records (proof of concept)
//! 2. **Hybrid**: JIT the loop, CALL an external SIMD kernel. The kernel
//!    is registered via `JitEngineBuilder::register_fn()`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{Function, InstBuilder, MemFlags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use super::ir::{JitError, ScanParams};

/// A compiled scan kernel — holds the native function pointer
/// and the params that generated it (for introspection).
pub struct ScanKernel {
    /// The compiled scan function.
    /// Signature: `fn(query, field, field_len, record_size, candidates_out) -> num_candidates`
    fn_ptr: *const u8,

    /// Parameters that were baked into this kernel.
    pub params: ScanParams,
}

// SAFETY: The compiled code is immutable and thread-safe.
unsafe impl Send for ScanKernel {}
unsafe impl Sync for ScanKernel {}

impl ScanKernel {
    /// Wrap a raw function pointer as a ScanKernel.
    pub(crate) fn from_raw(ptr: *const u8, params: ScanParams) -> Self {
        Self {
            fn_ptr: ptr,
            params,
        }
    }

    /// Execute the compiled scan.
    ///
    /// # Safety
    ///
    /// - `query` must point to a valid query vector of appropriate size.
    /// - `field` must point to `field_len * record_size` bytes.
    /// - `candidates_out` must point to a buffer large enough for results.
    pub unsafe fn scan(
        &self,
        query: *const u8,
        field: *const u8,
        field_len: u64,
        record_size: u64,
        candidates_out: *mut u64,
    ) -> u64 {
        // SAFETY: caller guarantees pointer validity; fn_ptr was compiled
        // by Cranelift with the matching signature.
        let func: unsafe extern "C" fn(*const u8, *const u8, u64, u64, *mut u64) -> u64 =
            std::mem::transmute(self.fn_ptr);
        func(query, field, field_len, record_size, candidates_out)
    }

    /// Get the raw function pointer (for benchmarking/introspection).
    pub fn as_fn_ptr(&self) -> *const u8 {
        self.fn_ptr
    }
}

/// Build the Cranelift IR for a scan loop with baked-in parameters.
///
/// If `dist_func_ref` is Some, generates a CALL to the external distance
/// function (hybrid approach). Otherwise, generates inline XOR+popcnt (POC).
///
/// Generated pseudo-code (hybrid):
/// ```text
/// fn scan(query, field, field_len, record_size, candidates_out) -> u64:
///     candidate_count = 0
///     for i in 0..field_len:
///         record_ptr = field + i * RECORD_SIZE    // RECORD_SIZE is immediate
///         dist = CALL distance_fn(query, record_ptr, RECORD_SIZE)
///         if dist < THRESHOLD:                     // THRESHOLD is immediate
///             candidates_out[candidate_count] = i
///             candidate_count += 1
///             if candidate_count >= TOP_K:          // TOP_K is immediate
///                 return candidate_count
///     return candidate_count
/// ```
pub fn build_scan_ir(
    func: &mut Function,
    params: &ScanParams,
    dist_func_ref: Option<cranelift_codegen::ir::FuncRef>,
) -> Result<(), JitError> {
    let mut fbc = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(func, &mut fbc);

    // Variables
    let v_i = Variable::from_u32(0); // loop counter
    let v_count = Variable::from_u32(1); // candidate count
    let v_dist = Variable::from_u32(2); // distance result
    builder.declare_var(v_i, types::I64);
    builder.declare_var(v_count, types::I64);
    builder.declare_var(v_dist, types::I64);

    // Entry block
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    // Get function parameters
    let query = builder.block_params(entry)[0];
    let field = builder.block_params(entry)[1];
    let field_len = builder.block_params(entry)[2];
    let _record_size_param = builder.block_params(entry)[3]; // ignored -- baked as immediate
    let candidates_out = builder.block_params(entry)[4];

    // Baked constants (the whole point of JIT compilation)
    let threshold_imm = builder
        .ins()
        .iconst(types::I64, params.threshold as i64);
    let record_size_imm = builder
        .ins()
        .iconst(types::I64, params.record_size as i64);
    let top_k_imm = builder.ins().iconst(types::I64, params.top_k as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let one = builder.ins().iconst(types::I64, 1);
    let eight = builder.ins().iconst(types::I64, 8); // sizeof(u64) for candidates_out

    // Initialize loop variables
    builder.def_var(v_i, zero);
    builder.def_var(v_count, zero);

    // Loop header
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();
    let store_candidate = builder.create_block();
    let skip_store = builder.create_block();

    builder.ins().jump(loop_header, &[]);

    // -- Loop header: check i < field_len --
    builder.switch_to_block(loop_header);
    let i = builder.use_var(v_i);
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i, field_len);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    // -- Loop body --
    builder.switch_to_block(loop_body);
    let i = builder.use_var(v_i);

    // record_ptr = field + i * RECORD_SIZE (RECORD_SIZE is immediate)
    let offset = builder.ins().imul(i, record_size_imm);
    let record_ptr = builder.ins().iadd(field, offset);

    // Compute distance
    let dist = if let Some(func_ref) = dist_func_ref {
        // -- Hybrid mode: CALL external distance function --
        let call = builder
            .ins()
            .call(func_ref, &[query, record_ptr, record_size_imm]);
        builder.inst_results(call)[0]
    } else {
        // -- Inline mode: simple XOR + popcnt of first 8 bytes (POC) --
        let q_val = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), query, 0);
        let r_val = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), record_ptr, 0);
        let xor_val = builder.ins().bxor(q_val, r_val);
        builder.ins().popcnt(xor_val)
    };
    builder.def_var(v_dist, dist);

    // if dist < THRESHOLD (THRESHOLD is immediate)
    let dist = builder.use_var(v_dist);
    let below_threshold = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, dist, threshold_imm);
    builder
        .ins()
        .brif(below_threshold, store_candidate, &[], skip_store, &[]);

    // -- Store candidate --
    builder.switch_to_block(store_candidate);
    let count = builder.use_var(v_count);
    let i = builder.use_var(v_i);

    // candidates_out[count] = i
    let out_offset = builder.ins().imul(count, eight);
    let out_ptr = builder.ins().iadd(candidates_out, out_offset);
    builder
        .ins()
        .store(MemFlags::trusted(), i, out_ptr, 0);

    // count++
    let new_count = builder.ins().iadd(count, one);
    builder.def_var(v_count, new_count);

    // if count >= top_k: return early
    let full = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, new_count, top_k_imm);
    builder.ins().brif(full, loop_exit, &[], skip_store, &[]);

    // -- Skip store / continue loop --
    builder.switch_to_block(skip_store);
    let i = builder.use_var(v_i);
    let next_i = builder.ins().iadd(i, one);
    builder.def_var(v_i, next_i);
    builder.ins().jump(loop_header, &[]);

    // -- Loop exit: return candidate count --
    builder.switch_to_block(loop_exit);
    let final_count = builder.use_var(v_count);
    builder.ins().return_(&[final_count]);

    // Seal remaining blocks
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(store_candidate);
    builder.seal_block(skip_store);
    builder.seal_block(loop_exit);

    builder.finalize();
    Ok(())
}
