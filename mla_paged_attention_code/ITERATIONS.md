# MLA Paged Attention Optimization Iterations

## Stage A: Initial Baseline (mla_paged_baseline.cu)

**Status:** ✅ CORRECT - Ready for online submission
**Date:** 2026-08-17
**SHA256:** `c66924dd9a7ecddfe74759602215f8cdc109d9f9ddb13836bc80954c34defbd6`

### Implementation

Simple warp-per-(batch, head) approach:
- One warp processes one (batch_idx, head_idx) pair
- Each warp reads entire KV sequence for its batch
- Uses scalar loads (2 bytes at a time)
- Online softmax with correct FP32 accumulation
- Reuses ckv as V (matches MLA specification)

### Correctness

✅ **All 22 test cases pass:**
- `match_ratio = 1.0` for all cases
- `severe_error_count = 0` for all cases
- `max_error ≤ 0.00390625` (well within tolerance)
- Tested with identity and permuted page tables

✅ **Key correctness points verified:**
1. `sm_scale = 1.0 / sqrt(head_dim_ckv + head_dim_kpe)` = 1/sqrt(576) ✓
2. V equals ckv (no separate V tensor) ✓
3. Respects `kv_lens[b]` for actual sequence length ✓
4. Page table indirection via `kv_indices` ✓
5. FP32 online softmax with proper m/d tracking ✓
6. Warp reduction for dot products ✓

### Performance

❌ **Very slow (90-1050x slower than FlashInfer):**

| Case | B | L | H | Baseline | FlashInfer | Slowdown | BW |
|------|---|------|-----|----------|------------|----------|-----|
| b1_l1024_h64 | 1 | 1024 | 64 | 5.74ms | 0.038ms | 151x | 0.2 GB/s |
| b16_l1024_h128 | 16 | 1024 | 128 | 20.7ms | 0.327ms | 63x | 3.6 GB/s |
| b1_l16384_h128 | 1 | 16384 | 128 | 91ms | 0.087ms | 1046x | 0.2 GB/s |

**Root causes:**
1. **Redundant reads:** Each of `num_heads` warps reads the entire KV cache independently
   - 64 heads → KV read 64 times
   - 128 heads → KV read 128 times
2. **Scalar loads:** Uses 2-byte loads instead of 128-bit vectorized reads
3. **Low parallelism:** Only `B * H` warps total (e.g., 64 warps for B=1, H=64)
4. **No MMA usage:** Scalar FP operations instead of tensor cores

### Expected Improvement Path

**Stage B (planned after online confirmation):** CTA-per-batch architecture
- One CTA handles all heads for one batch
- Load KV cache once per CTA into shared memory (64-128x read reduction)
- All warps in CTA process different heads in parallel
- Expected: 50-100x speedup (eliminates redundant reads)

**Stage C (planned):** Vectorized memory access
- 128-bit loads for ckv/kpe
- Coalesced global memory access
- Expected: 2-4x additional speedup

**Stage D (planned):** MMA optimization
- Use WMMA/MMA for QK and PV matmuls
- Expected: 2-3x additional speedup

**Stage E (planned):** Split-KV for small batches
- When B is small, split KV dimension across CTAs
- Enables higher occupancy
- Expected: 1.5-2x for small batch cases

### Hypothesis for Next Stage (DO NOT EXECUTE YET)

**After online baseline is confirmed correct:**

The dominant bottleneck is redundant KV cache reads. Moving to CTA-per-batch where one CTA processes all heads for one batch will eliminate 64x-128x redundant reads. This should improve:
- b1_l16384_h128 from 91ms → ~1-2ms (50-90x speedup)
- b16_l1024_h128 from 20.7ms → ~0.4-0.6ms (30-50x speedup)

The McFlashInfer reference at `@McFlashInfer/include/flashinfer/attention/mla_kernels_xcore1000.cuh` implements exactly this pattern:
- `BatchDecodeMLAKernel` launches one CTA per batch
- Uses `SharedStorageQKVO` for shared KV cache
- All warps collaborate on loading, then process different heads

## Next Steps

1. ✅ User submits baseline to online evaluation
2. ⏳ Wait for online results
3. ⏳ Record online baseline performance
4. ⏸️ Begin Stage B optimization (only after step 3)

---

**Note:** This file tracks the optimization journey. Each stage is immutable once promoted. Rejected candidates are preserved with their evidence.
