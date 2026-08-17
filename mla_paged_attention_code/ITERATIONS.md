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

---

## Stage C: Full KPE MMA path with guarded exact merge (current promoted candidate)

**Source:** `mla_paged_optimized.cu`  
**SHA256:** `0923b903eb1e3716ecf2f55cfb6f69651a16a8c24bd2ea87739da358cecf7a5f`

- Replaced the old scalar per-head implementation with the installed official
  xcore1000 MMA attention path, including both `q_nope·ckv` and `q_pe·kpe`.
- Explicitly excluded the supplied reference experiment's pointer replay and
  zero-QPE/KPE assumption. Every invocation derives its result from current
  device inputs.
- Published-online local suite: 24/24 pass on standard-normal BF16 inputs;
  representative max error is at most `0.001953125`.
- Permuted page-table regression also passed on representative short and long
  cases, confirming that `kv_indices` remains authoritative.
- A planner undercoverage at B=4/L=1024/H=128 is repaired by a same-stream
  exact partial merge; the other fast-path shapes keep the faster persistent
  merge. The non-32-aligned and B=64 local-only stress cases take the trusted
  exact scalar fallback.

### Measured negative branches

- A standalone merge for all shapes restored correctness but added 25--40% to
  short-B latency, so it was rejected.
- CTQ32 and 104-cluster B16/H128-long schedule variants produced allocator or
  merge-metadata failures; both were reverted and recorded in `candidates.jsonl`.

### Calibration status

`online/stage_c_projection.csv` is a 24-case projection using the online
reference report and local FlashInfer per-case timing ratios. It predicts mean
display score **71.87**, with no correction history; this is below the 75 goal
and is not an actual online submission result. The original 74.87 report is
not a valid correctness anchor for arbitrary inputs because its source uses
zero-QPE/KPE and pointer-replay probes.

### Stage AQ / final revalidation

The canonical source was rebuilt from the complete C500 xcore1000 MLA kernel
with the tuned persistent split-KV planner. Full online-order local coverage
passed 24/24; an additional permuted-page run passed 4/4 for cases 21--24.
The exact source/library export was checked with `nm -D` and only `run_kernel`
is exported. Projection from `online/checkpoint_result_reference` and the
FlashInfer local anchor is 72.44 mean display score (24 cases), so this is a
correctness/reproducibility revalidation, not a promotion over the 75 target.

Rejected after measurement: xcore1500-only LDS transpose, CTQ32 two-stage
shared-memory pipeline (invalid launch on xcore1000), CTQ32 dense B16/H64
long scheduling (planner allocator fault), device metadata wrapper (large
overhead), and composite shape schedules (72.68 projection). The canonical
source remains the best fully measured legal C500 implementation.

### Resume stages AR–AV (2026-08-17)

- **AR, B16/H128 long CTQ32:** exact and correct, but 0.903/1.699 ms on
  cases 23/24 versus the canonical ~0.466/0.870 ms. Rejected: the smaller
  CTA loses too much tensor-core throughput.
- **AS, unsharded QK:** rejected at compile time: the xcore1000 64-bit MLA
  helpers statically require `QK_SHARD=true`.
- **AT, B16/H64 six chunks:** exact but slower (0.277/0.491 ms on cases
  11/12), confirming that the promoted seven-chunk scheduling is needed for
  load balance.
- **AU, aligned CTQ64 control-flow specialization:** exact, but 20% slower
  on long cases because the hand-written sequence lost an upstream overlap.
  Rejected.
- **AV, exact no-op softmax rescale elision:** if an online-softmax tile does
  not update its running maximum, multiplying the prior denominator/output by
  one is skipped. Full 24/24 ABI suite and four permuted-page long cases pass.
  Its 4.366 ms local total is 0.7% below AQ's 4.398 ms; the one-anchor online
  projection is **72.61**, only +0.17 and therefore still below the 75 target
  and inside normal cross-run uncertainty. It remains `investigate`, not a
  promotion.

The current measured C500 CTQ64 path is compute-bound (256 MT registers,
108 ST registers, static max two warps/PEU; prior case-24 profiler reports
about 90% compute-instruction busy duty). Further legal gains require a
verified full-CKV+KPE kernel-pipeline redesign; the zero-QPE/KPE and
pointer-replay code in `mla_paged_reference.cu` is explicitly excluded.
