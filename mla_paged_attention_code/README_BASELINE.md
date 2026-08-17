# MLA Paged Attention Baseline - Ready for Submission

## ✅ Status: CORRECT - All Tests Pass

The baseline implementation in `mla_paged_baseline.cu` has been verified and is ready for online evaluation.

## Test Results Summary

**Correctness:** ✅ **22/22 cases passed** (100% pass rate)
- All cases: `match_ratio = 1.0`
- All cases: `severe_error_count = 0`
- Maximum error: `0.00390625` (well within tolerance)

**Performance Baseline:**
- Average time: 34.3ms per case
- Slowdown vs FlashInfer: 32-1046x (expected for naive baseline)
- Bandwidth: 0.2-3.7 GB/s (very low, as expected)

## Key Implementation Details

The baseline correctly implements:

1. **SM Scale:** `1/√(head_dim_ckv + head_dim_kpe) = 1/√576`
2. **MLA attention:** Q @ CKV^T → scores, then scores @ CKV → output
3. **Online softmax:** FP32 accumulation with numerically stable algorithm
4. **Page table indirection:** Correctly follows `kv_indices` for paged KV cache
5. **Sequence masking:** Respects `kv_lens[b]` for actual sequence length

## Architecture (Intentionally Simple)

```
Launch: (B * H) warps total
Each warp handles one (batch, head) pair:
  - Loads query from Q[b, h, :]
  - Iterates over all KV sequence positions
  - Computes attention weights with online softmax
  - Accumulates weighted sum into output
```

**Why it's slow:**
- Each of 64-128 warps reads the **same** KV cache independently
- Total redundant reads: 64-128x per batch
- Scalar memory access (2 bytes at a time)
- Very low occupancy

**This is intentional** - it's a correct baseline for optimization to beat.

## Files Generated

```
mla_paged_attention_code/
├── mla_paged_baseline.cu                    # ← Submit this file
├── benchmark_mla.py                         # Test harness
├── OPERATOR_CONTRACT.md                     # Complete specification
├── ITERATIONS.md                            # Optimization log
├── BASELINE_VERIFICATION.md                 # Detailed verification report
├── README_BASELINE.md                       # This file
├── explaination.md                          # Architecture analysis
├── candidates.jsonl                         # Iteration tracking
├── stage_a_baseline_results.csv             # Full test results
└── stage_a_baseline_results.meta.txt        # Environment metadata
```

## Next Steps

### 1. Submit to Online Evaluation

Submit `mla_paged_baseline.cu` to the online judge.

### 2. Record Online Results

After receiving the online evaluation result, paste it into:
```
mla_paged_attention_code/online/baseline_result.txt
```

### 3. Begin Optimization

Once the online baseline is confirmed correct, optimization will proceed with:

**Stage B - CTA-level parallelism:**
- Move from warp-per-head to CTA-per-batch
- Share KV cache reads across all heads in a batch
- Expected improvement: 50-100x speedup
- Reference: McFlashInfer's `BatchDecodeMLAKernel`

## Source SHA256

```
c66924dd9a7ecddfe74759602215f8cdc109d9f9ddb13836bc80954c34defbd6  mla_paged_baseline.cu
```

Use this hash to track which version was submitted online.

---

**⚠️ No optimization has been performed yet.**

The baseline is intentionally simple to ensure correctness first. Performance optimization will begin after online confirmation.
