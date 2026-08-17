# Pre-Submission Checklist for Optimized V1

## File to Submit
**File:** `mla_paged_optimized.cu`  
**SHA256:** `3cdca1031d1298694e7479542efb7d00b9361d3f863a45ba6d6de36e4968b524`

## Verification Status

### ✅ Local Correctness
- [x] All 22 test cases pass
- [x] 100% match ratio (1.0 for every case)
- [x] Maximum error: 0.00390625 (within tolerance)
- [x] No severe errors
- [x] Compiles without errors on MetaX C500

### ✅ Implementation Correctness
- [x] SM scale formula: `1/√576` ✓
- [x] Attention computation: Q @ CKV^T ✓
- [x] Softmax: Online algorithm with FP32 accumulation ✓
- [x] Output: Weighted sum with proper normalization ✓
- [x] Page table: Correct indirection via `kv_indices` ✓

### ✅ Interface Contract
- [x] Entry point: `extern "C" void run_kernel(...)` ✓
- [x] All parameters match specification ✓
- [x] Returns void ✓
- [x] Uses bf16 for tensors ✓

### ✅ Resource Constraints
- [x] Shared memory: 0 bytes (under 65536 limit) ✓
- [x] No compilation warnings (except TORCH_CUDA_ARCH_LIST) ✓
- [x] Compiles with: `mxcc -O3 -std=c++17 --offload-arch=xcore1000` ✓

## Before Submitting

1. **Verify file integrity:**
   ```bash
   sha256sum mla_paged_optimized.cu
   # Should output: 3cdca1031d1298694e7479542efb7d00b9361d3f863a45ba6d6de36e4968b524
   ```

2. **Final local test:**
   ```bash
   python benchmark_mla.py --source mla_paged_optimized.cu
   # Should show: 22/22 passed, all match=1.0
   ```

3. **Prepare for result:**
   - Have `online/optimized_v1_result.txt` ready to paste result

## After Submission

1. **Save complete output** to `online/optimized_v1_result.txt`

2. **Notify the optimizer** so it can:
   - Verify online pass/fail status
   - Calibrate local benchmarks to online scoring
   - Begin Stage B optimization with real performance data

## Expected Outcome

This V1 should:
- ✅ **PASS** all correctness checks
- 📊 Establish baseline performance (likely similar to naive baseline)
- 🎯 Provide calibration anchor for future optimizations

**Performance Note:** This version is intentionally NOT optimized yet. It's ~1.0-1.1x faster than naive baseline. Real optimization begins in Stage B after online calibration.

---

**READY TO SUBMIT** ✅  
Date: 2026-08-17
