# Baseline Submission Checklist

## Pre-Submission Verification

- [x] All 22 local test cases pass
- [x] Correctness: match_ratio = 1.0 for all cases
- [x] No severe errors (severe_error_count = 0)
- [x] Maximum error within tolerance (< 0.004)
- [x] Implementation matches MLA specification
- [x] Page table indirection works correctly
- [x] SM scale formula verified: 1/sqrt(576)
- [x] Online softmax with FP32 accumulation
- [x] Source SHA256 recorded

## File to Submit

**Submit this file:** `mla_paged_baseline.cu`

**SHA256:** `c66924dd9a7ecddfe74759602215f8cdc109d9f9ddb13836bc80954c34defbd6`

## After Online Submission

### Step 1: Save Online Result
Copy the complete online evaluation result and paste it into:
```
online/baseline_result.txt
```

Create the directory if needed:
```bash
mkdir -p online
# Paste result into online/baseline_result.txt
```

### Step 2: Notify for Optimization
Once `online/baseline_result.txt` exists, optimization can begin.

### Step 3: Expected Online Behavior
- **Should pass:** All correctness tests
- **Expected score:** Very low (this is a naive baseline)
- **Purpose:** Establish correctness anchor for optimization

## What NOT to Do Yet

- ❌ Do not optimize the code yet
- ❌ Do not change the kernel launch configuration
- ❌ Do not add shared memory usage
- ❌ Do not change the algorithm

**Optimization begins AFTER online baseline confirmation.**

## Questions?

Check these files for details:
- `OPERATOR_CONTRACT.md` - Task specification
- `BASELINE_VERIFICATION.md` - Detailed correctness verification
- `README_BASELINE.md` - High-level summary
- `ITERATIONS.md` - Optimization roadmap (for later)
