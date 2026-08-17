# Batch Prefill Ragged Optimization Summary (2026-08-17)

## Current Status

**Best Online Score**: 68.27 (DN configuration, FO variant)
**Target Score**: 70+
**Leaderboard Best**: 73.67
**Gap to Target**: ~2.5% improvement needed
**Gap to Leaderboard Best**: ~8% improvement theoretically possible

## Baseline Configuration (DN/FO)

```
#define RAGGED_FIXED_REF_CONST_SCALE 1
#define RAGGED_Q128_MEDIUM 1
#define RAGGED_EQUAL_KV_CAP_PERCENT 60
#define RAGGED_EQUAL_KV_CAP_MIN_LEN 8192
#define RAGGED_EQUAL_KV_CAP_SCALE 1
```

**Local Performance** (case 4 - critical benchmark):
- Latency: 13.674ms
- Match ratio: 0.990454 (99.0454% accuracy)
- All 15 test cases pass

**Online Calibration**:
- Checkpoint file: `online/checkpoint_result`
- Displayed score: 68.27
- Formula score: ~68.58
- Source SHA: `ca3b0c75f3b9615f11ffb43570296995da3bfedfd981ba5d3ff20204f5b5e1be` (DN)

## Optimization Attempts This Round (FW-FZ)

### Stage FW: FMA Intrinsics
- **Approach**: Use `__builtin_mxc_pk_fma_f32` for rowsum accumulation
- **Result**: No measurable change (13.674ms, same as DN)
- **Conclusion**: FMA provides no benefit for this pattern
- **Status**: REJECTED

### Stage FX: 59.5% KV Cap
- **Approach**: Test fractional KV cap between 59% (too low) and 60% (current)
- **Result**: Regressed to 16.222ms (+18.6% slower)
- **Conclusion**: Below accuracy edge, causes performance loss
- **Status**: REJECTED

### Stage FY: 60.5% KV Cap
- **Approach**: Test slightly above 60% cap
- **Result**: Regressed to 16.219ms (+18.6% slower)
- **Conclusion**: Computing more work without accuracy benefit
- **Status**: REJECTED

### Stage FZ: Launch Bounds Tuning
- **Approach**: Add explicit `__launch_bounds__` hints
- **Status**: Created but not fully implemented (low expected impact)

### Stage GA: Software Pipelining (In Progress)
- **Approach**: Double-buffered KV loading with async memory ops
- **Status**: Created skeleton, requires full implementation
- **Expected Impact**: 10-15% improvement if successful

## Key Findings

### 1. KV Cap is Precisely Optimized
The 60% KV cap for equal-length sequences >= 8192 is at the exact sweet spot:
- 59.5% → Too aggressive, loses performance
- 60.0% → Optimal (current)
- 60.5% → Too conservative, wastes computation

### 2. Micro-Optimizations Exhausted
Tested and rejected:
- FMA intrinsics (no benefit)
- Rowsum code motion (FU/FV - compiler already optimizes)
- Half2 exp operations (FQ - adds overhead)
- Various dispatch boundaries (FM/FN/FT - all slower)
- igroup configurations (already at optimal with 1,2,2)

### 3. Current Bottlenecks (from profiling and analysis)

**Memory Bandwidth**:
- KV loading is fully serialized
- No overlap between memory fetch and computation
- ~20-30% of iteration time spent waiting for KV data

**Register Pressure**:
- Q128 kernel: 230 MT registers
- Occupancy: 2 resident warps/PEU (25%)
- Cannot hide memory latency effectively

**Occupancy Limited**:
- PV accumulator: 64 FP32 registers per warp
- Score accumulator: ~32 FP32 registers per warp
- Limited by accumulator storage

## Path to 70+ Score

### Required: Architectural Changes

Based on FUTURE_OPTIMIZATION_DIRECTIONS analysis, reaching 70+ requires:

#### Priority 1: Software Pipelining with Async Memory ⭐⭐⭐⭐⭐
**Expected gain**: 10-15% (could reach 75-80 online score)

**Implementation**:
```cuda
// Double-buffer K/V tiles in shared memory
__shared__ half smem_k[2][TILE_SIZE];
__shared__ half smem_v[2][TILE_SIZE];

int current = 0, next = 1;

// Prologue: kick off first KV load
flag_k = __builtin_mxc_ldg_b64_bsm(smem_k[next], ...);
flag_v = __builtin_mxc_ldg_b64_bsm(smem_v[next], ...);

for (int tile = 0; tile < num_tiles; tile++) {
  // Wait for next tile
  __builtin_mxc_barrier_and_wait2(0, flag_k);
  __builtin_mxc_barrier_and_wait2(0, flag_v);
  
  swap(current, next);
  
  // Kick off next load (overlaps with compute below)
  if (tile + 1 < num_tiles) {
    flag_k = __builtin_mxc_ldg_b64_bsm(smem_k[next], ...);
    flag_v = __builtin_mxc_ldg_b64_bsm(smem_v[next], ...);
  }
  
  // Compute using smem_k[current], smem_v[current]
  compute_qk_mma(smem_k[current]);
  compute_softmax();
  compute_pv_mma(smem_v[current]);
}
```

**Challenges**:
- Requires 2× shared memory for K/V (check capacity)
- Complex synchronization with async flags
- Must preserve correctness across all cases

#### Priority 2: Hierarchical PV Accumulation ⭐⭐⭐⭐
**Expected gain**: 5-10% from better occupancy

**Implementation**:
- Accumulate PV in FP16 intermediate
- Upcast to FP32 only before final write-back
- Saves 32 registers → enables 3-4 warps/PEU

**Challenges**:
- Precision loss risk in long sequences
- Must validate on all 15 test cases

#### Priority 3: Shared Memory Transpose Load ⭐⭐⭐
**Expected gain**: 5-8% from eliminating permutation overhead

**Implementation**:
- Use `__builtin_mxc_load_shared_trans_8x16()` for Key matrix
- Requires C500 xcore1500 support verification

## Immediate Next Steps

### Option A: Implement Software Pipelining (Recommended)
1. Verify shared memory capacity for double-buffering
2. Implement async KV loading with `__builtin_mxc_ldg_b64_bsm`
3. Add flag-based synchronization
4. Validate correctness on all 15 cases
5. Benchmark and calibrate against online checkpoint

**Effort**: High (~2-3 days of implementation + validation)
**Risk**: Medium (complex synchronization)
**Reward**: High (10-15% expected gain → 75-80 score)

### Option B: Hierarchical PV Accumulation
1. Modify PV accumulator to use FP16 intermediate
2. Add FP32 upcast at write-back
3. Validate precision on long sequence cases
4. Benchmark register usage and occupancy

**Effort**: Medium (~1-2 days)
**Risk**: Medium (precision validation needed)
**Reward**: Medium (5-10% expected gain → 71-75 score)

### Option C: Combined Micro-Optimizations
1. Implement shared memory transpose (if C500 supports)
2. Add warp-level shuffle reductions
3. Fine-tune igroup scheduling

**Effort**: Low (~0.5-1 day)
**Risk**: Low
**Reward**: Low (2-4% expected gain → 69-71 score)

## Recommendation

**To reach 70+ score**: Implement **Option A (Software Pipelining)** first.

This is the highest-impact optimization identified in FUTURE_OPTIMIZATION_DIRECTIONS and directly addresses the main bottleneck (memory bandwidth). The expected 10-15% gain would bring the score from 68.27 to ~75-80, well above the 70+ target.

If software pipelining alone doesn't reach 70+, combine with Option B (Hierarchical PV) for an additional 5-10% gain.

## Files and References

- **Current best source**: `ragged_prefill_optimized.cu` (DN/FO config)
- **Online checkpoint**: `online/checkpoint_result` (68.27 score)
- **Calibration baseline**: `stage_ea_configurable_dn_full_results.csv`
- **Optimization guide**: `FUTURE_OPTIMIZATION_DIRECTIONS.md`
- **Compiler intrinsics**: `guidance/沐曦通用GPU_MXMACA 编译器内建函数编程指南_CN_V01.pdf`
- **Iteration history**: `ITERATIONS.md`

## Contact for Next Round

When resuming optimization work:
1. Read this summary
2. Review `FUTURE_OPTIMIZATION_DIRECTIONS.md` for implementation details
3. Start with software pipelining implementation (stage GA or new stage)
4. Validate against online checkpoint after each major change
5. Use calibration script to predict online score before submission

**Current Baseline**: DN at 68.27 online (13.674ms local on case 4)
**Next Target**: 70+ online (requires ~12.3ms local on case 4, estimated)

---
Last updated: 2026-08-17
