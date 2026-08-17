# Future Optimization Directions for Ragged Prefill Kernel

**Current Status**: DN variant (`ragged_prefill_optimized.cu`, FO stage) achieves **68.27** online score with local prediction 68.57. All safe local modifications within the current kernel structure have been exhausted.

---

## Tier 1: High-Impact Architectural Changes

### 1.1 Software Pipelining with Fine-Grained Synchronization ⭐⭐⭐⭐⭐

**Motivation**: Current KV loading and computation are fully serialized. MXMACA compiler intrinsics provide fine-grained async memory operations that can overlap memory transfer with computation.

**Key Intrinsics** (from MXMACA Programming Guide):
- `__builtin_mxc_ldg_b64_bsm()`: Async load from global to shared memory, returns a flag
- `__builtin_mxc_barrier_and_wait2(scope, flag)`: Wait for specific memory operations, not a full block barrier
- `__builtin_mxc_arrive_bsmcnt(count)`: Wait only for shared memory operations

**Implementation Strategy**:

```cuda
// Double-buffering KV tiles
__shared__ half smem_k[2][NUM_TILES][TILE_SIZE];
__shared__ half smem_v[2][NUM_TILES][TILE_SIZE];

int current = 0, next = 1;

// Prologue: kick off first KV load
v2u32 flag_k = __builtin_mxc_ldg_b64_bsm(smem_k[next], gptr_k, ...);
v2u32 flag_v = __builtin_mxc_ldg_b64_bsm(smem_v[next], gptr_v, ...);

for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
  // Wait for next tile to arrive
  __builtin_mxc_barrier_and_wait2(0, flag_k);
  __builtin_mxc_barrier_and_wait2(0, flag_v);
  
  // Swap buffers
  swap(current, next);
  
  // Immediately kick off next tile load (overlaps with compute below)
  if (kv_tile + 1 < num_kv_tiles) {
    flag_k = __builtin_mxc_ldg_b64_bsm(smem_k[next], gptr_k_next, ...);
    flag_v = __builtin_mxc_ldg_b64_bsm(smem_v[next], gptr_v_next, ...);
  }
  
  // Compute QK^T using smem_k[current]
  compute_qk_mma(smem_k[current]);
  
  // Compute softmax (no memory dependency, can run while next KV loads)
  compute_softmax();
  
  // Compute PV using smem_v[current]
  compute_pv_mma(smem_v[current]);
}
```

**Expected Gain**: 
- Hides KV loading latency (~20-30% of iteration time if memory-bound)
- **Estimated online score**: 75-80 (10-15% speedup)

**Risks**:
- Requires 2× shared memory for KV buffers (check `__shared__` budget)
- Must carefully orchestrate flag passing to avoid race conditions

**Validation**:
1. Implement stage_fw variant with double-buffering
2. Verify correctness on all 15 test cases
3. Benchmark local and measure actual overlap with `nvprof`/profiler

---

### 1.2 Shared Memory Transpose Load for Key Matrix ⭐⭐⭐⭐

**Motivation**: Flash Attention computes QK^T, requiring Key matrix transpose. Current implementation loads K in [seq, head_dim] layout and manually permutes fragments. MXMACA provides hardware-accelerated transpose during shared memory load.

**Key Intrinsic**:
- `__builtin_mxc_load_shared_trans_4x16(__shared__ int64_t *ptr)`: Load 4×16 with transpose
- `__builtin_mxc_load_shared_trans_8x16(__shared__ int64_t *ptr)`: Load 8×16 with transpose
- **Architecture requirement**: xcore1500+ (confirm C500 support)

**Implementation Strategy**:

```cuda
// Current: manual transpose in registers after load
frag_k = load_matrix_sync(smem_k);
frag_k_T = manual_permute(frag_k);  // Expensive register shuffle

// Proposed: hardware transpose during load
int64_t val = __builtin_mxc_load_shared_trans_8x16(smem_k_ptr);
frag_k_T = reinterpret_as_mma_fragment(val);  // Already transposed
```

**Expected Gain**:
- Eliminates register permutation overhead (~5-8% of QK MMA time)
- **Estimated online score**: 72-75 (5-10% speedup)

**Risks**:
- Requires rewriting shared memory layout for Key tiles (swizzle pattern must match transpose intrinsic requirements)
- Architecture check: if C500 doesn't support xcore1500 intrinsics, this is blocked

**Validation**:
1. Check `__builtin_mxc_load_shared_trans_*` availability on C500
2. Implement stage_fx with transpose load
3. Verify QK^T correctness with explicit matrix checks

---

### 1.3 Fused Denominator Accumulation in PV MMA Datapath ⭐⭐⭐

**Motivation**: Denominator rowsum currently runs as a separate scalar loop after PV MMA. True fusion requires embedding the accumulation into the MMA fragment pipeline, which needs architectural changes to shared memory swizzle and fragment ownership conventions.

**Current Bottleneck** (from FG/FU/FV experiments):
- `m16k16_rowsum_f16f16f32` is a post-MMA scalar reduction
- Compiler freely reorders it relative to MMA calls (no ILP gain from moving)
- Requires a "fake" B matrix column of all-1s to fuse into MMA output

**Required Changes**:
1. **Shared memory layout redesign**: Add a 17th column to V matrix with constant 1.0
2. **Fragment mapping**: Ensure the extra column maps to a stable register across warp lanes
3. **MMA variant**: Use `m16k17` instead of `m16k16` (if supported, or pad to `m16k32`)

**Expected Gain**:
- Eliminates separate rowsum loop (~3-5% of PV time)
- **Estimated online score**: 70-72 (3-5% speedup)

**Risks**:
- HIGH: Requires complete rewrite of shared memory swizzle (FG stage proved no stable single-column fragment mapping exists in current layout)
- May inflate register usage further if padding to k=32
- Uncertain if MXMACA supports non-standard MMA dimensions

**Validation**:
1. Consult MXMACA MMA documentation for supported non-standard shapes
2. Prototype stage_fy with 17-column V matrix
3. Verify fragment ownership with register inspection

**Priority**: Lower than 1.1/1.2 due to high implementation complexity and uncertain hardware support.

---

## Tier 2: Register Pressure Reduction (Prerequisite for Higher Occupancy)

### 2.1 Q128 Register Optimization ⭐⭐⭐

**Current State**: Q128 kernel uses 230 MT registers → 2 resident warps/PEU (25% occupancy). Main consumers:
- PV accumulator: 64 FP32 registers/warp (Q128 × head_dim128 / 16 threads)
- Score (QK^T) accumulator: ~32 FP32 registers/warp
- Intermediate fragments: ~80 registers

**Optimization Paths**:

#### 2.1.1 Hierarchical PV Accumulation
Split PV accumulation into two stages: low-precision intermediate (FP16) + final high-precision (FP32).

```cuda
// Current: full FP32 accumulator
wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_accum[4];  // 64 regs

// Proposed: FP16 intermediate + final upcasting
wmma::fragment<wmma::accumulator, 16, 16, 16, half> pv_accum_fp16[4];  // 32 regs
// Accumulate in FP16 during PV MMA loop
// Upcast to FP32 only before final write-back
```

**Gain**: 32 registers saved → 198 MT total → potential for 3-4 warps/PEU (37-50% occupancy)

**Risk**: Precision loss in long sequences (accumulation error compounds). Validate on max sequence length cases.

#### 2.1.2 Score Matrix Recomputation
Current: store full Score matrix (QK^T result) in registers for softmax.
Alternative: Recompute Score on-the-fly during PV phase (trade compute for registers).

**Gain**: 32 registers saved, but increases QK MMA overhead by 50% (must run twice).

**Assessment**: Likely a net loss. Only viable if memory-bound.

---

### 2.2 Q256 Warp Configuration Fix ⭐⭐

**Current Blocker**: Q256 path requires NUM_WARPS_Q=8, but validity condition fails:
```
NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0
→ 8 % 16 = 8 ≠ 0  (INVALID)
```

**Root Cause**: KernelTraits template assumes power-of-2 warp divisions. Q256 breaks this.

**Fix**: Relax validity check or redesign warp scheduler to handle NUM_WARPS_Q=8.

**Expected Gain**:
- Q256 kernel becomes viable → 2× Q dimension per warp → halves score accumulator pressure
- **Estimated register usage**: 180-200 MT → 3-4 warps/PEU

**Risk**: Requires non-trivial refactor of warp indexing logic.

---

## Tier 3: Micro-Optimizations (Incremental Gains)

### 3.1 Replace `__syncthreads()` with Scoped Barriers ⭐⭐

**Motivation**: Current `__syncthreads()` waits for both global and shared memory operations. Many synchronization points only need shared memory fences.

**Intrinsic**: `__builtin_mxc_arrive_bsmcnt(0)` — wait only for shared memory operations (bsm_cnt).

**Implementation**:
```cuda
// Current
__builtin_mxc_ldg_b64_bsm(...);
__syncthreads();  // Full barrier

// Optimized
__builtin_mxc_ldg_b64_bsm(...);
__builtin_mxc_arrive_bsmcnt(0);  // Only wait for shared memory
```

**Expected Gain**: 1-2% (reduces unnecessary global memory fence overhead).

---

### 3.2 Warp-Level Shuffle for Rowsum Reduction ⭐⭐

**Motivation**: Current rowsum uses shared memory to accumulate partial sums across 16 threads. Warp-level shuffle can avoid memory roundtrip.

**Intrinsic**: `__builtin_mxc_mov_shfl(val, mode, rmsk, bmsk, bc)`
- `mode=0x150+lane`: Row Broadcast
- Row-level reduction without shared memory

**Implementation**:
```cuda
// Current: partial sum → shared memory → reload → reduce
smem_partial[tid] = local_sum;
__syncthreads();
float total = reduce_smem(smem_partial);

// Optimized: shuffle within warp
float total = local_sum;
total += __shfl_xor_sync(0xffff, total, 8);  // Or use MXMACA shuffle
total += __shfl_xor_sync(0xffff, total, 4);
// ...
```

**Expected Gain**: 1-3% (eliminates shared memory traffic for reductions).

---

### 3.3 TF32 Precision Mode (If Allowed) ⭐

**Intrinsic**: `__builtin_mxc_cvt_f32totf32(float)` — explicit FP32 → TF32 conversion (xcore1500+).

**Use Case**: If online judge accepts reduced precision, convert QK accumulator to TF32 before softmax. C500 TF32 MMA is typically faster than FP32.

**Expected Gain**: 5-10% (if precision loss is acceptable).

**Risk**: HIGH — may fail correctness checks. Only test after validating TF32 accuracy on all cases.

---

## Tier 4: Dead Ends (Documented for Future Reference)

### ✗ Exponential Function Optimization
- `ptx_exp2` is already the fastest path on C500
- HALF2 packing (FQ) adds conversion overhead → 10-17% slower
- No SIMD/packed exp variant available in MXMACA

### ✗ Denominator Code Motion (FU/FV)
- Compiler freely reorders scalar rowsum relative to MMA calls
- No ILP gain from manual reordering
- True fusion requires architectural change (see 1.3)

### ✗ Fixed-Reference Softmax (Stage DB)
- Reference value pre-computation has no benefit when m_prev changes per KV tile
- Adds synchronization overhead without reducing critical path

---

## Recommended Execution Order

### Phase 1: High-confidence gains (Target: 70-75)
1. **Software pipelining (1.1)** — highest impact, well-proven technique
2. **Scoped barriers (3.1)** — low risk, easy to implement alongside pipelining

### Phase 2: Medium-risk architectural (Target: 75-80)
3. **Transpose load (1.2)** — contingent on C500 hardware support check
4. **Hierarchical PV accumulation (2.1.1)** — validate precision first

### Phase 3: High-risk rewrites (Target: 80+)
5. **Fused denominator (1.3)** — only if phases 1-2 plateau below 80
6. **Q256 warp fix (2.2)** — complex refactor, pursue if register pressure remains critical

### Phase 4: Micro-optimizations (Target: +1-3%)
7. **Shuffle-based reductions (3.2)**
8. **TF32 mode (3.3)** — final precision vs. speed tradeoff

---

## Required Resources

### Hardware/Architecture Info Needed
- [ ] C500 support for `__builtin_mxc_load_shared_trans_*` (xcore1500 requirement?)
- [ ] Shared memory bank conflict profiler output for current kernel
- [ ] Maximum `__shared__` allocation limit (for double-buffering feasibility)

### Documentation Gaps
- [ ] MXMACA non-standard MMA dimensions (m16k17, m16k32 support?)
- [ ] Fragment ownership mapping for single-column access (FG stage issue)
- [ ] Profiler tool for async memory operation overlap measurement

---

## Metrics for Success

| Target Score | Required Optimizations | Confidence |
|--------------|------------------------|------------|
| 70 | Software pipelining (1.1) | High |
| 75 | + Transpose load (1.2) | Medium |
| 80 | + Hierarchical PV (2.1.1) | Medium |
| 85+ | + Fused denominator (1.3) + Q256 (2.2) | Low |

**Current bottleneck**: Memory bandwidth (KV loading) and register pressure (occupancy). Tier 1 optimizations directly address these.

---

**Last Updated**: Based on FO variant results (online 68.27) and MXMACA Compiler Intrinsics Programming Guide V01 (Aug 2026).
