# MLA Paged Attention Operator Contract

## Problem: XPU-OJ 20003 - FlashInfer MLA Paged Attention

## Kernel Entry ABI

```cpp
extern "C" void run_kernel(
    const __nv_bfloat16* q_nope,      // [batch_size, num_heads, head_dim_ckv]
    const __nv_bfloat16* q_pe,        // [batch_size, num_heads, head_dim_kpe]
    const __nv_bfloat16* ckv,         // [batch_size * seq_len, 1, head_dim_ckv]
    const __nv_bfloat16* kpe,         // [batch_size * seq_len, 1, head_dim_kpe]
    __nv_bfloat16* output,            // [batch_size, num_heads, head_dim_ckv]
    const int32_t* q_indptr,          // [batch_size + 1], decode: [0,1,...,batch_size]
    const int32_t* kv_indptr,         // [batch_size + 1]
    const int32_t* kv_indices,        // [batch_size * seq_len]
    const int32_t* kv_lens,           // [batch_size]
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim_ckv,             // Fixed: 512
    int64_t head_dim_kpe,             // Fixed: 64
    int64_t page_size,                // Fixed: 1
    int64_t causal                    // Fixed: 0
);
```

## Input/Output Layouts

### Fixed Parameters (from evaluation)
- `head_dim_ckv = 512`
- `head_dim_kpe = 64`
- `page_size = 1` (one page = one KV token, no intra-page offset)
- `causal = 0` (decode has single query, causal is no-op)
- `num_heads ∈ {64, 128}`
- All tensors are contiguous BF16 or INT32 on device

### MLA Structure
```
K_j = concat(ckv_j, kpe_j)    # 576 = 512 + 64
V_j = ckv_j                    # 512, reuses compressed KV
```

**Key insight:** All heads share the SAME ckv/kpe. This is extreme GQA with group_size = num_heads.

## Algorithm

For request `b`, head `h`:

```python
q_row = q_indptr[b]                    # decode: q_row = b
page_begin = kv_indptr[b]
page_end = kv_indptr[b + 1]
kv_len = min(kv_lens[b], page_end - page_begin)

sm_scale = 1.0 / sqrt(head_dim_ckv + head_dim_kpe)  # 1/sqrt(576), NOT 1/sqrt(512)

for j in range(kv_len):
    token = kv_indices[page_begin + j]
    score_j = (dot(q_nope[b,h,:], ckv[token,:]) + 
               dot(q_pe[b,h,:], kpe[token,:])) * sm_scale
    
# Online softmax over scores
p = softmax(scores)

# Output (V = ckv)
output[b,h,:] = sum_j p_j * ckv[token,:]
```

## Correctness Requirements

1. **sm_scale uses 576**: `1.0 / sqrt(512 + 64)`, NOT `1.0 / sqrt(512)`
2. **V equals ckv**: No separate V tensor, reuse ckv for attention output
3. **Respect kv_lens[b]**: Clamp to `min(kv_lens[b], kv_indptr[b+1] - kv_indptr[b])`
4. **Page table indirection**: `token = kv_indices[page_begin + j]`, don't skip even with identity pages
5. **Numerical stability**: FP32 online softmax with proper m/d state tracking

## Error Tolerance

- `ATOL = 1.6e-2`
- `RTOL = 1.6e-2`
- `tolerance = ATOL + RTOL * |reference|`
- **Pass criteria:** `match_ratio >= 0.99` AND `severe_error_count == 0` (severe = error > 8x tolerance)

## Test Cases

22 cases covering:
- `batch_size ∈ {1, 3, 4, 5, 16, 64}`
- `seq_len ∈ {257, 1023, 1024, 4096, 8192, 16384}`
- `num_heads ∈ {64, 128}`
- Non-power-of-2 lengths (cases 21, 22) to check boundary handling

## Submitted Artifact

CUDA source file compiled with:
```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  <source>.cu -o <library>.so
```

## Commands

**Correctness check:**
```bash
python mla_paged_attention_code/benchmark_mla.py \
  --source <source>.cu --library <library>.so \
  --cases all --max-repeats 3
```

**Fast regression:**
```bash
python mla_paged_attention_code/benchmark_mla.py \
  --cases 1,13,21,22 --max-repeats 3
```

**Random page table test:**
```bash
python mla_paged_attention_code/benchmark_mla.py \
  --cases 1,13 --permute-pages
```

## Baseline

- Source: `mla_paged_baseline.cu`
- Status: **Correct** (all 22 cases pass, match_ratio=1.0)
- Performance: 90-1000x slower than FlashInfer reference
- Bottleneck: Reads KV cache `num_heads` times (once per warp)
- Bandwidth: 0.2-3.6 GB/s (should be >>100 GB/s on C500)

## Reference Implementation

McFlashInfer located at `@McFlashInfer/`:
- Main kernel: `include/flashinfer/attention/mla_kernels_xcore1000.cuh`
- Params: `include/flashinfer/attention/mla_params.cuh`
- Uses CTA-per-batch with shared KV cache across all heads
- Vectorized 128-bit loads, MMA for QK/PV, split-KV for small batches

## Persistent User Constraints

- Optimize `mla_paged_optimized.cu` toward a conservatively projected online score above 75, using the raw `online/` reports as anchors.
- Never use the reference file's pointer replay or zero-QPE/KPE assumption: public contract requires arbitrary BF16 QPE/KPE and every invocation must compute from its inputs.

## Optimization Direction

The promoted path uses CTA-level KV reuse, 128-bit transactions and BF16 MMA
for full CKV+KPE attention, plus split-KV. Continue only with candidates that
pass the ABI-level full gate and improve the per-case online calibration.

After baseline is confirmed correct on online evaluation, optimization stages:
1. CTA per batch (all heads share KV) - 64-128x speedup expected
2. 128-bit vectorized memory access
3. MMA for QK and PV computation
4. Split-KV for small batch parallelism

## Current Phase

✅ Establish contract
✅ Baseline passes all local tests
⏳ Submit baseline to online evaluation (user will do this)
⏸️ Wait for online results
⏸️ Begin optimization after online baseline confirmed

## Current implementation note

`mla_paged_optimized.cu` is the canonical full-KPE/CKV C500 MMA implementation. It
uses the tuned persistent split-KV schedule and exact same-stream merge repair;
the official 24-case online proxy passes 24/24 and permuted-page regression passes
4/4. The conservative local-to-online projection is approximately 72.44, below
the requested 75 threshold; no score claim above 75 is made. Uniform decode
metadata is the currently measured fast-path contract, while the scalar fallback
remains metadata-aware for nonuniform/odd public shapes.
