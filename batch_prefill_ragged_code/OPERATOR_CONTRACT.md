# Ragged prefill operator contract

## Objective

Optimize XPU-OJ problem 20001 (`BatchPrefillWithRaggedKVCache`) while preserving exact
BF16 attention semantics. The requested stopping condition is a local, calibrated score
projection of at least 73.67; a projection is not an online result.

## ABI and public inputs

The submitted artifact is one self-contained CUDA/MACA source file exporting the unmangled
symbol `run_kernel`:

```cpp
extern "C" void run_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* output,
    const int32_t* qo_indptr, const int32_t* kv_indptr,
    int64_t batch_size, int64_t seq_len, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t head_dim_qk, int64_t head_dim_vo,
    int64_t causal);
```

Inputs/outputs are contiguous NHD BF16 tensors. The public tests use
`num_qo_heads=32`, `num_kv_heads=4`, `head_dim_qk=head_dim_vo=128`, and `causal=1`.
The true per-request lengths and total buffers are defined by the device `qo_indptr` and
`kv_indptr`; `seq_len` is only an upper bound. Causal masking is bottom-right aligned for
`q_len != kv_len`. GQA maps eight consecutive query heads to one KV head.

Correctness tolerance is `rtol=atol=1.6e-2`; ordinary cases require at least 99% elements
within tolerance and no element above 8x tolerance. Cases 14 and 15 require 100%.
Skipped computation, case identification, approximate output, explicit synchronization in
the hot path, and assumptions about hidden indptr values are forbidden.

## Source, build, and measurement

- Immutable correctness anchor: `ragged_prefill_baseline.cu`.
- Current promoted source at resume: `ragged_prefill_optimized.cu` (historical CQ,
  SHA-256 `3cee75cf0c376c81c235a0df25c9208220f36f28bc6de0d664091064e92192dc`).
- Historical candidates: `ragged_prefill_stage_*.cu`; new candidates use immutable
  `ragged_prefill_stage_<id>.cu` names and are never overwritten.
- Harness: `benchmark_stage_a.py`, which dynamically loads `run_kernel`, uses GPU events,
  checks against FlashInfer, and reports all 15 published proxy cases. Fast signal uses a
  representative subset; promotion uses all 15 plus an alternate seed and export check.
- Build: `mxcc -O3 -std=c++17 --offload-arch=xcore1000 -I/opt/maca/tools/cu-bridge/include
  -shared -fPIC <source> -o <library>`.
- Local environment at resume: MetaX C500, MACA/PyTorch/FlashInfer 3.7.1, `mxcc`
  1.0.0 (d9102a1572), xcore1000.

## Online calibration

`online/checkpoint_result` is preserved verbatim and currently reports the 67.60-ish
aggregate checkpoint, but its submitted source SHA is not recorded in the report. The
last documented 70.27 report for historical CL is not present as raw text in this checkout;
therefore any new projection must label its anchor and uncertainty. Use per-case ratios and
`calibrate_online.py` (or the repository projection helper) rather than aggregate latency.
