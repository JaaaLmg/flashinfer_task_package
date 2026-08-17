# Ragged prefill operator contract

## Objective

Optimize XPU-OJ problem 20001 (`BatchPrefillWithRaggedKVCache`) while preserving exact
BF16 attention semantics. The current user-requested stopping condition is an actual online
score strictly above 70.0; projections only choose what to submit.

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
- Current promoted source: `ragged_prefill_optimized.cu` (DN restored after DV, SHA-256
  `ca3b0c75f3b9615f11ffb43570296995da3bfedfd981ba5d3ff20204f5b5e1be`).
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

`online/checkpoint_result` is preserved verbatim and is the actual DN checkpoint supplied by
the user (online leaderboard reported 68.27; the visible per-case score-ratio mean is 68.7179).
Historical CL's 70.266667 record in Git commit `d2f1610` belongs to an obsolete online baseline:
the restored CL behavior scored only 65.93 on the current platform, so it is invalid as a
calibration or promotion anchor. Every future candidate must use this checkpoint's per-case online
times and a same-harness DN local anchor; after any submission, replace/add the raw online report
before selecting the next direction.

The latest same-source local reconfirmation is `stage_ea_configurable_dn_full_results.csv`
(15/15 pass, 27.350 ms total).  Projecting per case from `stage_ds_dn_reconfirm_full_results.csv`
and the raw checkpoint gives 68.579 formula points in
`online/stage_ea_configurable_dn_projection.csv`.  The raw report's visible mean exceeds the
user-supplied displayed aggregate by 0.448 points, so the conservatively display-calibrated
estimate is about 68.13; it is not an online result.

## Current user constraints (2026-08-17)

- Treat `online/checkpoint_result` and the user's reported **68.27** as the sole current online
  calibration anchor; do not use obsolete online submissions or historical score records as a
  numerical mapping.
- Do not spend new iterations on micro-parameter tuning. New candidates must be architectural
  rewrites/refactors motivated by Tier 1 in `FUTURE_OPTIMIZATION_DIRECTIONS.md`.
- The leaderboard reference is 73.67; local projections remain estimates and never replace an
  actual online result.
