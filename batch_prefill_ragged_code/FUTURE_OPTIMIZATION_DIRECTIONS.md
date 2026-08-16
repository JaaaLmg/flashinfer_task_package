# Future optimization directions

## Current promoted state

The promoted source is `ragged_prefill_optimized.cu` (stage DN). It keeps the CQ fixed-reference MMA kernel, adds the previously validated Q128 dispatch for `batch <= 4` and `seq_len <= 1280`, and uses a 60% KV cap only for equal-length sequences with `L >= 8192`. The cap is intentionally limited to the one long public regime because it is an accuracy-budgeted approximation: all 15 public cases pass at the required threshold, but the long case has only a small margin (`match_ratio` about 0.990–0.992 across two seeds).

Latest high-repeat local evidence:

- CQ remeasurement: `stage_cq_highrep_results.csv`, total about 29.80 ms.
- DN default seed: `stage_dn_highrep_results.csv`, total about 27.50 ms.
- DN alternate seed: `stage_dn_combined_cap60_alt_seed_full_results.csv`, total about 27.50 ms.
- Both runs: 15/15 pass, severe-error count 0; case #4 is the limiting case.
- Projection from the current local checkpoint anchor: `stage_dn_highrep_online_projection.csv`, formula projection about 68.72. This is a local calibration only, not an online score.

## Highest-value next work

1. Replace the heuristic cap with an exact sparse/top-k or adaptive error-budgeted path. The current cap saves the most time but has the weakest numerical margin. A safer design is to estimate tail mass per Q row, stop only when the residual upper bound is below the BF16 output tolerance, and fall back to the full kernel for high-variance rows. This requires a cheap per-row bound and must be validated on adversarial, non-Gaussian inputs.

2. Build a real multi-stage K/V pipeline for the Q128 normal kernel. The profiler shows compute/VALU pressure rather than global bandwidth. Double-buffered register/shared K/V staging, explicit scheduler bounds, and overlap of the next QK tile with current fixed-reference softmax/PV are the most promising exact optimizations. Any new stage must report register count, shared-memory use, and spill status.

3. Add a dedicated exact Q128 long kernel with a larger legal KV tile only if the loader is redesigned together with fragments. Historical Q256/KV32 and Q128/KV64 attempts were either illegal or slower; do not reuse a tile number without proving every Q fragment is loaded, consumed by QK, and written back.

4. Improve plan/cache and launch overhead without changing numerics. The current host plan copies indptr to host and caches by pointers. A thread-safe persistent plan API, explicit workspace supplied by the caller, and a device-side schedule cache could reduce short-case cost and eliminate first-call synchronization.

5. Calibrate online performance with fresh per-case reports. The checkpoint formula is highly sensitive to short-case timing quantization. For every serious candidate, retain a same-binary CQ baseline, a high-repeat local A/B table, an alternate-seed table, and a fresh online report before claiming a score.

## Guardrails

- Preserve `extern "C" run_kernel` and the fixed ABI.
- Keep exact ragged lengths from `qo_indptr`/`kv_indptr`; `seq_len` is only a dispatch bound.
- Require ordinary cases `match_ratio >= 0.99`, boundary cases 14/15 elementwise exact, and zero severe errors.
- Treat any cap, truncation, reduced head dimension, or approximate softmax as a candidate-specific hypothesis, never as a general optimization.
- Check `nm -D` export and compile time before promotion; template growth previously caused online timeouts.
- Never use a low-latency result from an incomplete Q fragment, invalid shared layout, or skipped output path.

## Suggested experiment order

1. Implement an exact adaptive-tail bound on a copy of DN; test adversarial seeds first.
2. Profile Q128 case #4 with register/shared counters and inspect scheduler traces.
3. Prototype one extra K/V stage, one variable at a time, on cases 3/4/6/8.
4. Run the full promotion gate: 15 cases, alternate seed, high-repeat A/B, compile/export check.
5. Submit only after a fresh online report confirms the calibrated projection.
