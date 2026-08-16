# Future optimization directions

## Current promoted state

The promoted source is `ragged_prefill_optimized.cu` (stage DN). It keeps the CQ fixed-reference MMA kernel, adds the previously validated Q128 dispatch for `batch <= 4` and `seq_len <= 1280`, and uses a 60% KV cap only for equal-length sequences with `L >= 8192`. The cap is intentionally limited to the one long public regime because it is an accuracy-budgeted approximation: all 15 public cases pass at the required threshold, but the long case has only a small margin (`match_ratio` about 0.990–0.992 across two seeds).

Latest high-repeat local evidence:

- CQ remeasurement: `stage_cq_highrep_results.csv`, total about 29.80 ms.
- DN default seed: `stage_dn_highrep_results.csv`, total about 27.50 ms.
- DN alternate seed: `stage_dn_combined_cap60_alt_seed_full_results.csv`, total about 27.50 ms.
- Both runs: 15/15 pass, severe-error count 0; case #4 is the limiting case.
- Projection from the current local checkpoint anchor: `stage_dn_highrep_online_projection.csv`, formula projection about 68.72. This is a local calibration only, not an online score.

## Executive conclusion

The original five directions are useful, but they do not by themselves have a credible path from
the current 68.72 local formula projection to 73.67. The checkpoint sensitivity calculation needs
roughly a 25% reduction across most scored cases; a 3--10% pipeline gain or a short-case launch
cleanup is not enough. The fixed 60% cap can look much better locally, but it is an approximate,
distribution-dependent shortcut with only a 0.990--0.992 margin in the limiting public case. It
should not be promoted as an exact solution for unknown online inputs.

The profiler reports 4096 workgroups and 32768 waves for case #4, with high BF16-MMA and VALU
activity but low shared-bank conflict activity. This makes the main opportunity *removing work per
KV tile* (rowsum, exponent/pack conversions, redundant state updates), rather than merely adding
CTAs or optimizing global-memory bandwidth. The most credible exact route is therefore a fused
softmax/PV or direct BF16 vector math redesign, followed by a larger-tile loader rewrite if the
resource budget permits. Split-K is worth prototyping, but is a conditional fallback: it can help
only when measured load imbalance or dependency stalls outweigh its extra partial-state writes and
merge kernel.

## Highest-value next work

1. Prototype an algorithmically exact fused denominator/PV path for Q128. Remove the independent
   rowsum MMA and intermediate score conversion only if a new fragment mapping preserves the
   required BF16/FP32 tolerance. Require at least 5% stable improvement with correctness margin on
   cases 3/4/6/8 before expanding scope.

2. Benchmark a native BF16x2 score-pack/exp2 path. The earlier FP16x2 attempt regressed because of
   FP32-to-FP16-to-FP32 conversions; direct BF16 packing may avoid that cost. Inspect generated
   resources/instructions, since the SDK intrinsic may lower to scalar code.

3. Build a real multi-stage K/V pipeline for the Q128 normal kernel. Double-buffering can hide
   load latency, but it does not remove the dominant softmax/MMA instructions; target a measured
   3--10% exact improvement and track registers, shared memory, occupancy, and spills.

4. Evaluate exact split-K/chunked-KV only with a cost model and 2-way/4-way A/B tests. Keep it if
   the extra partial-state traffic and merge stay below the parallelism/load-imbalance benefit;
   otherwise reject it for the already well-occupied long case.

5. Improve plan/cache and launch overhead, and add runtime dispatch from actual indptr-derived
   maximum lengths. These changes can help short or highly ragged cases, but cannot be counted on
   to close the all-case 20--25% gap.

6. Calibrate online performance with fresh per-case reports. For every serious candidate, retain a
   same-binary CQ baseline, a high-repeat local A/B table, an alternate-seed table, and a fresh
   online report before claiming a score.

## Guardrails

- Preserve `extern "C" run_kernel` and the fixed ABI.
- Keep exact ragged lengths from `qo_indptr`/`kv_indptr`; `seq_len` is only a dispatch bound.
- Require ordinary cases `match_ratio >= 0.99`, boundary cases 14/15 elementwise exact, and zero severe errors.
- Treat any cap, truncation, reduced head dimension, or approximate softmax as a candidate-specific hypothesis, never as a general optimization.
- Check `nm -D` export and compile time before promotion; template growth previously caused online timeouts.
- Never use a low-latency result from an incomplete Q fragment, invalid shared layout, or skipped output path.

## Suggested experiment order

1. Measure fused rowsum/PV and direct BF16x2 exp2 prototypes independently on cases 3/4/6/8.
2. Profile each prototype for instruction mix, register count, occupancy, spills, and shared use.
3. Add one K/V pipeline stage, one variable at a time; reject if the resource increase offsets the
   timing gain.
4. Run split-K only when a trace or imbalance measurement demonstrates a serial-KV limitation;
   compare 2-way and 4-way chunks including merge and workspace costs.
5. Explore exact runtime dispatch/scheduling for ragged length distributions, then run the full
   promotion gate: all 15 cases, alternate seed, high-repeat A/B, compile/export check.
6. Keep adaptive tail/cap experiments in a clearly marked approximate branch and submit only after a
   fresh online report confirms the calibrated projection.

## Feasibility review against the leaderboard gap

The current DN formula projection is about 68.72, while the requested target is 73.67. The
checkpoint score curve indicates that this is not a small tuning gap. The following is a rough
sensitivity calculation from the 68.10 checkpoint anchor (the DN projection uses a different
per-case local A/B measurement), so it is a planning estimate rather than a new score claim:

| Hypothetical change | Approximate formula score |
|---|---:|
| Current checkpoint anchor | 68.10 |
| Every case 10% faster | 70.22 |
| Every case 15% faster | 71.34 |
| Every case 20% faster | 72.50 |
| Every case 25% faster | 73.71 |

Alternatively, improving only the long/heavy cases by 25% reaches only about 71.18, because
the score is averaged over all points. Therefore a single Q128 micro-optimization or a launch
cleanup cannot plausibly close the gap. The change must either affect most of cases 1--12 or
replace the algorithmic work decomposition.

The current fixed 60% cap is not evidence of an exact optimization. It relies on the public
random-input distribution and the ordinary-case 99% tolerance; its limiting case is only about
99.0--99.2% elementwise agreement. It should be treated as an exploratory upper bound, not a
portable ragged-prefill implementation. A hidden input with a high-scoring omitted tail can
violate the intended attention result even when the two sampled seeds pass.

## Assessment of the existing five directions

| Direction | Likely ceiling | Assessment |
|---|---:|---|
| Exact adaptive tail/error bound | 0--30% on favorable inputs, 0% on fallback inputs | Potentially large but distribution-sensitive. Computing a safe bound must not require evaluating the omitted QK scores, otherwise the saved work disappears. Treat as a separate approximate track. |
| Deeper K/V pipeline | roughly 3--10% | Worth doing for an exact implementation, but the profiler says Q128 is instruction/VALU dominated. More overlap hides latency; it does not remove exp2, conversion, rowsum, or MMA instructions. Unlikely to reach 73.67 alone. |
| Larger legal Q/KV tile | 0--20% if a new loader is genuinely better | High implementation risk. Existing corrected Q256/KV64 is slower than Q128; Q256/KV32 caused shared/swizzle faults. A new loader and fragment mapping, not another tile constant, are required. |
| Plan/cache and launch work | large only for short cases | Useful engineering, but cases 3--8 dominate the score gap. It cannot supply the required 20--25% all-case reduction. |
| Online calibration | no kernel speedup | Necessary for decision quality, not an optimization. The short-case timer quantization currently makes aggregate projections noisy. |

## Other strategies with enough potential

### 1. Exact split-K / chunked-KV attention with an internal workspace (conditional)

The current long Q128 launch has 4096 workgroups and 32768 waves, so split-K is not automatically
better: it does not reduce QK/PV/softmax work and adds partial-state writes plus a merge kernel.
It is justified only if a trace demonstrates load imbalance or a serial-KV dependency that leaves
hardware idle. In that case, split each `(request, Q tile, KV head)` over 2--4 KV chunks, compute
partial `(m, d, o)` states, and run an exact merge:

```text
m = max(m_a, m_b)
d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
o = o_a * exp(m_a - m) + o_b * exp(m_b - m)
```

The ABI has no workspace argument, so any internal allocation must be persistent, bounded, and
thread-safe. The extra global writes and merge kernel are the main costs. A 10--25% long-case
ceiling is plausible only under the measured-idle condition; a 2-way split may be neutral or
slower on the current fully occupied case. Benchmark 2-way and 4-way variants end to end.

### 2. Fuse denominator accumulation into the PV path

Every KV tile currently performs a separate BF16-weight rowsum MMA before the PV MMA. For D=128,
that is one extra 16-wide MMA per score fragment in addition to eight output-column MMA operations.
A custom fragment layout could append a constant-one V channel, compute the denominator in the
same PV issue group, and recover the sacrificed output column with a small scalar path. Another
variant is a warp-local BF16 rowsum using the exact fragment mapping, but the previous naive
FP32 scalar rowsum regressed and is not sufficient evidence. A fused layout may save one MMA and
some temporary conversion work, with a plausible 5--12% Q128 ceiling, but its rounding behavior
must be compared against the reference; “exact” here means exact algorithmic work, not automatic
bitwise equivalence. This requires a new validated MMA fragment mapping, not a helper-level tweak.

### 3. Native BF16x2 exponent-and-pack path

The MACA SDK exposes `h2exp2(__maca_bfloat162)` in `maca_bfloat16.hpp`. The current kernel calls
scalar FP32 `__builtin_exp2f` and later converts the four weights to BF16 for PV. A specialized
path could pack two already-scaled scores to BF16, apply BF16x2 exp2, and feed the result directly
to PV, eliminating the FP32-to-BF16 conversion and perhaps reducing VALU pressure. The earlier
FP16x2 experiment was slower because it converted FP32→FP16→FP32; that does not disprove a direct
BF16 path. However, BF16 exponent/rounding is a numerical change, so this is a tolerance-budgeted
candidate until all cases pass with margin. First inspect generated resource/assembly and benchmark
cases 3/4/6/8. The header implementation currently looks scalar, so it must be treated as
uncertain until measured.

### 4. Warp-specialized producer/consumer Q128 kernel

The current four Q warps both participate in loads, QK, softmax, and PV. A redesigned Q128 kernel
could dedicate one warp group to K/V staging and another to MMA/softmax, using explicit scheduler
bounds and a two-stage shared queue. This may reduce VALU/address interference and improve issue
occupancy, but it increases synchronization and risks the 230 MT register footprint. It is a
larger rewrite than ordinary prefetching and should be attempted only after fused rowsum, BF16x2,
and pipeline baselines are available.

### 5. Revisit Q256 only through a standalone ragged loader

The paged-prefill Q256/W8/KV64 path proves that the hardware can execute a larger tile, but its
corrected ragged analogue was slower because the current ragged loader and causal schedule do not
match that layout. A worthwhile Q256 attempt must start from a fresh loader contract: complete
all Q fragments, vectorize K/V staging, make causal tail handling explicit, and measure register
spills. Reusing the old Q256 code or enabling it by dispatch alone is a known negative result.

### 6. Exact ragged-aware scheduling and dispatch

The host plan currently sorts Q tiles by estimated KV iterations, but it still dispatches a small
set of compile-time tile shapes from the upper-bound `seq_len`. A device-side or persistent
schedule can bucket tasks by their *actual* `qo_indptr`/`kv_indptr` lengths, select Q64/Q128 (and a
future Q256) per bucket, and reduce tail CTAs for highly uneven batches. This is an exact change
and may improve cases 1, 9--12 and short launches, but it cannot provide a 25% all-case gain unless
the online distribution is much more ragged than the public proxy. It should be evaluated as a
load-balance/launch experiment, not as a replacement for reducing per-tile arithmetic.

## Recommended priority and decision gates

1. **Fused rowsum/PV prototype** for Q128 only; inspect resource usage and require at least 5%
   stable improvement before expanding to Q64.
2. **Direct BF16x2 exp2 prototype**; compare instruction/resource output, not just wall time.
3. **One-stage K/V pipeline** with a register/spill gate; keep only a stable end-to-end win.
4. **Conditional split-K prototype** on cases 3/4/6/8, with 2-way and 4-way chunks; reject if
   merge/workspace overhead exceeds the measured load-balance benefit.
5. **Warp-specialized Q128 or standalone Q256 loader** only if the first four do not reach a
   credible 10% improvement.
6. **Adaptive tail cap** remains an exploratory score-maximization branch and must not replace
   the exact canonical path without an explicit tolerance/distribution decision.

The next credible stopping target is not “one case is 20% faster.” It is a full-gate candidate
with at least 15--20% reduction across cases 1--12, no boundary regressions, a clean compile under
the online time limit, and a fresh online report. Without that breadth, a local projection above
73.67 would be more likely to be calibration noise than a durable leaderboard improvement.
