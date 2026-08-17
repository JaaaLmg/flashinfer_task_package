# Ragged prefill: next optimization plan

This document is the working guide for the next optimization round. It deliberately keeps only
directions that can plausibly move the online score; the earlier parameter sweeps are preserved in
`ITERATIONS.md` and `candidates.jsonl` as evidence, not as a tuning checklist.

## Current state and target

The promoted artifact is `ragged_prefill_optimized.cu` (DN), SHA-256
`ca3b0c75f3b9615f11ffb43570296995da3bfedfd981ba5d3ff20204f5b5e1be`. It exports the required
unmangled `run_kernel` ABI and passes the 15-case local correctness gate, including exact cases 14
and 15. The latest same-source full run is `stage_ei_final_full_results.csv`:

| quantity | latest evidence |
|---|---:|
| local candidate total | 27.284 ms |
| cases passed | 15/15 |
| limiting case | #4, match ratio 0.990454 |
| current checkpoint formula projection | 68.747 |
| conservative correction to the user-reported 68.27 display score | about 68.30 |

The projection is not an online result. All projections must use the raw
`online/checkpoint_result` and a local CSV made by the exact submitted source. The historical CL
70.27 result is from an obsolete online baseline and is not evidence for this target.

The requested next milestone is an actual online score above 70. The current score curve implies
roughly an 8--12% reduction across the major scored cases (#1--#12), not merely a 20% win on #4.
The former leaderboard target 73.67 would require approximately 20--25% broad reduction and should
not be used to justify unsafe approximations.

## What is actually limiting performance

The dominant path is Q128, four Q warps, one KV warp, and `NUM_MMA_KV=2` (32 KV rows per tile).
The C500 resource report is approximately 230 MT registers, 48 ST registers, and two resident warps
per PEU. mcProfiler case #4 shows 4096 workgroups and 32768 waves, high BF16-MMA and VALU activity,
and low shared-memory bank-conflict activity. This classifies the kernel as compute/resource
limited, not primarily bandwidth or shared-memory-conflict limited.

Each tile still performs the following independent work:

```text
QK MMA -> logits/mask -> online state update -> denominator rowsum MMA
       -> FP32/BF16 conversion and exponentiation -> PV MMA
```

The separate denominator rowsum, score conversion, exponentiation, and state updates are the only
credible local sources of a double-digit gain. The current fragment layout does not expose a safe
way to remove them. Bigger tiles do not solve this automatically: Q256/W8/KV64 is correct but uses
about 256 MT registers and a 64-byte stack frame, making #4 about 18--21 ms instead of DN's 13.7 ms.

## Priority 1: build a real denominator/PV fusion prototype

This is the highest-value near-term experiment and should start as a microkernel, not as an edit to
the full inlined source.

### Proposed design

1. Define and validate the exact lane ownership of one Q128 score fragment and one PV fragment.
2. Add a constant-one value column to a scratch PV fragment, or use an equivalent packed MMA
   mapping, so the same issue group produces both output accumulation and a denominator.
3. Recover any sacrificed output column with an explicitly measured second path; never silently
   drop a dimension.
4. Compare FP32 state, BF16 PV weights, mask handling, and bottom-right causal tails against
   FlashInfer on adversarial and alternate-seed inputs.
5. Only after the fragment validator passes, integrate it into the Q128 fixed-reference loop.

### Promotion gate

- At least 5% stable reduction on cases 3, 4, 6, and 8 before expanding scope.
- No severe errors; cases 14 and 15 must remain elementwise exact.
- Resource output must not exceed the current register/spill budget.
- Full 15-case gate, alternate seed, clean `nm -D`, and per-case online projection.

A successful fusion has an estimated 5--12% Q128 ceiling. This is the most realistic path to 70+;
it is medium/high difficulty because the compiler's MMA fragment convention is hardware-specific.

## Priority 2: standalone Q256 ragged loader

The paged-prefill work proves that C500 can execute Q256/W8/KV64, but the ragged implementation
cannot reuse that path safely. The current ragged code has a tuned 32-row KV guard and assumes
specific K/V producer layouts. A useful Q256 rewrite must change all of these together:

- Q global/shared/register loading for both Q fragments;
- K/V producer and consumer ownership;
- 64-row shared swizzle and vectorized stores;
- causal tail and ragged indirection;
- double-buffer synchronization and register lifetime.

The retained `ragged_prefill_stage_db_q256_kv64_fixedref.cu` is a reference starting point, not a
promoted candidate. Its resource output (about 256 MT registers plus stack frame) is a failure
condition to eliminate. Do not spend time trying more launch constants around it.

### Q256 success criteria

- Complete fragment-coverage validator for every Q row and KV row;
- no stack spill and preferably below 220 MT registers;
- exact output on cases 3/4/6/8 plus ragged and tail cases;
- at least 8% improvement on the heavy cases before full integration.

This route has the highest potential (possibly 10--25% on long cases) but also the highest risk. It
is the correct fallback if denominator fusion cannot remove a measurable MMA/VALU group.

## Priority 3: GQA cooperative ownership

Eight query heads share one KV head. The current implementation reuses K/V within a CTA, but its
producer/consumer ownership is inherited from the generic layout. A new kernel can dedicate a
producer subgroup to K/V and let eight query-head consumers reuse the same tile and address state.
This is not the rejected W8 constant change: it requires a new shared queue, explicit barriers, and
validated fragment ownership.

Gate this only after measuring a prototype with counters. It is worthwhile if it reduces duplicate
global address/load work without increasing synchronization or register pressure; otherwise reject
it quickly.

## Directions that are closed

Do not revisit these without a new algorithmic premise:

- Q64/Q128/Q256 constant or warp scans;
- Q192 or other non-supported packed query tile sizes;
- Q256/W4/KV32, Q256/W8/KV32, and Q256/W16/KV128 in the current loader;
- BF16x2 exponent helpers (the SDK lowers them to scalar work and prior candidates regressed);
- scalar denominator rowsum replacement;
- split-K under the current ABI (no safe workspace/merge lifetime, and #4 is already highly
  populated);
- plan/cache cleanup as a route to 70 (useful only for short cases);
- lower KV caps or broader truncation (already at the correctness boundary and not exact).

## Required experiment protocol

For every new architecture:

1. Keep `ragged_prefill_baseline.cu` untouched and branch from DN.
2. State one falsifiable bottleneck hypothesis.
3. Compile a self-contained source and verify `nm -D ... | rg ' run_kernel$'`.
4. Run fast cases 3/4/6/8, record resource output, and retain the raw CSV and metadata.
5. Reject immediately on a device fault, wrong fragment coverage, or a regression larger than timer
   noise.
6. Run all 15 cases, alternate seed, and repeatability only for a local best.
7. Project online time per case from the latest checkpoint; never map aggregate local milliseconds
   directly to score.

The submitted hot path must not synchronize the device explicitly, identify test cases, skip
mathematical work, assume hidden indptr values, or use approximate outputs as an optimization.
The true `qo_indptr`/`kv_indptr` lengths and bottom-right causal convention remain authoritative.

## Durable files

After cleanup, the source set intentionally contains only:

- `ragged_prefill_baseline.cu` — immutable correctness/performance anchor;
- `ragged_prefill_optimized.cu` — current DN submission;
- `ragged_prefill_stage_db_q256_kv64_fixedref.cu` — Q256 loader rewrite seed;
- `ragged_prefill_stage_do2_bf16x2_exp_selfcontained.cu` — direct BF16x2 experiment retained as
  an SDK/code-generation reference;
- `ragged_prefill_stage_eg_q128_w8_kv4.cu` — legal alternative warp ownership reference.

The retained measurement evidence is limited to the current DN full-gate CSV and metadata, its
latest checkpoint projection, and the resource profiles cited above. `ITERATIONS.md` and
`candidates.jsonl` retain the conclusions and measurements of rejected candidates; their bulky raw
CSV/metadata artifacts are intentionally pruned. Raw online reports remain. Intermediate shared
libraries are disposable and are not retained.

## Priority 1--3 closure after the next architecture pass (2026-08-17)

The remaining unverified directions were tested against the current C500/MACA environment and did
not produce a promotion candidate:

| priority | candidate | result | decision |
|---|---|---|---|
| 1 | FG actual-wrapper fragment probe; FH KV-subfragment rowsum/PV interleave; FI per-Q rowsum/PV interleave | FG did not preserve the producer-side fragment ABI; FH was only 0.26/0.26/0.41/0.30% faster on #3/#4/#6/#8; FI regressed 0.99/1.00/0.27/0.61% | reject/investigate |
| 2 | FK standalone Q256/KV64 loader reconfirmation | 4/4 correct, but #3/#4/#6/#8 were 1.288/18.207/4.968/5.488 ms versus DN 1.084/13.677/4.289/4.521 ms | reject |
| 3 | FL two-KV-subgroup GQA cooperative probe (`NUM_MMA_KV=1`, two KV producer subgroups) | 4/4 correct, but #3/#4/#6/#8 were 1.091/13.705/4.292/4.528 ms; no stable gain and no evidence of a reusable producer/consumer queue | reject |

Priority 1 therefore still requires a new, actual `lds_v -> permute_64bx4 -> MMA` fragment
validator before any constant-one column can be integrated. Priority 2 requires a fresh loader
rewrite below the Q256 register/stack budget, not another dispatch constant. Priority 3 requires an
explicit shared queue and ownership protocol; merely increasing `NUM_WARPS_KV` is not that design.

No candidate passed the 5%/8% architecture gates or the full promotion gate, so the canonical
artifact remains DN, SHA-256
`ca3b0c75f3b9615f11ffb43570296995da3bfedfd981ba5d3ff20204f5b5e1be`. Its current online anchor is
68.27. The historical CL 70.266667 result is from a different online-baseline generation and its
current-platform submission measured 65.93; it must not be selected or presented as a current
70+ result. This closes this optimization round without claiming an unverified 70+ projection.
