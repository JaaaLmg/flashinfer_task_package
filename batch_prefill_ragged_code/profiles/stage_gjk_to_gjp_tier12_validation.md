# Remaining Tier-1 and Tier-2 validation (2026-08-17)

All online projections in this note use only `online/checkpoint_result` and the
same-source local anchor `stage_gj_anchor_full_results.csv`. The user-reported
checkpoint aggregate is 68.27; its raw per-case mean is 68.717867, so an
uncompressed per-case projection is display-calibrated by -0.447867 points.

| Direction | Candidate / feasibility result | Outcome |
|---|---|---|
| Tier-1 shared transpose K load | `__builtin_mxc_load_shared_trans_8x16` rejected for `--offload-arch=xcore1000`; it compiles for xcore1500 | blocked by submitted target, not attempted in kernel |
| Tier-1 fused denominator PV MMA | no `__builtin_mxc_mma_16x16x32bf16`; BF16 ABI exposes only m16n16k16 | no spare constant-one dimension without losing V output or adding a separate exact pass |
| Tier-2 hierarchical PV | BF16 MMA requires `v4f32` C/D; `v4f16` C/D is a compile-time type error | native FP16 intermediate accumulator is unavailable on C500/xcore1000 |
| Tier-2 score recomputation | GJO Q128: 206 MT registers, 4/4 representative correctness | 1.670/21.527/6.573/6.798 ms for cases 3/4/6/8, versus 1.081/13.684/4.285/4.518 ms anchor; reject |
| Tier-2 Q256 scheduler | Q256/W8/KV64 full 15/15 correctness | 34.985 ms vs 27.296 ms anchor; raw projected mean 62.143, checkpoint-display-calibrated about **61.695**; reject |

The native-ABI probes intentionally fail compilation and are retained with their
exact compiler logs: `stage_gjk_transpose_xcore1000_build.txt`,
`stage_gjn2_fp16_c_fragment_build.txt`, and
`stage_gjp_fused_denominator_mma_build.txt`.
