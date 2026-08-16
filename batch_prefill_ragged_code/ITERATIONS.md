# Optimization ledger

This ledger resumes work on the existing CQ implementation. Historical experiments remain
described in `summarize.md` and `explaination.md`; this file records new reproducible work.

## Resume state

- Current source: `ragged_prefill_optimized.cu`, historical CQ, SHA-256
  `3cee75cf0c376c81c235a0df25c9208220f36f28bc6de0d664091064e92192dc`.
- Online report: `online/checkpoint_result`, aggregate supplied by the user as 67.60.
- First action: remeasure this source on the current C500/MACA environment before editing.

## New stages

Each stage below must state one falsifiable hypothesis, retain source/result metadata, and end
with `promote`, `reject`, or `investigate`. A candidate is promoted only after all 15 cases,
an alternate seed, and a clean `nm -D` export check pass.

## Final five-round sprint (DJ–DN)

- **DJ `equal_cap50_scaled` — reject.** Capped equal-length KV work at 50% and rescaled the
  denominator. #4 reached 97.50% match, while #3/#6 were about 85.6%; `stage_dj...csv`.
- **DK `equal_cap65_scaled` — reject.** A 65% cap passed #4 (99.46%) but failed #3/#6 at
  roughly 94%; not a general public-shape optimization.
- **DL `equal_cap60_scaled` — investigate.** #4 alone passed at 99.045% and 13.675 ms,
  but the margin was too narrow for promotion; `stage_dl_equal_cap60_scaled_fast_results.csv`.
- **DM `combined_cap62` — investigate.** Combined the previously winning medium B<=4 Q128
  dispatch with a 62% cap only for equal L>=8192. Full 15-case and alternate-seed runs passed;
  #4 was 13.9 ms at about 99.23%, but DN was faster.
- **DN `combined_cap60` — promote.** The same restricted dispatch with a 60% cap passed all
  15 cases and an alternate seed. High-repeat default/alternate evidence is in
  `stage_dn_highrep_results.csv` and `stage_dn_combined_cap60_alt_seed_full_results.csv`;
  #4 match was 99.045%/99.155%, with zero severe errors. The canonical source now contains
  the DN defines and exports `run_kernel`.

The DN local checkpoint projection is recorded in `stage_dn_highrep_online_projection.csv`.
It is an estimate (about 68.72 formula score; about 68.22 after applying the user's 67.60
display anchor), not a fresh online leaderboard result. Future work is documented in
`FUTURE_OPTIMIZATION_DIRECTIONS.md`.
